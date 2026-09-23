#pragma once
#include <float.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "common.h"

// Dispatch head_dim: shared macro — avoids C++20 lambda template syntax.
// Usage: DISPATCH_HEAD_DIM(hd, fn, args...)
//   Expands to: fn<32>(args...); fn<64>(args...); etc.
#define DISPATCH_HEAD_DIM(hd, fn, ...) \
    switch (hd) { \
        case 32:  fn<32>(__VA_ARGS__); break; \
        case 64:  fn<64>(__VA_ARGS__); break; \
        case 128: fn<128>(__VA_ARGS__); break; \
        case 256: fn<256>(__VA_ARGS__); break; \
        default: \
            TORCH_CHECK(false, "unsupported head_dim ", hd, \
                         " (supported: 32, 64, 128, 256)"); \
    }

namespace astrai {
namespace attention {

using bf16 = __nv_bfloat16;

// ---- Micro-checks shared by the packers ----
inline void check_bf16(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kBFloat16,
                name, " must be a CUDA bf16 tensor");
}

inline void check_int32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kInt32,
                name, " must be a CUDA int32 tensor");
}

// Scalar knobs every packer shares: the causal flag and the mask-present
// bit (an optional holding an undefined tensor counts as "no mask").
template <typename P>
inline void start_pack(const c10::optional<torch::Tensor>& mask,
                       int64_t causal_offset, P& p) {
    p.causal_offset = (int)causal_offset;
    p.use_mask = (mask.has_value() && mask.value().defined()) ? 1 : 0;
}

// Default scale resolution + null output pointers; the entry .cu fills
// o_ptr / split partials right after packing.
template <typename P>
inline void finish_pack(double scale, P& p) {
    p.scale = (scale > 0.0) ? (float)scale : 1.0f / sqrtf((float)p.head_dim);
    p.o_ptr = nullptr;
    p.o_part = nullptr;
    p.ml_part = nullptr;
}

// The split kernel unconditionally writes every (batch, q_head, split) slot it
// owns — including empty split ranges, which store m = -FLT_MAX so the combine
// skips them. Allocators are therefore left uninitialized (torch::empty); the
// per-call memset (torch::zeros / torch::full) was pure overhead.
template<typename P>
inline void alloc_split_partials(P& p) {
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto o_part = torch::empty(at::IntArrayRef{p.batch, p.q_head, MAX_SPLITS, p.head_dim}, fopt);
    auto ml_part = torch::empty(at::IntArrayRef{p.batch, p.q_head, MAX_SPLITS, 2}, fopt);
    p.o_part = (float*)o_part.data_ptr();
    p.ml_part = (float*)ml_part.data_ptr();
}

// Split partials: validate caller-provided buffers or allocate fresh ones.
template <typename P>
inline void resolve_split_buffers(
    const c10::optional<torch::Tensor>& o_part_buf,
    const c10::optional<torch::Tensor>& ml_part_buf,
    P& p) {
    if (o_part_buf.has_value() && ml_part_buf.has_value()
        && o_part_buf->defined() && ml_part_buf->defined()) {
        TORCH_CHECK(o_part_buf->scalar_type() == torch::kFloat32, "o_part_buf must be f32");
        TORCH_CHECK(ml_part_buf->scalar_type() == torch::kFloat32, "ml_part_buf must be f32");
        int64_t o_needed = (int64_t)p.batch * p.q_head * MAX_SPLITS * p.head_dim;
        int64_t ml_needed = (int64_t)p.batch * p.q_head * MAX_SPLITS * 2;
        TORCH_CHECK(o_part_buf->numel() >= o_needed,
                     "o_part_buf too small: need ", o_needed, " got ", o_part_buf->numel());
        TORCH_CHECK(ml_part_buf->numel() >= ml_needed,
                     "ml_part_buf too small: need ", ml_needed, " got ", ml_part_buf->numel());
        TORCH_CHECK(o_part_buf->is_cuda() && ml_part_buf->is_cuda(),
                    "split buffers must be CUDA tensors");
        TORCH_CHECK(o_part_buf->is_contiguous() && ml_part_buf->is_contiguous(),
                    "split buffers must be contiguous");
        p.o_part = (float*)o_part_buf->data_ptr();
        p.ml_part = (float*)ml_part_buf->data_ptr();
    } else {
        alloc_split_partials(p);
    }
}

// ---- Shared Q-dims + strides extraction ----
template <typename P>
inline void extract_q_dims_and_strides(torch::Tensor& q, int64_t layout, P& p) {
    if (layout == BLHD) q = q.transpose(1, 2);
    p.batch = (int)q.size(0);
    p.q_head = (int)q.size(1);
    p.q_len = (int)q.size(2);
    p.head_dim = (int)q.size(3);
    p.q_b_stride = (int)q.stride(0);
    p.q_h_stride = (int)q.stride(1);
    p.q_l_stride = (int)q.stride(2);
    p.q_d_stride = (int)q.stride(3);
}

// ---- Shared mask packing ----
// Accepts 2D [batch, kv_len], 3D [batch, q_len, kv_len],
// or 4D [batch, n_heads, q_len, kv_len].
// Head/q dimensions with size 1 broadcast (stride set to 0).

// Stride of a mask dim that broadcasts when its extent is 1.
inline int bc_stride(const torch::Tensor& m, int dim) {
    return (m.size(dim) == 1) ? 0 : (int)m.stride(dim);
}

// 2D masks: batch stride only, head/q strides broadcast away.
template <typename P>
inline void set_mask_2d(const torch::Tensor& m, P& p) {
    p.mask_b_stride = (int)m.stride(0);
    p.mask_h_stride = 0;
    p.mask_l_stride = 0;
    p.mask = m.data_ptr<bool>();
}

template <typename P>
inline void set_mask_null(P& p) {
    p.mask = nullptr;
    p.mask_b_stride = 0;
    p.mask_h_stride = 0;
    p.mask_l_stride = 0;
}

template <typename P>
inline void pack_mask(const c10::optional<torch::Tensor>& mask, P& p) {
    if (p.use_mask) {
        auto m = mask.value();
        TORCH_CHECK(m.is_cuda(), "mask must be on CUDA");
        TORCH_CHECK(m.dtype() == torch::kBool, "mask must be bool");
        TORCH_CHECK(m.size(0) == p.batch, "mask batch mismatch");
        TORCH_CHECK(m.size(m.dim() - 1) == p.kv_len, "mask kv_len mismatch");
        if (m.dim() == 2) {
            set_mask_2d(m, p);
        } else if (m.dim() == 3) {
            TORCH_CHECK(m.size(1) == 1 || m.size(1) == p.q_len, "mask q_len mismatch");
            p.mask_b_stride = (int)m.stride(0);
            p.mask_h_stride = 0;
            p.mask_l_stride = bc_stride(m, 1);
            p.mask = m.data_ptr<bool>();
        } else if (m.dim() == 4) {
            TORCH_CHECK(m.size(2) == 1 || m.size(2) == p.q_len, "mask q_len mismatch");
            p.mask_b_stride = (int)m.stride(0);
            p.mask_h_stride = bc_stride(m, 1);
            p.mask_l_stride = bc_stride(m, 2);
            p.mask = m.data_ptr<bool>();
        } else {
            TORCH_CHECK(false, "mask must be 2D, 3D, or 4D");
        }
    } else {
        set_mask_null(p);
    }
}

// ---- attn_pack_params (contiguous KV) ----
template<typename T>
inline void attn_pack_params(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    int64_t layout,
    AttentionParams<T>& p
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    check_bf16(q, "q");
    check_bf16(k, "k");
    check_bf16(v, "v");
    TORCH_CHECK(k.sizes() == v.sizes(), "K and V must have identical shapes");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "Q/K/V must be 4D");
    extract_q_dims_and_strides(q, layout, p);

    if (layout == BLHD) k = k.transpose(1, 2), v = v.transpose(1, 2);

    p.kv_head = (int)k.size(1);
    p.kv_len = (int)k.size(2);
    TORCH_CHECK(p.q_head % p.kv_head == 0,
                "q_head must be divisible by kv_head");
    TORCH_CHECK(k.size(3) == p.head_dim, "K/V head_dim must match Q");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
                "Q/K/V head_dim must be contiguous");

    p.kv_b_stride = (int)k.stride(0);
    p.kv_h_stride = (int)k.stride(1);
    p.kv_l_stride = (int)k.stride(2);
    p.kv_d_stride = (int)k.stride(3);

    start_pack(mask, causal_offset, p);

    p.q_ptr = (const T*)q.data_ptr();
    p.k_ptr = (const T*)k.data_ptr();
    p.v_ptr = (const T*)v.data_ptr();
    p.new_k_ptr = nullptr;
    p.new_v_ptr = nullptr;
    finish_pack(scale, p);

    pack_mask(mask, p);
}

// ---- Paged preamble shared by the decode and prefill packers ----
// Flat-pool tensor checks, common dim/stride extraction and raw pointers.
// Batch resolution differs (decode: q rows; prefill: req_pool_indices) and
// stays at the caller, as do the head_dim granularity checks.
template <typename T>
inline void pack_paged_common(
    torch::Tensor& q,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& req_to_token,
    torch::Tensor& req_pool_indices,
    torch::Tensor& kv_indptr,
    AttentionParams<T>& p)
{
    check_bf16(q, "q");
    check_bf16(k_cache, "k_cache");
    check_bf16(v_cache, "v_cache");
    check_int32(req_to_token, "req_to_token");
    check_int32(req_pool_indices, "req_pool_indices");
    check_int32(kv_indptr, "kv_indptr");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache must match");
    TORCH_CHECK(k_cache.dim() == 3, "k_cache must be 3D [size, kv_head, head_dim]");
    TORCH_CHECK(q.dim() == 3, "q must be 3D");

    p.q_head = (int)q.size(1);
    p.head_dim = (int)q.size(2);
    p.kv_head = (int)k_cache.size(1);
    TORCH_CHECK(k_cache.size(2) == p.head_dim, "k_cache head_dim mismatch");
    TORCH_CHECK(q.stride(2) == 1 && k_cache.stride(2) == 1 && v_cache.stride(2) == 1,
                "Q/K/V head_dim must be contiguous");
    TORCH_CHECK(p.q_head % p.kv_head == 0, "q_head must be divisible by kv_head");

    p.q_l_stride = (int)q.stride(0);
    p.q_h_stride = (int)q.stride(1);
    p.q_d_stride = (int)q.stride(2);

    p.k_ptr = (const T*)k_cache.data_ptr();
    p.v_ptr = (const T*)v_cache.data_ptr();
    p.q_ptr = (const T*)q.data_ptr();
    p.req_to_token = req_to_token.data_ptr<int>();
    p.req_pool_indices = req_pool_indices.data_ptr<int>();
    p.kv_indptr = kv_indptr.data_ptr<int>();
    p.max_context_len = (int)req_to_token.size(1);
}

// ---- attn_pack_paged_decode_params ----
// SGLang-style: flat KV pool + req_to_token indexing + variable
// seq_lens via kv_indptr.  Q is [batch, q_head, head_dim] (q_len=1 per req).
template<typename T>
inline void attn_pack_paged_decode_params(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor kv_indptr,
    const c10::optional<torch::Tensor>& new_k,
    const c10::optional<torch::Tensor>& new_v,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    AttentionParams<T>& p
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    pack_paged_common(q, k_cache, v_cache, req_to_token, req_pool_indices,
                      kv_indptr, p);
    p.batch = (int)q.size(0);
    TORCH_CHECK(p.head_dim % 32 == 0, "head_dim must be multiple of 32");
    p.qo_indptr = nullptr;

    TORCH_CHECK(new_k.has_value() == new_v.has_value(),
                "new_k and new_v must be provided together");
    if (new_k.has_value()) {
        auto nk = new_k.value();
        auto nv = new_v.value();
        TORCH_CHECK(nk.is_cuda() && nv.is_cuda(), "new K/V must be CUDA tensors");
        TORCH_CHECK(nk.dtype() == torch::kBFloat16 && nv.dtype() == torch::kBFloat16,
                    "new K/V must be bf16");
        TORCH_CHECK(nk.dim() == 3 && nv.dim() == 3,
                    "new K/V must be 3D [batch, kv_head, head_dim]");
        TORCH_CHECK(nk.sizes() == nv.sizes(), "new K and V must have identical shapes");
        TORCH_CHECK(nk.strides() == nv.strides(),
                    "new K and V must have identical strides");
        TORCH_CHECK(nk.size(0) == p.batch && nk.size(1) == p.kv_head
                    && nk.size(2) == p.head_dim, "new K/V shape mismatch");
        TORCH_CHECK(nk.stride(2) == 1 && nv.stride(2) == 1,
                    "new K/V head_dim must be contiguous");
        p.new_k_ptr = (const T*)nk.data_ptr();
        p.new_v_ptr = (const T*)nv.data_ptr();
        p.new_kv_b_stride = (int)nk.stride(0);
        p.new_kv_h_stride = (int)nk.stride(1);
    } else {
        p.new_k_ptr = nullptr;
        p.new_v_ptr = nullptr;
        p.new_kv_b_stride = p.new_kv_h_stride = 0;
    }

    start_pack(mask, causal_offset, p);

    if (p.use_mask) {
        auto m = mask.value();
        TORCH_CHECK(m.is_cuda() && m.dtype() == torch::kBool, "mask must be bool CUDA");
        TORCH_CHECK(m.size(0) == p.batch, "mask batch mismatch");
        set_mask_2d(m, p);
    } else {
        set_mask_null(p);
    }
    finish_pack(scale, p);
}

// ---- attn_pack_paged_prefill_params ----
// SGLang-style: flat KV pool + req_to_token + ragged batch via qo_indptr.
// Q is [total_q, q_head, head_dim] (flattened across all requests).
template<typename T>
inline void attn_pack_paged_prefill_params(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor kv_indptr,
    torch::Tensor qo_indptr,
    torch::Tensor q_tile_to_batch,
    torch::Tensor q_tile_to_index,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    AttentionParams<T>& p
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    check_int32(qo_indptr, "qo_indptr");
    check_int32(q_tile_to_batch, "q_tile_to_batch");
    check_int32(q_tile_to_index, "q_tile_to_index");
    pack_paged_common(q, k_cache, v_cache, req_to_token, req_pool_indices,
                      kv_indptr, p);

    p.q_len = (int)q.size(0);
    p.batch = (int)req_pool_indices.size(0);
    TORCH_CHECK(p.head_dim % 16 == 0, "head_dim must be multiple of 16");
    TORCH_CHECK(kv_indptr.size(0) == p.batch + 1, "kv_indptr must be [batch+1]");
    TORCH_CHECK(qo_indptr.size(0) == p.batch + 1, "qo_indptr must be [batch+1]");
    TORCH_CHECK(q_tile_to_batch.dim() == 1 && q_tile_to_index.dim() == 1,
                "Q tile mappings must be 1D");
    TORCH_CHECK(q_tile_to_batch.size(0) == q_tile_to_index.size(0),
                "Q tile mappings must have equal length");

    p.new_k_ptr = nullptr;
    p.new_v_ptr = nullptr;
    p.qo_indptr = qo_indptr.data_ptr<int>();
    p.q_tile_to_batch = q_tile_to_batch.data_ptr<int>();
    p.q_tile_to_index = q_tile_to_index.data_ptr<int>();
    p.num_q_tiles = (int)q_tile_to_batch.size(0);

    start_pack(mask, causal_offset, p);
    if (p.use_mask) {
        auto m = mask.value();
        TORCH_CHECK(m.is_cuda() && m.dtype() == torch::kBool, "mask must be bool CUDA");
        TORCH_CHECK(m.size(0) == p.batch, "mask batch mismatch");
        if (m.dim() == 2) {
            TORCH_CHECK(m.size(1) <= p.max_context_len, "mask kv_len mismatch");
            set_mask_2d(m, p);
        } else if (m.dim() == 4) {
            TORCH_CHECK(m.size(1) == 1 || m.size(1) == p.q_head, "mask head mismatch");
            TORCH_CHECK(m.size(2) > 0 && m.size(2) <= p.q_len, "mask q_len mismatch");
            TORCH_CHECK(m.size(3) <= p.max_context_len, "mask kv_len mismatch");
            p.mask_b_stride = (int)m.stride(0);
            p.mask_h_stride = bc_stride(m, 1);
            p.mask_l_stride = bc_stride(m, 2);
            p.mask = m.data_ptr<bool>();
        } else {
            TORCH_CHECK(false, "mask must be 2D or 4D");
        }
    } else {
        set_mask_null(p);
    }
    finish_pack(scale, p);
}

}  // namespace attention
}  // namespace astrai
