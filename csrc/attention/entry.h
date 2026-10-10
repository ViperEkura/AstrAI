#pragma once
/* Shared by attention entry TUs for tensor checks, partial allocation, and packing. */
#include <cmath>
#include <float.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <api/attention_common.h>
#include <launcher/attention.cuh>

namespace astrai {
namespace attention {

// Micro-checks shared by the packers
inline void check_int32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kInt32, name, " must be a CUDA int32 tensor");
}

/* Kernels use one dtype for Q/K/V/O, dispatched from Q's scalar type. */
inline void
check_qkv_dtype(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "Q/K/V must be on the same CUDA device");
    TORCH_CHECK(k.strides() == v.strides(), "K/V must have identical strides");
    TORCH_CHECK(k.scalar_type() == q.scalar_type(), "K dtype must match Q (", q.scalar_type(),
                "), got ", k.scalar_type());
    TORCH_CHECK(v.scalar_type() == q.scalar_type(), "V dtype must match Q (", q.scalar_type(),
                "), got ", v.scalar_type());
}

/* Resolve the default scale; the entry fills output pointers after packing. */
inline void finish_pack(c10::optional<double> scale, AttentionParams& p) {
    const double value = scale.value_or(1.0 / std::sqrt((double)p.head_dim));
    TORCH_CHECK(std::isfinite(value) && value > 0.0 && std::isfinite((float)value) &&
                    (float)value > 0.0f,
                "native attention scale must be finite and positive");
    p.scale = (float)value;
    p.o_ptr = nullptr;
    p.o_part = nullptr;
    p.ml_part = nullptr;
}

/* Tensor owners are retained by the entry until both kernels are launched. */
struct SplitWorkspace {
    torch::Tensor o_part;
    torch::Tensor ml_part;
};

inline SplitWorkspace resolve_split_buffers(const c10::optional<torch::Tensor>& o_part_buf,
                                            const c10::optional<torch::Tensor>& ml_part_buf,
                                            AttentionParams& p,
                                            const DecodeLaunchPlan& plan,
                                            const torch::Tensor& q) {
    const bool has_o = o_part_buf.has_value() && o_part_buf->defined();
    const bool has_ml = ml_part_buf.has_value() && ml_part_buf->defined();
    TORCH_CHECK(has_o == has_ml, "split buffers must be provided together");
    SplitWorkspace workspace;
    const int64_t o_needed = (int64_t)p.batch * p.q_head * MAX_SPLITS * p.head_dim;
    const int64_t ml_needed = (int64_t)p.batch * p.q_head * MAX_SPLITS * 2;
    if (has_o) {
        for (const auto* buffer_ptr : {&*o_part_buf, &*ml_part_buf}) {
            const auto& buffer = *buffer_ptr;
            TORCH_CHECK(buffer.device() == q.device(), "split buffers must be on Q's device");
            TORCH_CHECK(buffer.scalar_type() == torch::kFloat32 && buffer.is_contiguous(),
                        "split buffers must be contiguous f32 tensors");
        }
        TORCH_CHECK(o_part_buf->numel() >= o_needed, "o_part_buf too small");
        TORCH_CHECK(ml_part_buf->numel() >= ml_needed, "ml_part_buf too small");
        // The optional function arguments already retain these caller buffers.
        p.o_part = o_part_buf->data_ptr<float>();
        p.ml_part = ml_part_buf->data_ptr<float>();
    } else if (!plan.direct_output) {
        const auto options = q.options().dtype(torch::kFloat32);
        workspace.o_part = torch::empty({p.batch, p.q_head, MAX_SPLITS, p.head_dim}, options);
        workspace.ml_part = torch::empty({p.batch, p.q_head, MAX_SPLITS, 2}, options);
    }
    if (workspace.o_part.defined()) {
        p.o_part = workspace.o_part.data_ptr<float>();
        p.ml_part = workspace.ml_part.data_ptr<float>();
    }
    return workspace;
}

/* One dtype/head dispatch owns planning, scratch lifetime, and execution. */
struct DecodeCall {
    AttentionParams& params;
    const torch::Tensor& q;
    const c10::optional<torch::Tensor>& o_part;
    const c10::optional<torch::Tensor>& ml_part;
};

template <typename KV> struct DecodeEntry {
    static void run(DecodeCall& call, cudaStream_t stream) {
        auto& p = call.params;
        with_decode_kernel<KV>(p, [&](auto kernel) {
            const auto plan = kernel.plan(p);
            auto workspace = resolve_split_buffers(call.o_part, call.ml_part, p, plan, call.q);
            kernel.launch(p, plan, stream);
        });
    }
};
template <typename T> using ContiguousDecodeEntry = DecodeEntry<ContigKV<T>>;
template <typename T> using PagedDecodeEntry = DecodeEntry<PagedKV<T>>;

// Shared Q dimensions and stride extraction
inline void extract_q_dims_and_strides(torch::Tensor& q, int64_t layout, AttentionParams& p) {
    TORCH_CHECK(layout == BHLD || layout == BLHD, "unknown attention tensor layout");
    if (layout == BLHD)
        q = q.transpose(1, 2);
    p.batch = (int)q.size(0);
    p.q_head = (int)q.size(1);
    p.q_len = (int)q.size(2);
    p.head_dim = (int)q.size(3);
    p.q_b_stride = (int)q.stride(0);
    p.q_h_stride = (int)q.stride(1);
    p.q_l_stride = (int)q.stride(2);
    p.q_d_stride = (int)q.stride(3);
}

/*
 * ---- Shared mask packing ----
 * Accepts 2D [batch, kv_len], 3D [batch, q_len, kv_len],
 * or 4D [batch, n_heads, q_len, kv_len].
 * Head/q dimensions with size 1 broadcast (stride set to 0).
 */

// Stride of a mask dim that broadcasts when its extent is 1.
inline int bc_stride(const torch::Tensor& m, int dim) {
    return (m.size(dim) == 1) ? 0 : (int)m.stride(dim);
}

inline void set_mask_null(AttentionParams& p) {
    p.mask = nullptr;
    p.mask_b_stride = 0;
    p.mask_h_stride = 0;
    p.mask_l_stride = 0;
    p.mask_k_len = 0;
    p.mask_q_len = 0;
}

inline void pack_mask(const c10::optional<torch::Tensor>& mask,
                      AttentionParams& p,
                      const torch::Device& device,
                      bool paged = false) {
    if (!mask.has_value() || !mask->defined()) {
        set_mask_null(p);
        return;
    }
    const auto& m = mask.value();
    TORCH_CHECK(m.device() == device && m.dtype() == torch::kBool,
                "mask must be bool on Q's device");
    TORCH_CHECK(m.dim() >= 2 && m.dim() <= 4, "mask must be 2D, 3D, or 4D");
    TORCH_CHECK(m.size(0) == 1 || m.size(0) == p.batch, "mask batch mismatch");
    TORCH_CHECK(m.stride(-1) == 1, "mask key dimension must be contiguous");
    const int limit = paged ? p.max_context_len : p.kv_len;
    TORCH_CHECK(paged ? (m.size(-1) > 0 && m.size(-1) <= limit) : m.size(-1) == limit,
                "mask kv_len mismatch");
    p.mask = m.data_ptr<bool>();
    p.mask_b_stride = bc_stride(m, 0);
    p.mask_h_stride = 0;
    p.mask_l_stride = 0;
    p.mask_k_len = (int)m.size(-1);
    p.mask_q_len = 1;
    if (m.dim() == 4) {
        TORCH_CHECK(m.size(1) == 1 || m.size(1) == p.q_head, "mask head mismatch");
        p.mask_h_stride = bc_stride(m, 1);
    }
    if (m.dim() >= 3) {
        const int q_dim = (int)m.dim() - 2;
        TORCH_CHECK(m.size(q_dim) == 1 || (paged ? (m.size(q_dim) > 0 && m.size(q_dim) <= p.q_len)
                                                 : m.size(q_dim) == p.q_len),
                    "mask q_len mismatch");
        p.mask_l_stride = bc_stride(m, q_dim);
        p.mask_q_len = (int)m.size(q_dim);
    }
}

// Contiguous-KV parameter packing
inline void attn_pack_params(torch::Tensor q,
                             torch::Tensor k,
                             torch::Tensor v,
                             c10::optional<torch::Tensor> mask,
                             c10::optional<double> scale,
                             int64_t layout,
                             AttentionParams& p,
                             bool is_causal) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    check_qkv_dtype(q, k, v);
    TORCH_CHECK(k.sizes() == v.sizes(), "K and V must have identical shapes");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "Q/K/V must be 4D");
    extract_q_dims_and_strides(q, layout, p);

    if (layout == BLHD)
        k = k.transpose(1, 2), v = v.transpose(1, 2);

    TORCH_CHECK(k.size(0) == p.batch, "K/V batch must match Q");
    p.kv_head = (int)k.size(1);
    p.kv_len = (int)k.size(2);
    TORCH_CHECK(p.kv_head > 0 && p.q_head > 0 && p.q_head % p.kv_head == 0,
                "q_head must be divisible by kv_head");
    TORCH_CHECK(k.size(3) == p.head_dim, "K/V head_dim must match Q");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
                "Q/K/V head_dim must be contiguous");

    p.kv_b_stride = (int)k.stride(0);
    p.kv_h_stride = (int)k.stride(1);
    p.kv_l_stride = (int)k.stride(2);
    p.kv_d_stride = (int)k.stride(3);

    p.is_causal = is_causal;

    p.q_ptr = q.data_ptr();
    p.k_ptr = k.data_ptr();
    p.v_ptr = v.data_ptr();
    p.new_k_ptr = nullptr;
    p.new_v_ptr = nullptr;
    finish_pack(scale, p);

    pack_mask(mask, p, q.device());
}

/*
 * ---- Paged preamble shared by the decode and prefill packers ----
 * Flat-pool tensor checks, common dim/stride extraction and raw pointers.
 * Batch resolution differs (decode: q rows; prefill: req_pool_indices) and
 * stays at the caller, as do the head_dim granularity checks.
 */
inline void pack_paged_common(torch::Tensor& q,
                              torch::Tensor& k_cache,
                              torch::Tensor& v_cache,
                              torch::Tensor& req_to_token,
                              torch::Tensor& req_pool_indices,
                              torch::Tensor& kv_indptr,
                              AttentionParams& p) {
    check_qkv_dtype(q, k_cache, v_cache);
    check_int32(req_to_token, "req_to_token");
    check_int32(req_pool_indices, "req_pool_indices");
    check_int32(kv_indptr, "kv_indptr");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache must match");
    TORCH_CHECK(k_cache.dim() == 3, "k_cache must be 3D [size, kv_head, head_dim]");
    TORCH_CHECK(q.dim() == 3, "q must be 3D");

    for (const auto* metadata_ptr : {&req_to_token, &req_pool_indices, &kv_indptr}) {
        const auto& metadata = *metadata_ptr;
        TORCH_CHECK(metadata.device() == q.device() && metadata.is_contiguous(),
                    "paged metadata must be contiguous on Q's device");
    }
    TORCH_CHECK(req_to_token.dim() == 2 && req_pool_indices.dim() == 1 && kv_indptr.dim() == 1,
                "invalid paged metadata rank");
    p.q_head = (int)q.size(1);
    p.head_dim = (int)q.size(2);
    p.kv_head = (int)k_cache.size(1);
    TORCH_CHECK(k_cache.size(2) == p.head_dim, "k_cache head_dim mismatch");
    TORCH_CHECK(q.stride(2) == 1 && k_cache.stride(2) == 1 && v_cache.stride(2) == 1,
                "Q/K/V head_dim must be contiguous");
    TORCH_CHECK(p.kv_head > 0 && p.q_head > 0 && p.q_head % p.kv_head == 0,
                "q_head must be divisible by kv_head");

    p.q_l_stride = (int)q.stride(0);
    p.q_h_stride = (int)q.stride(1);
    p.q_d_stride = (int)q.stride(2);

    p.k_ptr = k_cache.data_ptr();
    p.v_ptr = v_cache.data_ptr();
    p.q_ptr = q.data_ptr();
    p.req_to_token = req_to_token.data_ptr<int>();
    p.req_pool_indices = req_pool_indices.data_ptr<int>();
    p.kv_indptr = kv_indptr.data_ptr<int>();
    p.max_context_len = (int)req_to_token.size(1);
}

/*
 * ---- attn_pack_paged_decode_params ----
 * SGLang-style: flat KV pool + req_to_token indexing + variable
 * seq_lens via kv_indptr.  Q is [batch, q_head, head_dim] (q_len=1 per req).
 */
inline void attn_pack_paged_decode_params(torch::Tensor q,
                                          torch::Tensor k_cache,
                                          torch::Tensor v_cache,
                                          torch::Tensor req_to_token,
                                          torch::Tensor req_pool_indices,
                                          torch::Tensor kv_indptr,
                                          const c10::optional<torch::Tensor>& new_k,
                                          const c10::optional<torch::Tensor>& new_v,
                                          c10::optional<torch::Tensor> mask,
                                          c10::optional<double> scale,
                                          AttentionParams& p,
                                          bool is_causal) {
    pack_paged_common(q, k_cache, v_cache, req_to_token, req_pool_indices, kv_indptr, p);
    p.batch = (int)q.size(0);
    p.q_len = 1;
    TORCH_CHECK(req_pool_indices.size(0) == p.batch && kv_indptr.size(0) == p.batch + 1,
                "decode metadata batch mismatch");
    TORCH_CHECK(p.head_dim % 32 == 0, "head_dim must be multiple of 32");
    p.qo_indptr = nullptr;

    TORCH_CHECK(new_k.has_value() == new_v.has_value(),
                "new_k and new_v must be provided together");
    if (new_k.has_value()) {
        auto nk = new_k.value();
        auto nv = new_v.value();
        TORCH_CHECK(nk.device() == q.device() && nv.device() == q.device(),
                    "new K/V must be CUDA tensors");
        TORCH_CHECK(nk.scalar_type() == q.scalar_type() && nv.scalar_type() == q.scalar_type(),
                    "new K/V dtype must match Q");
        TORCH_CHECK(nk.dim() == 3 && nv.dim() == 3,
                    "new K/V must be 3D [batch, kv_head, head_dim]");
        TORCH_CHECK(nk.sizes() == nv.sizes(), "new K and V must have identical shapes");
        TORCH_CHECK(nk.strides() == nv.strides(), "new K and V must have identical strides");
        TORCH_CHECK(nk.size(0) == p.batch && nk.size(1) == p.kv_head && nk.size(2) == p.head_dim,
                    "new K/V shape mismatch");
        TORCH_CHECK(nk.stride(2) == 1 && nv.stride(2) == 1, "new K/V head_dim must be contiguous");
        p.new_k_ptr = nk.data_ptr();
        p.new_v_ptr = nv.data_ptr();
        p.new_kv_b_stride = (int)nk.stride(0);
        p.new_kv_h_stride = (int)nk.stride(1);
    } else {
        p.new_k_ptr = nullptr;
        p.new_v_ptr = nullptr;
        p.new_kv_b_stride = p.new_kv_h_stride = 0;
    }

    p.is_causal = is_causal;

    pack_mask(mask, p, q.device(), true);
    finish_pack(scale, p);
}

/*
 * ---- attn_pack_paged_prefill_params ----
 * SGLang-style: flat KV pool + req_to_token + ragged batch via qo_indptr.
 * Q is [total_q, q_head, head_dim] (flattened across all requests).
 */
inline void attn_pack_paged_prefill_params(torch::Tensor q,
                                           torch::Tensor k_cache,
                                           torch::Tensor v_cache,
                                           torch::Tensor req_to_token,
                                           torch::Tensor req_pool_indices,
                                           torch::Tensor kv_indptr,
                                           torch::Tensor qo_indptr,
                                           torch::Tensor q_tile_to_batch,
                                           torch::Tensor q_tile_to_index,
                                           c10::optional<torch::Tensor> mask,
                                           c10::optional<double> scale,
                                           AttentionParams& p,
                                           bool is_causal) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    check_int32(qo_indptr, "qo_indptr");
    check_int32(q_tile_to_batch, "q_tile_to_batch");
    check_int32(q_tile_to_index, "q_tile_to_index");
    pack_paged_common(q, k_cache, v_cache, req_to_token, req_pool_indices, kv_indptr, p);
    for (const auto* metadata_ptr : {&qo_indptr, &q_tile_to_batch, &q_tile_to_index}) {
        const auto& metadata = *metadata_ptr;
        TORCH_CHECK(metadata.device() == q.device() && metadata.is_contiguous(),
                    "prefill metadata must be contiguous on Q's device");
    }

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

    p.is_causal = is_causal;
    pack_mask(mask, p, q.device(), true);
    finish_pack(scale, p);
}

} // namespace attention
} // namespace astrai
