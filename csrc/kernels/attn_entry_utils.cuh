#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "attn_common.h"

using bf16 = __nv_bfloat16;

// ---------------------------------------------------------------------------
// Shared dispatch helpers — eliminates duplication across .cu entry files.
// ---------------------------------------------------------------------------

inline int compute_num_splits(int base_blocks, int tiles_total) {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int n = (2 * sm_count + base_blocks - 1) / base_blocks;
    return std::max(1, std::min(n, std::min(tiles_total, 32)));
}

// Dispatch head_dim to a generic lambda FN (callable as FN.operator()<D>()).
template <typename Fn>
inline void dispatch_head_dim(int hd, Fn&& fn) {
    switch (hd) {
        case 32:  fn.template operator()<32>();  break;
        case 64:  fn.template operator()<64>();  break;
        case 128: fn.template operator()<128>(); break;
        case 256: fn.template operator()<256>(); break;
        default:
            TORCH_CHECK(false, "unsupported head_dim ", hd,
                         " (supported: 32, 64, 128, 256)");
    }
}

// Allocate split-KV partial buffers and wire into params.
template<typename P>
inline void alloc_split_partials(P& p) {
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto o_part = torch::empty({p.batch, p.q_head, p.num_splits, p.head_dim}, fopt);
    auto ml_part = torch::empty({p.batch, p.q_head, p.num_splits, 2}, fopt);
    p.o_part = (float*)o_part.data_ptr();
    p.ml_part = (float*)ml_part.data_ptr();
}

// ---------------------------------------------------------------------------
// Param packing — fills AttentionParams from torch tensors with validation.
// ---------------------------------------------------------------------------

template<typename T>
inline void attn_pack_params(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    int64_t layout,   // 0 = b h l d, 1 = b l h d
    AttentionParams<T>& p
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(k.dtype() == torch::kBFloat16);
    TORCH_CHECK(v.dtype() == torch::kBFloat16);
    TORCH_CHECK(k.sizes() == v.sizes(), "K and V must have identical shapes");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "Q/K/V must be 4D");

    // Normalize to b h l d view (zero-copy transpose if user passed b l h d)
    if (layout == 1) {
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);
    }

    p.batch = (int)q.size(0);
    p.q_head = (int)q.size(1);
    p.q_len = (int)q.size(2);
    p.head_dim = (int)q.size(3);
    p.kv_head = (int)k.size(1);
    p.kv_len = (int)k.size(2);
    TORCH_CHECK(k.size(3) == p.head_dim, "K/V head_dim must match Q");

    // Strides (layout-agnostic: works for b h l d and b l h d)
    p.q_stride_b = (int)q.stride(0);
    p.q_stride_h = (int)q.stride(1);
    p.q_stride_l = (int)q.stride(2);
    p.q_stride_d = (int)q.stride(3);
    p.kv_stride_b = (int)k.stride(0);
    p.kv_stride_h = (int)k.stride(1);
    p.kv_stride_l = (int)k.stride(2);
    p.kv_stride_d = (int)k.stride(3);

    p.causal_offset = (int)causal_offset;
    p.use_mask = mask.has_value() ? 1 : 0;
    p.scale = (scale > 0.0) ? (float)scale : 1.0f / sqrtf((float)p.head_dim);

    p.q = (const T*)q.data_ptr();
    p.k = (const T*)k.data_ptr();
    p.v = (const T*)v.data_ptr();
    p.o = nullptr;
    p.o_part = nullptr;
    p.ml_part = nullptr;

    if (p.use_mask) {
        auto m = mask.value();
        TORCH_CHECK(m.is_cuda(), "mask must be on CUDA");
        TORCH_CHECK(m.dtype() == torch::kBool, "mask must be bool");
        TORCH_CHECK(m.size(0) == p.batch, "mask batch mismatch");
        TORCH_CHECK(m.size(m.dim() - 1) == p.kv_len, "mask kv_len mismatch");
        if (m.dim() == 2) {
            p.mask_b_stride = (int)m.stride(0);
            p.mask_q_stride = 0;
        } else if (m.dim() == 3) {
            TORCH_CHECK(m.size(1) == p.q_len, "mask q_len mismatch");
            p.mask_b_stride = (int)m.stride(0);
            p.mask_q_stride = (int)m.stride(1);
        } else {
            TORCH_CHECK(false, "mask must be 2D [batch, kv_len] or 3D [batch, q_len, kv_len]");
        }
        p.mask = m.data_ptr<bool>();
    } else {
        p.mask = nullptr;
        p.mask_b_stride = 0;
        p.mask_q_stride = 0;
    }
}

// ---------------------------------------------------------------------------
// Param packing for paged attention.
// ---------------------------------------------------------------------------

template<typename T>
inline void attn_pack_paged_params(
    torch::Tensor q,
    torch::Tensor page_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t page_size,
    int64_t kv_len,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    int64_t layout,   // 0 = b h l d, 1 = b l h d
    PagedAttentionParams<T>& p
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    TORCH_CHECK(q.is_cuda() && page_table.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k_cache.dtype() == torch::kBFloat16, "k_cache must be bf16");
    TORCH_CHECK(v_cache.dtype() == torch::kBFloat16, "v_cache must be bf16");
    TORCH_CHECK(page_table.dtype() == torch::kLong, "page_table must be int64");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache must have identical shapes");

    // Normalize Q to b h l d view if user passed b l h d
    if (layout == 1) {
        q = q.transpose(1, 2);
    }

    p.batch = (int)q.size(0);
    p.q_head = (int)q.size(1);
    p.q_len = (int)q.size(2);
    p.head_dim = (int)q.size(3);
    p.kv_head = (int)k_cache.size(2);
    p.kv_len = (int)kv_len;
    p.page_size = (int)page_size;
    p.max_pages = (int)page_table.size(1);

    TORCH_CHECK(q.size(2) == 1, "Q seq_len must be 1 (decode)");
    TORCH_CHECK(p.head_dim % 32 == 0, "head_dim must be multiple of 32");
    TORCH_CHECK(k_cache.size(1) == page_size,
                "k_cache dim 1 must equal page_size, got ",
                k_cache.size(1), " vs ", page_size);

    // Q strides
    p.q_stride_b = (int)q.stride(0);
    p.q_stride_h = (int)q.stride(1);
    p.q_stride_l = (int)q.stride(2);
    p.q_stride_d = (int)q.stride(3);

    p.causal_offset = (int)causal_offset;
    p.use_mask = (mask.has_value() && mask.value().defined()) ? 1 : 0;
    p.scale = (scale > 0.0) ? (float)scale : 1.0f / sqrtf((float)p.head_dim);

    p.page_table = page_table.data_ptr<int64_t>();
    p.k_cache = (const T*)k_cache.data_ptr();
    p.v_cache = (const T*)v_cache.data_ptr();
    p.q = (const T*)q.data_ptr();
    p.o = nullptr;
    p.o_part = nullptr;
    p.ml_part = nullptr;

    if (p.use_mask) {
        auto m = mask.value();
        TORCH_CHECK(m.is_cuda(), "mask must be on CUDA");
        TORCH_CHECK(m.dtype() == torch::kBool, "mask must be bool");
        TORCH_CHECK(m.size(0) == p.batch, "mask batch mismatch");
        TORCH_CHECK(m.size(m.dim() - 1) == p.kv_len, "mask kv_len mismatch");
        if (m.dim() == 2) {
            p.mask_b_stride = (int)m.stride(0);
            p.mask_q_stride = 0;
        } else if (m.dim() == 3) {
            TORCH_CHECK(m.size(1) == p.q_len, "mask q_len mismatch");
            p.mask_b_stride = (int)m.stride(0);
            p.mask_q_stride = (int)m.stride(1);
        } else {
            TORCH_CHECK(false, "mask must be 2D [batch, kv_len] or 3D [batch, q_len, kv_len]");
        }
        p.mask = m.data_ptr<bool>();
    } else {
        p.mask = nullptr;
        p.mask_b_stride = 0;
        p.mask_q_stride = 0;
    }
}
