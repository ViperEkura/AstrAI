#pragma once
/* Tensor checks and packing shared by the native attention entries. */
#include <cmath>
#include <float.h>
#include <limits>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <api/native/attention.h>
#include <api/torch/dispatch.cuh>
#include <launcher/attention.cuh>

namespace astrai {
namespace attention {

// Q, mask, and contiguous KV bases use int offsets inside the kernels.
inline void check_int32_offset(const torch::Tensor& tensor, const char* name, int dimensions) {
    int64_t last_offset = 0;
    for (int dim = 0; dim < dimensions; ++dim) {
        const int64_t size = tensor.size(dim);
        const int64_t stride = tensor.stride(dim);
        TORCH_CHECK(size <= std::numeric_limits<int>::max() && stride >= 0 &&
                        stride <= std::numeric_limits<int>::max(),
                    name, " shape or stride exceeds int32 range");
        const int64_t last_index = size - 1;
        if (last_index <= 0)
            continue;
        TORCH_CHECK(stride <= (std::numeric_limits<int>::max() - last_offset) / last_index, name,
                    " indexed offset exceeds int32 range");
        last_offset += last_index * stride;
    }
}

/* Supported kernel element types, shared by the three Torch entries. */
using AttentionTypes = TypeList<bf16, fp16>;

/* Kernels use one dtype for Q/K/V/O, dispatched from Q's scalar type. */
inline void
check_qkv_dtype(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "Q/K/V must be on the same CUDA device");
    TORCH_CHECK(k.scalar_type() == q.scalar_type(), "K dtype must match Q (", q.scalar_type(),
                "), got ", k.scalar_type());
    TORCH_CHECK(v.scalar_type() == q.scalar_type(), "V dtype must match Q (", q.scalar_type(),
                "), got ", v.scalar_type());
    visit_dtype<AttentionTypes>(
        q.scalar_type(), [](auto) {},
        [](at::ScalarType dtype) {
            return unsupported_dtype_message<AttentionTypes>("attention", dtype);
        });
}

// K and V use one stride tuple in the kernel. MLA can project V as an interleaved view.
inline void match_kv_layout(torch::Tensor& k, torch::Tensor& v) {
    if (k.strides() != v.strides()) {
        k = k.contiguous();
        v = v.contiguous();
    }
}

/* Resolve the default scale; the entry fills output pointers after packing. */
inline void finish_pack(c10::optional<double> scale, AttentionParams& p) {
    const double value = scale.value_or(1.0 / std::sqrt((double)p.head_dim));
    TORCH_CHECK(std::isfinite(value) && value > 0.0 && std::isfinite((float)value) &&
                    (float)value > 0.0f,
                "native attention scale must be finite and positive");
    TORCH_CHECK(p.batch > 0 && p.q_head > 0 && p.kv_head > 0 && p.q_len > 0,
                "attention batch, heads, and query length must be positive");
    TORCH_CHECK(static_cast<int64_t>(p.batch) * p.q_head <= std::numeric_limits<int>::max(),
                "attention batch * q_head exceeds int32 range");
    detail::with_head_dim(p.head_dim, [](auto) {});
    p.scale = (float)value;
    p.o_ptr = nullptr;
    p.o_part = nullptr;
    p.ml_part = nullptr;
}

inline void set_kv_split_size(AttentionParams& p, int64_t size) {
    TORCH_CHECK(size >= 0 && size <= std::numeric_limits<int>::max(), "invalid KV split size");
    p.kv_split_size = static_cast<int>(size);
    TORCH_CHECK(p.kv_split_size % 16 == 0, "KV split size must be a multiple of 16");
}

/*
 * ---- Shared mask packing ----
 * Accepts 2D [batch, kv_len], 3D [batch, q_len, kv_len],
 * or 4D [batch, n_heads, q_len, kv_len].
 * Head/q dimensions with size 1 broadcast (stride set to 0).
 */

inline void pack_mask(const c10::optional<torch::Tensor>& mask,
                      AttentionParams& p,
                      const torch::Device& device,
                      bool paged = false) {
    p.mask = nullptr;
    p.mask_q_size = 1;
    p.mask_k_size = 0;
    p.mask_b_stride = p.mask_h_stride = p.mask_q_stride = 0;
    if (!mask.has_value() || !mask->defined())
        return;
    const auto& m = mask.value();
    TORCH_CHECK(m.device() == device && m.dtype() == torch::kBool,
                "mask must be bool on Q's device");
    TORCH_CHECK(m.dim() >= 2 && m.dim() <= 4, "mask must be 2D, 3D, or 4D");
    TORCH_CHECK(m.size(0) == 1 || m.size(0) == p.batch, "mask batch mismatch");
    TORCH_CHECK(m.stride(-1) == 1, "mask key dimension must be contiguous");
    const int limit = p.kv_len;
    TORCH_CHECK(paged ? (m.size(-1) > 0 && m.size(-1) <= limit) : m.size(-1) == limit,
                "mask kv_len mismatch");

    check_int32_offset(m, "mask", m.dim());
    p.mask = m.data_ptr<bool>();
    p.mask_k_size = static_cast<int>(m.size(-1));
    p.mask_b_stride = m.size(0) == 1 ? 0 : static_cast<int>(m.stride(0));
    if (m.dim() == 4) {
        TORCH_CHECK(m.size(1) == 1 || m.size(1) == p.q_head, "mask head mismatch");
        p.mask_h_stride = m.size(1) == 1 ? 0 : static_cast<int>(m.stride(1));
    }
    if (m.dim() >= 3) {
        const int q_dim = (int)m.dim() - 2;
        TORCH_CHECK(m.size(q_dim) == 1 || (paged ? (m.size(q_dim) > 0 && m.size(q_dim) <= p.q_len)
                                                 : m.size(q_dim) == p.q_len),
                    "mask q_len mismatch");
        p.mask_q_size = static_cast<int>(m.size(q_dim));
        p.mask_q_stride = p.mask_q_size == 1 ? 0 : static_cast<int>(m.stride(q_dim));
    }
}

} // namespace attention
} // namespace astrai
