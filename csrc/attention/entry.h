#pragma once
/* Shared by attention entry TUs for tensor checks, partial allocation, and packing. */
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

inline int checked_int(int64_t value, const char* name) {
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), name,
                " exceeds the nonnegative int32 range");
    return static_cast<int>(value);
}

// Q, mask, and contiguous KV bases use int offsets inside the kernels.
inline void check_int32_offset(const torch::Tensor& tensor, const char* name, int dimensions) {
    int64_t last_offset = 0;
    for (int dim = 0; dim < dimensions; ++dim) {
        const int64_t last_index = tensor.size(dim) - 1;
        if (last_index <= 0)
            continue;
        const int64_t stride = tensor.stride(dim);
        TORCH_CHECK(stride >= 0 &&
                        stride <= (std::numeric_limits<int>::max() - last_offset) / last_index,
                    name, " indexed offset exceeds int32 range");
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
    checked_int(static_cast<int64_t>(p.batch) * p.q_head, "attention batch * q_head");
    with_head_dim(p.head_dim, [](auto) {});
    p.scale = (float)value;
    p.o_ptr = nullptr;
    p.o_part = nullptr;
    p.ml_part = nullptr;
}

inline void set_kv_split_size(AttentionParams& p, int64_t size) {
    p.kv_split_size = checked_int(size, "KV split size");
    TORCH_CHECK(p.kv_split_size % 16 == 0, "KV split size must be a multiple of 16");
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
    p.split_stride = std::max(MAX_SPLITS, static_cast<int>(plan.grid.z));
    const int64_t o_needed = (int64_t)p.batch * p.q_head * p.split_stride * p.head_dim;
    const int64_t ml_needed = (int64_t)p.batch * p.q_head * p.split_stride * 2;
    // The standard reusable workspace has MAX_SPLITS slots. Longer fixed
    // contexts use a larger stream-local allocation.
    if (has_o && p.split_stride == MAX_SPLITS) {
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
        workspace.o_part = torch::empty({p.batch, p.q_head, p.split_stride, p.head_dim}, options);
        workspace.ml_part = torch::empty({p.batch, p.q_head, p.split_stride, 2}, options);
    }
    if (workspace.o_part.defined()) {
        p.o_part = workspace.o_part.data_ptr<float>();
        p.ml_part = workspace.ml_part.data_ptr<float>();
    }
    return workspace;
}

// Tensor ownership stays at the Torch boundary, outside the native launcher.
// Shared Q dimensions and stride extraction
inline void extract_q_dims_and_strides(torch::Tensor& q, AttentionParams& p) {
    q = q.transpose(1, 2);
    check_int32_offset(q, "Q", 4);
    p.batch = checked_int(q.size(0), "q size(0)");
    p.q_head = checked_int(q.size(1), "q size(1)");
    p.q_len = checked_int(q.size(2), "q size(2)");
    p.head_dim = checked_int(q.size(3), "q size(3)");
    p.q_b_stride = checked_int(q.stride(0), "q stride(0)");
    p.q_h_stride = checked_int(q.stride(1), "q stride(1)");
    p.q_l_stride = checked_int(q.stride(2), "q stride(2)");
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
    p.mask_k_size = checked_int(m.size(-1), "m size(-1)");
    p.mask_b_stride = m.size(0) == 1 ? 0 : checked_int(m.stride(0), "m stride(0)");
    if (m.dim() == 4) {
        TORCH_CHECK(m.size(1) == 1 || m.size(1) == p.q_head, "mask head mismatch");
        p.mask_h_stride = m.size(1) == 1 ? 0 : checked_int(m.stride(1), "m stride(1)");
    }
    if (m.dim() >= 3) {
        const int q_dim = (int)m.dim() - 2;
        TORCH_CHECK(m.size(q_dim) == 1 || (paged ? (m.size(q_dim) > 0 && m.size(q_dim) <= p.q_len)
                                                 : m.size(q_dim) == p.q_len),
                    "mask q_len mismatch");
        p.mask_q_size = checked_int(m.size(q_dim), "m size(q_dim)");
        p.mask_q_stride = p.mask_q_size == 1 ? 0 : checked_int(m.stride(q_dim), "m stride(q_dim)");
    }
}

// Contiguous-KV parameter packing
inline void attn_pack_params(torch::Tensor q,
                             torch::Tensor k,
                             torch::Tensor v,
                             c10::optional<torch::Tensor> mask,
                             c10::optional<double> scale,
                             AttentionParams& p,
                             bool is_causal) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    check_qkv_dtype(q, k, v);
    TORCH_CHECK(k.sizes() == v.sizes(), "K and V must have identical shapes");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "Q/K/V must be 4D");
    extract_q_dims_and_strides(q, p);

    k = k.transpose(1, 2);
    v = v.transpose(1, 2);

    TORCH_CHECK(k.size(0) == p.batch, "K/V batch must match Q");
    p.kv_head = checked_int(k.size(1), "k size(1)");
    p.kv_len = checked_int(k.size(2), "k size(2)");
    TORCH_CHECK(p.kv_len > 0 && p.kv_len <= std::numeric_limits<int>::max() - 16,
                "K/V sequence length must fit attention tile arithmetic");

    TORCH_CHECK(p.kv_head > 0 && p.q_head > 0 && p.q_head % p.kv_head == 0,
                "q_head must be divisible by kv_head");
    TORCH_CHECK(k.size(3) == p.head_dim, "K/V head_dim must match Q");
    check_int32_offset(k, "K/V base", 2);
    TORCH_CHECK(static_cast<int64_t>(p.q_len) * (p.q_head / p.kv_head) <=
                    std::numeric_limits<int>::max() - 1024,
                "packed query rows exceed int32 range");

    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
                "Q/K/V head_dim must be contiguous");

    p.kv_b_stride = checked_int(k.stride(0), "k stride(0)");
    p.kv_h_stride = checked_int(k.stride(1), "k stride(1)");
    p.kv_l_stride = checked_int(k.stride(2), "k stride(2)");

    p.is_causal = is_causal;

    p.q_ptr = q.data_ptr();
    p.k_ptr = k.data_ptr();
    p.v_ptr = v.data_ptr();
    p.new_k_ptr = nullptr;
    p.new_v_ptr = nullptr;
    finish_pack(scale, p);

    pack_mask(mask, p, q.device());
}

} // namespace attention
} // namespace astrai
