#include "entry.h"
#include <api/torch/attention.h>
#include <launcher/attention.cuh>

namespace astrai::attention {

/* Tensor owners are retained by the entry until both kernels are launched. */
struct SplitWorkspace {
    torch::Tensor o_part;
    torch::Tensor ml_part;
};

static SplitWorkspace resolve_split_buffers(const c10::optional<torch::Tensor>& o_part_buf,
                                            const c10::optional<torch::Tensor>& ml_part_buf,
                                            AttentionParams& p,
                                            bool direct_output,
                                            const torch::Tensor& q) {
    const bool has_o = o_part_buf.has_value() && o_part_buf->defined();
    const bool has_ml = ml_part_buf.has_value() && ml_part_buf->defined();
    TORCH_CHECK(has_o == has_ml, "split buffers must be provided together");
    SplitWorkspace workspace;
    p.split_stride = std::max(MAX_SPLITS, p.num_splits);
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
    } else if (!direct_output) {
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

torch::Tensor attention_with_kvcache(torch::Tensor q,
                                     torch::Tensor kc,
                                     torch::Tensor vc,
                                     torch::Tensor k,
                                     torch::Tensor v,
                                     torch::Tensor past,
                                     torch::Tensor pages,
                                     c10::optional<torch::Tensor> qs,
                                     c10::optional<torch::Tensor> mask,
                                     bool causal,
                                     int64_t split,
                                     c10::optional<torch::Tensor> op,
                                     c10::optional<torch::Tensor> ml,
                                     c10::optional<torch::Tensor> output) {
    const at::cuda::OptionalCUDAGuard guard(device_of(q));
    TORCH_CHECK(!at::GradMode::is_enabled() ||
                    !(q.requires_grad() || k.requires_grad() || v.requires_grad() ||
                      kc.requires_grad() || vc.requires_grad()),
                "cache attention supports forward only");
    TORCH_CHECK(q.dim() == 3 || q.dim() == 4, "cache queries must be THD or BSHD");
    const auto shape = q.sizes().vec();
    if (q.dim() == 4) {
        TORCH_CHECK(!qs && past.dim() == 1 && q.size(0) == past.numel() && q.size(1) > 0 &&
                        k.dim() == 4 && v.dim() == 4 && k.sizes() == v.sizes() &&
                        k.sizes().slice(0, 2) == q.sizes().slice(0, 2),
                    "regular cache attention expects matching BSHD inputs");
        if (q.size(1) > 1)
            qs = torch::arange(q.size(0) + 1, q.options().dtype(torch::kInt32)) * q.size(1);
        q = q.flatten(0, 1);
        k = k.flatten(0, 1);
        v = v.flatten(0, 1);
    }
    match_kv_layout(k, v);
    check_qkv_dtype(q, k, v);
    check_qkv_dtype(q, kc, vc);
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && k.sizes() == v.sizes() && k.size(0) == q.size(0),
                "invalid appended THD tensors");
    TORCH_CHECK(kc.dim() == 4 && kc.is_contiguous() && vc.is_contiguous() &&
                    kc.sizes() == vc.sizes(),
                "expected contiguous 4D caches");
    TORCH_CHECK(kc.size(1) > 0 && kc.size(2) == k.size(1) && kc.size(3) == q.size(2) &&
                    k.size(2) == q.size(2) && k.size(1) > 0 && q.size(1) > 0 &&
                    q.size(1) % k.size(1) == 0 && q.stride(2) == 1 && k.stride(2) == 1,
                "invalid cache heads or strides");
    for (const auto& t : {past, pages})
        TORCH_CHECK(t.device() == q.device() && t.scalar_type() == torch::kInt32 &&
                        t.is_contiguous(),
                    "invalid cache metadata dtype, device or stride");
    TORCH_CHECK(past.dim() == 1 && past.numel() > 0 && pages.dim() == 2 &&
                    pages.size(0) == past.numel(),
                "cache batch mismatch");
    const int64_t limit = std::numeric_limits<int>::max();
    TORCH_CHECK(past.numel() <= limit && kc.size(1) <= limit && kc.size(0) <= limit / kc.size(1) &&
                    pages.size(1) <= limit / kc.size(1),
                "cache batch, slots or capacity exceeds int32 range");
    AttentionParams p{};
    p.q_ptr = q.data_ptr();
    p.k_ptr = kc.data_ptr();
    p.v_ptr = vc.data_ptr();
    p.new_k_ptr = k.data_ptr();
    p.new_v_ptr = v.data_ptr();
    p.page_table = pages.data_ptr<int>();
    p.cache_seqlens = past.data_ptr<int>();
    p.batch = static_cast<int>(past.numel());
    p.num_pages = static_cast<int>(kc.size(0));
    p.page_size = static_cast<int>(kc.size(1));
    p.table_stride = static_cast<int>(pages.size(1));
    p.kv_len = static_cast<int>(pages.size(1) * kc.size(1));
    check_int32_offset(q, "Q", 3);
    check_int32_offset(k, "new K", 3);
    p.q_head = static_cast<int>(q.size(1));
    p.kv_head = static_cast<int>(k.size(1));
    p.head_dim = static_cast<int>(q.size(2));
    p.q_l_stride = static_cast<int>(q.stride(0));
    p.q_h_stride = static_cast<int>(q.stride(1));
    p.new_kv_token_stride = static_cast<int>(k.stride(0));
    p.new_kv_h_stride = static_cast<int>(k.stride(1));
    p.q_len = qs ? static_cast<int>(q.size(0)) : 1;
    p.is_causal = causal;
    if (qs) {
        TORCH_CHECK(qs->device() == q.device() && qs->scalar_type() == torch::kInt32 &&
                        qs->is_contiguous() && qs->dim() == 1 && qs->numel() == p.batch + 1,
                    "invalid query offsets");
        p.cu_seqlens_q = qs->data_ptr<int>();
    } else
        TORCH_CHECK(q.size(0) == p.batch, "decode requires one token per request");
    pack_mask(mask, p, q.device(), true);
    finish_pack(c10::nullopt, p);
    set_kv_split_size(p, split);
    torch::Tensor out;
    if (output) {
        TORCH_CHECK(output->device() == q.device() && output->scalar_type() == q.scalar_type() &&
                        output->dim() == 3 && output->size(0) >= q.size(0) &&
                        output->sizes().slice(1) == q.sizes().slice(1) && output->is_contiguous() &&
                        q.is_contiguous(),
                    "invalid decode output workspace");
        out = output->narrow(0, 0, q.size(0));
    } else
        out = torch::empty_strided(q.sizes(), q.strides(), q.options());
    p.o_ptr = out.data_ptr();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto allocate = [&](bool direct_output) {
        return resolve_split_buffers(op, ml, p, direct_output, q);
    };
    if (q.scalar_type() == at::kBFloat16)
        launch_attention_with_kvcache<bf16>(p, stream, allocate);
    else
        launch_attention_with_kvcache<fp16>(p, stream, allocate);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return shape.size() == 4 ? out.view(shape) : out;
}
} // namespace astrai::attention
