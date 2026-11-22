#include "entry.h"
#include <api/torch/attention.h>
#include <launcher/attention.cuh>

namespace astrai::attention {
torch::Tensor attention_with_kvcache(torch::Tensor q, torch::Tensor kc, torch::Tensor vc,
                                     torch::Tensor k, torch::Tensor v, torch::Tensor past,
                                     torch::Tensor pages, c10::optional<torch::Tensor> qs,
                                     c10::optional<torch::Tensor> mask, bool causal, int64_t split,
                                     c10::optional<torch::Tensor> op,
                                     c10::optional<torch::Tensor> ml,
                                     c10::optional<torch::Tensor> output) {
    const at::cuda::OptionalCUDAGuard guard(device_of(q));
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
                    q.size(1) % k.size(1) == 0 &&
                    q.stride(2) == 1 && k.stride(2) == 1,
                "invalid cache heads or strides");
    for (const auto& t : {past, pages})
        TORCH_CHECK(t.device() == q.device() && t.scalar_type() == torch::kInt32 &&
                        t.is_contiguous(),
                    "invalid cache metadata dtype, device or stride");
    TORCH_CHECK(past.dim() == 1 && past.numel() > 0 && pages.dim() == 2 &&
                    pages.size(0) == past.numel(),
                "cache batch mismatch");
    AttentionParams p{};
    p.q_ptr = q.data_ptr();
    p.k_ptr = kc.data_ptr();
    p.v_ptr = vc.data_ptr();
    p.new_k_ptr = k.data_ptr();
    p.new_v_ptr = v.data_ptr();
    p.page_table = pages.data_ptr<int>();
    p.cache_seqlens = past.data_ptr<int>();
    p.batch = checked_int(past.numel(), "batch");
    p.num_pages = checked_int(kc.size(0), "pages");
    p.page_size = checked_int(kc.size(1), "page size");
    p.table_stride = checked_int(pages.size(1), "table stride");
    checked_int(kc.size(0) * kc.size(1), "physical slots");
    p.kv_len = checked_int(pages.size(1) * kc.size(1), "context capacity");
    check_int32_offset(q, "Q", 3);
    check_int32_offset(k, "new K", 3);
    p.q_head = q.size(1);
    p.kv_head = k.size(1);
    p.head_dim = q.size(2);
    p.q_l_stride = q.stride(0);
    p.q_h_stride = q.stride(1);
    p.new_kv_token_stride = k.stride(0);
    p.new_kv_h_stride = k.stride(1);
    p.q_len = qs ? checked_int(q.size(0), "tokens") : 1;
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
    visit_dtype<AttentionTypes>(
        q.scalar_type(),
        [&](auto tag) {
            using T = typename decltype(tag)::Type;
            launch_attention_with_kvcache<T>(p, stream, [&](auto kernel, auto plan) {
                auto workspace = resolve_split_buffers(op, ml, p, plan, q);
                kernel.launch(p, plan, stream);
            });
        },
        [](at::ScalarType d) {
            return unsupported_dtype_message<AttentionTypes>("cache attention", d);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return shape.size() == 4 ? out.view(shape) : out;
}
} // namespace astrai::attention
