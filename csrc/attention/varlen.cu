#include "entry.h"
#include <api/torch/attention.h>
#include <launcher/attention.cuh>

namespace astrai::attention {

static AttentionParams pack_varlen(const torch::Tensor& q,
                                   const torch::Tensor& k,
                                   const torch::Tensor& v,
                                   const torch::Tensor& qs,
                                   const torch::Tensor& ks,
                                   bool causal,
                                   double scale) {
    check_qkv_dtype(q, k, v);
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && k.sizes() == v.sizes() && q.size(2) == k.size(2) &&
                    k.size(1) > 0 && q.size(1) > 0 && q.size(1) % k.size(1) == 0,
                "incompatible varlen head or token layout");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "expected contiguous THD");
    for (const auto& ptr : {qs, ks})
        TORCH_CHECK(ptr.device() == q.device() && ptr.scalar_type() == torch::kInt32 &&
                        ptr.dim() == 1 && ptr.is_contiguous() && ptr.numel() >= 2,
                    "expected contiguous int32 segment offsets on Q's device");
    TORCH_CHECK(qs.numel() == ks.numel(), "segment counts differ");
    AttentionParams p{};
    check_int32_offset(q, "Q", 3);
    check_int32_offset(k, "KV", 3);
    TORCH_CHECK(q.numel() <= std::numeric_limits<int>::max() &&
                    k.numel() <= std::numeric_limits<int>::max() &&
                    qs.numel() - 1 <= std::numeric_limits<int>::max(),
                "varlen extent or batch exceeds int32 range");
    p.batch = static_cast<int>(qs.numel() - 1);
    p.q_len = static_cast<int>(q.size(0));
    p.kv_len = static_cast<int>(k.size(0));
    p.q_head = static_cast<int>(q.size(1));
    p.kv_head = static_cast<int>(k.size(1));
    p.head_dim = static_cast<int>(q.size(2));
    p.q_l_stride = static_cast<int>(q.stride(0));
    p.q_h_stride = static_cast<int>(q.stride(1));
    p.kv_l_stride = static_cast<int>(k.stride(0));
    p.kv_h_stride = static_cast<int>(k.stride(1));
    p.q_ptr = q.data_ptr();
    p.k_ptr = k.data_ptr();
    p.v_ptr = v.data_ptr();
    p.cu_seqlens_q = qs.data_ptr<int>();
    p.cu_seqlens_k = ks.data_ptr<int>();
    p.is_causal = causal;
    p.scale = scale;
    TORCH_CHECK(std::isfinite(p.scale) && p.scale > 0, "native scale must be positive and finite");
    detail::with_head_dim(p.head_dim, [](auto) {});
    return p;
}

torch::Tensor attention_varlen(torch::Tensor q,
                               torch::Tensor k,
                               torch::Tensor v,
                               torch::Tensor qs,
                               c10::optional<torch::Tensor> ks,
                               bool causal,
                               c10::optional<double> scale) {
    const at::cuda::OptionalCUDAGuard guard(device_of(q));
    TORCH_CHECK(!at::GradMode::is_enabled() ||
                    !(q.requires_grad() || k.requires_grad() || v.requires_grad()),
                "attention_varlen supports forward only");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && k.sizes() == v.sizes() &&
                    (ks || q.size(0) == k.size(0)),
                "shared offsets require equal token counts");
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    auto p = pack_varlen(q, k, v, qs, ks.value_or(qs), causal,
                         scale.value_or(1.0 / std::sqrt(q.size(2))));
    auto out = torch::empty_like(q);
    auto stream = at::cuda::getCurrentCUDAStream();
    p.o_ptr = out.data_ptr();
    if (p.q_len)
        visit_dtype<AttentionTypes>(
            q.scalar_type(),
            [&](auto tag) {
                using T = typename decltype(tag)::Type;
                launch_attention_varlen<T>(p, stream);
            },
            [](at::ScalarType d) {
                return unsupported_dtype_message<AttentionTypes>("varlen", d);
            });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
} // namespace astrai::attention
