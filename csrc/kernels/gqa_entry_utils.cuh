#pragma once
#include <torch/extension.h>
#include "gqa_common.cuh"

inline void gqa_pack_params(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    bool is_causal,
    int64_t causal_offset,
    c10::optional<double> scale,
    GQAParams& p
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(k.dtype() == torch::kBFloat16);
    TORCH_CHECK(v.dtype() == torch::kBFloat16);

    p.batch = (int)q.size(0);
    p.q_head = (int)q.size(1);
    p.kv_head = (int)k.size(1);
    p.q_len = (int)q.size(2);
    p.kv_len = (int)k.size(2);
    p.head_dim = (int)q.size(3);
    p.use_mask = mask.has_value() ? 1 : 0;
    p.is_causal = is_causal ? 1 : 0;
    p.causal_offset = (int)causal_offset;
    p.scale = scale.has_value() ? (float)scale.value() : 1.0f / sqrtf((float)p.head_dim);
    p.q = (const bf16*)q.data_ptr();
    p.k = (const bf16*)k.data_ptr();
    p.v = (const bf16*)v.data_ptr();
    if (p.use_mask) {
        TORCH_CHECK(mask.value().dtype() == torch::kBool);
        TORCH_CHECK(mask.value().dim() == 2);
        TORCH_CHECK(mask.value().size(0) == p.batch);
        TORCH_CHECK(mask.value().size(1) == p.kv_len);
        p.mask = mask.value().data_ptr<bool>();
    } else {
        p.mask = nullptr;
    }
}
