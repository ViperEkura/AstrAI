#pragma once

#include <torch/types.h>
#include <c10/util/Optional.h>

namespace astrai::attention {

torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                        c10::optional<torch::Tensor> mask, c10::optional<double> scale,
                        bool is_causal, int64_t kv_split_size);

torch::Tensor attention_varlen(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                               torch::Tensor cu_seqlens_q,
                               c10::optional<torch::Tensor> cu_seqlens_k, bool is_causal,
                               c10::optional<double> scale);

torch::Tensor attention_with_kvcache(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                                     torch::Tensor k_new, torch::Tensor v_new,
                                     torch::Tensor cache_seqlens, torch::Tensor page_table,
                                     c10::optional<torch::Tensor> cu_seqlens_q,
                                     c10::optional<torch::Tensor> mask, bool is_causal,
                                     int64_t kv_split_size, c10::optional<torch::Tensor> o_part,
                                     c10::optional<torch::Tensor> ml_part,
                                     c10::optional<torch::Tensor> output);

} // namespace astrai::attention
