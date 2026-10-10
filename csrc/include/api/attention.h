/*
 * Attention entry points: implemented by the attention .cu files, which own
 * the pybind surface; the device vocabulary stays in kernel/ (torch-free).
 */

#pragma once

#include <torch/extension.h>

#include <c10/util/Optional.h>

namespace astrai {
namespace attention {

torch::Tensor attn_decode(torch::Tensor q,
                          torch::Tensor k,
                          torch::Tensor v,
                          c10::optional<torch::Tensor> mask,
                          c10::optional<double> scale,
                          int64_t layout,
                          c10::optional<torch::Tensor> o_part_buf,
                          c10::optional<torch::Tensor> ml_part_buf,
                          bool is_causal);

torch::Tensor attn_prefill(torch::Tensor q,
                           torch::Tensor k,
                           torch::Tensor v,
                           c10::optional<torch::Tensor> mask,
                           c10::optional<double> scale,
                           int64_t layout,
                           bool is_causal);

torch::Tensor attn_paged_decode(torch::Tensor q,
                                torch::Tensor k_cache,
                                torch::Tensor v_cache,
                                torch::Tensor req_to_token,
                                torch::Tensor req_pool_indices,
                                torch::Tensor kv_indptr,
                                c10::optional<torch::Tensor> new_k,
                                c10::optional<torch::Tensor> new_v,
                                c10::optional<torch::Tensor> mask,
                                c10::optional<double> scale,
                                c10::optional<torch::Tensor> o_part_buf,
                                c10::optional<torch::Tensor> ml_part_buf,
                                c10::optional<torch::Tensor> out_buf,
                                bool is_causal);

torch::Tensor attn_paged_prefill(torch::Tensor q,
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
                                 bool is_causal);

} // namespace attention
} // namespace astrai
