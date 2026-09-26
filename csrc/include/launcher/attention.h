// Entry points of the attention kernel modules.
//
// One declaration per entry (the api.h / gated_deltanet.h shape): the .cu
// files implement these and attach each module's own pybind surface; the
// device-side vocabulary (kernels, launchers, dispatchers) stays in the
// kernel/ headers — pure CUDA, no torch.

#pragma once

#include <torch/extension.h>

#include <c10/util/Optional.h>

namespace astrai {
namespace attention {

torch::Tensor attn_decode(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    int64_t layout,
    c10::optional<torch::Tensor> o_part_buf,
    c10::optional<torch::Tensor> ml_part_buf
);

torch::Tensor attn_prefill(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    int64_t layout
);

torch::Tensor attn_paged_decode(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor kv_indptr,
    c10::optional<torch::Tensor> new_k,
    c10::optional<torch::Tensor> new_v,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    c10::optional<torch::Tensor> o_part_buf,
    c10::optional<torch::Tensor> ml_part_buf,
    c10::optional<torch::Tensor> out_buf
);

torch::Tensor attn_paged_prefill(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor kv_indptr,
    torch::Tensor qo_indptr,
    torch::Tensor q_tile_to_batch,
    torch::Tensor q_tile_to_index,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale
);

}  // namespace attention
}  // namespace astrai
