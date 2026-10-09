/*
 * SGLang-style paged GQA prefill (flat KV pool, ragged batch via
 * qo_indptr/kv_indptr) — the implementation of the entry declared in
 * api/attention.h. Device-side code is in launcher/attention.cuh
 * + kernel/attention/split_q.cuh.
 */

#include "entry.h"
#include <api/attention.h>
#include <api/attention_dtypes.h>
#include <launcher/attention.cuh>

namespace astrai {
namespace attention {

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
                                 c10::optional<double> scale, bool is_causal) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams p;
    attn_pack_paged_prefill_params(q, k_cache, v_cache, req_to_token, req_pool_indices, kv_indptr,
                                   qo_indptr, q_tile_to_batch, q_tile_to_index, mask,
                                   scale, p, is_causal);

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    p.o_ptr = O.data_ptr();

    attn_dtype_dispatch<AttnDispatchPagedPrefill>(q.scalar_type(), p, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

} // namespace attention
} // namespace astrai
