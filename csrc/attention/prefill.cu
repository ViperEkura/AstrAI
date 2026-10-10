/*
 * GQA prefill flash attention, contiguous K/V — the implementation
 * of the entry declared in api/attention.h (the gated_deltanet_fwd.cu
 * shape). Device-side code (kernels, launchers, dispatchers) is in
 * launcher/attention.cuh + kernel/attention/split_q.cuh.
 */

#include "entry.h"
#include <api/attention.h>
#include <api/attention_dtypes.h>
#include <launcher/attention.cuh>

namespace astrai {
namespace attention {

torch::Tensor attn_prefill(torch::Tensor q,
                           torch::Tensor k,
                           torch::Tensor v,
                           c10::optional<torch::Tensor> mask,
                           c10::optional<double> scale,
                           int64_t layout,
                           bool is_causal) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams p;
    attn_pack_params(q, k, v, mask, scale, layout, p, is_causal);
    TORCH_CHECK(p.head_dim % 16 == 0, "head_dim must be multiple of 16");

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == BLHD) ? O.transpose(1, 2) : O;
    p.o_ptr = O_view.data_ptr();

    attn_dtype_dispatch<AttnDispatchPrefill>(q.scalar_type(), p, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

} // namespace attention
} // namespace astrai
