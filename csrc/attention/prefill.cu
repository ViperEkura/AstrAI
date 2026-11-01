/*
 * GQA prefill flash attention, contiguous K/V — the implementation
 * of the entry declared in api/torch/attention.h (the gated_deltanet_fwd.cu
 * shape). Device-side code (kernels, launchers, dispatchers) is in
 * launcher/attention.cuh + kernel/attention/mainloop.cuh.
 */

#include "entry.h"
#include <api/torch/attention.h>
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

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == BLHD) ? O.transpose(1, 2) : O;
    p.o_ptr = O_view.data_ptr();

    visit_dtype<AttentionTypes>(
        q.scalar_type(),
        [&](auto tag) {
            using T = typename decltype(tag)::Type;
            launch_prefill<DenseQSchedule, ContigKV<T>>(p, stream);
        },
        [](at::ScalarType dtype) {
            return unsupported_dtype_message<AttentionTypes>("attention", dtype);
        });
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

} // namespace attention
} // namespace astrai
