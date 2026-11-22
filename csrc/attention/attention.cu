#include "entry.h"
#include <api/torch/attention.h>
#include <launcher/attention.cuh>

namespace astrai {
namespace attention {

torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                        c10::optional<torch::Tensor> mask, c10::optional<double> scale,
                        bool is_causal, int64_t kv_split_size) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams p{};
    match_kv_layout(k, v);
    attn_pack_params(q, k, v, mask, scale, p, is_causal);

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    p.o_ptr = O.data_ptr();
    set_kv_split_size(p, kv_split_size);

    visit_dtype<AttentionTypes>(
        q.scalar_type(),
        [&](auto tag) {
            using T = typename decltype(tag)::Type;
            launch_attention<T>(p, stream);
        },
        [](at::ScalarType dtype) {
            return unsupported_dtype_message<AttentionTypes>("attention", dtype);
        });
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

} // namespace attention
} // namespace astrai
