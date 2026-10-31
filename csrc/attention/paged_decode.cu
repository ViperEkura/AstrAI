/*
 * SGLang-style paged GQA decode (flat KV pool + req_to_token + kv_indptr)
 * — the implementation of the entry declared in api/torch/attention.h.
 * Device-side code is in launcher/attention.cuh +
 * kernel/attention/mainloop.cuh.
 */

#include "entry.h"
#include <api/torch/attention.h>
#include <launcher/attention.cuh>

namespace astrai {
namespace attention {

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
                                bool is_causal) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams p;
    attn_pack_paged_decode_params(q, k_cache, v_cache, req_to_token, req_pool_indices, kv_indptr,
                                  new_k, new_v, mask, scale, p, is_causal);

    torch::Tensor O;
    if (out_buf.has_value() && out_buf->defined()) {
        TORCH_CHECK(out_buf->dtype() == q.dtype(), "out_buf dtype must match q");
        TORCH_CHECK(out_buf->device() == q.device() && out_buf->is_contiguous(),
                    "out_buf must be a contiguous CUDA tensor");
        TORCH_CHECK(out_buf->size(0) >= q.size(0), "out_buf batch too small");
        TORCH_CHECK(out_buf->size(1) == q.size(1), "out_buf heads must match q");
        TORCH_CHECK(out_buf->size(2) == q.size(2), "out_buf head_dim must match q");
        TORCH_CHECK(q.is_contiguous(), "q must be contiguous when out_buf is provided");
        O = out_buf->size(0) == q.size(0) ? *out_buf : out_buf->slice(0, 0, q.size(0));
    } else {
        O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    }
    p.o_ptr = O.data_ptr();

    visit_dtype<AttentionTypes>(
        q.scalar_type(),
        [&](auto tag) {
            using T = typename decltype(tag)::Type;
            launch_decode<PagedKV<T>>(p, q, o_part_buf, ml_part_buf, stream);
        },
        [](at::ScalarType dtype) {
            return unsupported_dtype_message<AttentionTypes>("attention", dtype);
        });
    return O;
}

} // namespace attention
} // namespace astrai
