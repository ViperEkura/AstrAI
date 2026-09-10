#include "dispatchers.cuh"
#include "entry_utils.cuh"

using namespace astrai::attention;

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
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams<bf16> p;
    attn_pack_paged_decode_params(q, k_cache, v_cache,
                                   req_to_token, req_pool_indices, kv_indptr,
                                   new_k, new_v,
                                   mask, causal_offset, scale, p);

    torch::Tensor O;
    if (out_buf.has_value() && out_buf->defined()) {
        TORCH_CHECK(out_buf->dtype() == q.dtype(), "out_buf dtype must match q");
        TORCH_CHECK(out_buf->is_cuda() && out_buf->is_contiguous(),
                    "out_buf must be a contiguous CUDA tensor");
        TORCH_CHECK(out_buf->size(0) >= q.size(0), "out_buf batch too small");
        TORCH_CHECK(out_buf->size(1) == q.size(1), "out_buf heads must match q");
        TORCH_CHECK(out_buf->size(2) == q.size(2), "out_buf head_dim must match q");
        TORCH_CHECK(q.is_contiguous(),
                    "q must be contiguous when out_buf is provided");
        O = out_buf.value().slice(0, 0, q.size(0));
    } else {
        O = torch::empty({q.size(0), q.size(1), q.size(2)}, q.options());
    }
    p.o_ptr = (bf16*)O.data_ptr();

    resolve_split_buffers(o_part_buf, ml_part_buf, p);
    DISPATCH_HEAD_DIM(p.head_dim, dispatch_paged_decode, p, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_paged_decode", &attn_paged_decode,
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("req_to_token"),
        py::arg("req_pool_indices"),
        py::arg("kv_indptr"),
        py::arg("new_k") = py::none(),
        py::arg("new_v") = py::none(),
        py::arg("mask") = py::none(),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        py::arg("o_part_buf") = py::none(),
        py::arg("ml_part_buf") = py::none(),
        py::arg("out_buf") = py::none(),
        "SGLang-style paged decode: flat KV pool + req_to_token + kv_indptr.");
}
