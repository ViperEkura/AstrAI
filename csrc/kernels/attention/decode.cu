#include "dispatchers.cuh"
#include "entry_utils.cuh"

using namespace astrai::attention;

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
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams<bf16> p;
    attn_pack_params(q, k, v, mask, causal_offset, scale, layout, p);
    TORCH_CHECK(p.q_len == 1, "Q seq_len must be 1");
    TORCH_CHECK(p.head_dim % 32 == 0, "head_dim must be multiple of 32");

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == BLHD) ? O.transpose(1, 2) : O;
    p.o_ptr = (bf16*)O_view.data_ptr();

    resolve_split_buffers(o_part_buf, ml_part_buf, p);
    DISPATCH_HEAD_DIM(p.head_dim, dispatch_decode, p, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_decode", &attn_decode,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        py::arg("layout") = (int64_t)BHLD,
        py::arg("o_part_buf") = py::none(),
        py::arg("ml_part_buf") = py::none(),
        "GQA decode (tensor-core head-packing on sm_80+, scalar fallback)");
}
