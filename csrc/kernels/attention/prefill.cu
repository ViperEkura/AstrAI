#include "attention/dispatchers.cuh"
#include "attention/dtype_list.cuh"
#include "attention/entry_utils.cuh"

using namespace astrai::attention;

torch::Tensor attn_prefill(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    int64_t layout
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams p;
    attn_pack_params(q, k, v, mask, causal_offset, scale, layout, p);
    TORCH_CHECK(p.head_dim % 16 == 0, "head_dim must be multiple of 16");

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == BLHD) ? O.transpose(1, 2) : O;
    p.o_ptr = O_view.data_ptr();

    switch (q.scalar_type()) {
#define ASTRAI_ATTN_DTYPE_ROW(tag, type) \
        case tag: dispatch_prefill<type>(p, stream); break;
        ASTRAI_ATTN_DTYPE_LIST(ASTRAI_ATTN_DTYPE_ROW)
#undef ASTRAI_ATTN_DTYPE_ROW
        default: attn_dtype_unsupported(q.scalar_type());
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_prefill", &attn_prefill,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        py::arg("layout") = (int64_t)BHLD,
        "GQA prefill (tensor-core mma on sm_80+, scalar fallback)");
}
