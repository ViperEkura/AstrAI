// SGLang-style paged GQA prefill (flat KV pool + ragged batch via
// qo_indptr/kv_indptr) — the torch-facing entry, one function per module.
// Everything device-side (kernel templates, launchers, the dtype/head_dim
// dispatchers) lives in the shared family headers:
// kernel/attention_launch.cuh and
// kernel/attention_prefill_split_q[_mma].cuh (prefill.cu instantiates the
// same templates with DenseQSchedule+ContigKV).

#include <kernel/attention_launch.cuh>
#include <launcher/dtype_list.h>
#include <launcher/entry_utils.h>

using namespace astrai::attention;

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
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams p;
    attn_pack_paged_prefill_params(q, k_cache, v_cache,
                                     req_to_token, req_pool_indices,
                                     kv_indptr, qo_indptr,
                                     q_tile_to_batch, q_tile_to_index, mask,
                                    causal_offset, scale, p);

    auto O = torch::empty({q.size(0), q.size(1), q.size(2)}, q.options());
    p.o_ptr = O.data_ptr();

    switch (q.scalar_type()) {
#define ASTRAI_ATTN_DTYPE_ROW(tag, type) \
        case tag: dispatch_paged_prefill<type>(p, stream); break;
        ASTRAI_ATTN_DTYPE_LIST(ASTRAI_ATTN_DTYPE_ROW)
#undef ASTRAI_ATTN_DTYPE_ROW
        default: attn_dtype_unsupported(q.scalar_type());
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_paged_prefill", &attn_paged_prefill,
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("req_to_token"),
        py::arg("req_pool_indices"),
        py::arg("kv_indptr"),
        py::arg("qo_indptr"),
        py::arg("q_tile_to_batch"),
        py::arg("q_tile_to_index"),
        py::arg("mask") = py::none(),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        "SGLang-style paged prefill: flat KV pool + ragged batch.");
}
