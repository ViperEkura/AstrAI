#include "attn_dispatchers.cuh"
#include "attn_entry_utils.cuh"

torch::Tensor attn_paged_decode(
    torch::Tensor q,
    torch::Tensor page_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t page_size,
    int64_t kv_len,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    int64_t layout
) {
    PagedAttentionParams<bf16> p;
    attn_pack_paged_params(q, page_table, k_cache, v_cache,
                           page_size, kv_len, mask, causal_offset, scale, layout, p);

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == 1) ? O.transpose(1, 2) : O;
    p.o = (bf16*)O_view.data_ptr();

    alloc_split_partials(p);
    DISPATCH_HEAD_DIM(p.head_dim, dispatch_paged_decode, p);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_paged_decode", &attn_paged_decode,
        py::arg("q"),
        py::arg("page_table"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("page_size"),
        py::arg("kv_len"),
        py::arg("mask") = py::none(),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        py::arg("layout") = 0,
        "Paged GQA decode — split-KV with direct page-table access.");
}
