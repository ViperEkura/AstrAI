#include <torch/extension.h>

#include <api/attention.h>
#include <api/attention_common.h>

/*
 * The attention family's whole pybind surface: four entries, one module
 * (`attention`) — a single PYBIND block, one def per api/attention.h entry.
 */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_decode", &astrai::attention::attn_decode, py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("mask") = py::none(), py::arg("scale") = py::none(),
          py::arg("layout") = (int64_t)astrai::attention::BHLD, py::arg("o_part_buf") = py::none(),
          py::arg("ml_part_buf") = py::none(), py::arg("is_causal") = false, "GQA decode (tensor-core head-packing on sm_80+)");
    m.def("attn_prefill", &astrai::attention::attn_prefill, py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("mask") = py::none(),
          py::arg("scale") = py::none(), py::arg("layout") = (int64_t)astrai::attention::BHLD,
          py::arg("is_causal") = false, "GQA prefill (tensor-core mma on sm_80+)");
    m.def("attn_paged_decode", &astrai::attention::attn_paged_decode, py::arg("q"),
          py::arg("k_cache"), py::arg("v_cache"), py::arg("req_to_token"),
          py::arg("req_pool_indices"), py::arg("kv_indptr"), py::arg("new_k") = py::none(),
          py::arg("new_v") = py::none(), py::arg("mask") = py::none(),
          py::arg("scale") = py::none(), py::arg("o_part_buf") = py::none(),
          py::arg("ml_part_buf") = py::none(), py::arg("out_buf") = py::none(), py::arg("is_causal") = false,
          "SGLang-style paged decode: flat KV pool + req_to_token + kv_indptr");
    m.def("attn_paged_prefill", &astrai::attention::attn_paged_prefill, py::arg("q"),
          py::arg("k_cache"), py::arg("v_cache"), py::arg("req_to_token"),
          py::arg("req_pool_indices"), py::arg("kv_indptr"), py::arg("qo_indptr"),
          py::arg("q_tile_to_batch"), py::arg("q_tile_to_index"), py::arg("mask") = py::none(),
          py::arg("scale") = py::none(),
          py::arg("is_causal") = false, "SGLang-style paged prefill: flat KV pool + ragged batch");
}
