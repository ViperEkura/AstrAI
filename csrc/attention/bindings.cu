#include <api/torch/attention.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace astrai::attention;
    m.def("attention", &attention, py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("mask") = py::none(), py::arg("scale") = py::none(), py::arg("is_causal") = false,
          py::arg("kv_split_size") = 0);
    m.def("attention_varlen", &attention_varlen, py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k") = py::none(),
          py::arg("is_causal") = false, py::arg("scale") = py::none());
    m.def("attention_with_kvcache", &attention_with_kvcache, py::arg("q"), py::arg("k_cache"),
          py::arg("v_cache"), py::arg("k_new"), py::arg("v_new"), py::arg("cache_seqlens"),
          py::arg("page_table"), py::arg("cu_seqlens_q") = py::none(), py::arg("mask") = py::none(),
          py::arg("is_causal") = false, py::arg("kv_split_size") = 0,
          py::arg("o_part") = py::none(), py::arg("ml_part") = py::none(),
          py::arg("output") = py::none());
}
