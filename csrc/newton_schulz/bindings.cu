#include "entry.h"
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("iterate", &astrai::newton_schulz::iterate, py::arg("x"), py::arg("gram"),
          py::arg("polynomial"), py::arg("work"), py::arg("spare"), py::arg("final"),
          py::arg("steps"), py::arg("a"), py::arg("b"), py::arg("c"), py::arg("choices"));
    m.def("tiles", &astrai::newton_schulz::symmetric::tiles, py::arg("operation"));
    m.def("plan", &astrai::newton_schulz::symmetric::plan, py::arg("operation"), py::arg("rows"),
          py::arg("cols"), py::arg("batch_size") = 1, py::arg("input_layout") = "row",
          py::arg("output_layout") = "row", py::arg("addend") = false, py::arg("device") = 0,
          py::arg("mode") = "model");
    m.def("symm_out", &astrai::newton_schulz::symmetric::symm_out, py::arg("symmetric"),
          py::arg("x"), py::arg("output"), py::arg("addend") = py::none(), py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f, py::arg("tile") = "64x64x32_W16x32_S2", py::arg("raster") = 1);
    m.def("syrk_out", &astrai::newton_schulz::symmetric::syrk_out, py::arg("x"), py::arg("output"),
          py::arg("addend") = py::none(), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("tile") = "64x64x32_W16x32_S2");
}
