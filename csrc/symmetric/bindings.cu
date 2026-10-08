#include "entry.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiles", &astrai::symmetric::tiles, py::arg("operation"));
    m.def("plan", &astrai::symmetric::plan, py::arg("operation"), py::arg("rows"), py::arg("cols"),
          py::arg("batch_size") = 1, py::arg("input_layout") = "row",
          py::arg("output_layout") = "row", py::arg("addend") = false, py::arg("device") = 0);
    m.def("symm_out", &astrai::symmetric::symm_out, py::arg("symmetric"), py::arg("x"), py::arg("output"),
          py::arg("addend") = py::none(), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("tile") = "64x64x32_W16x32_S2", py::arg("raster") = 1);
    m.def("syrk_out", &astrai::symmetric::syrk_out, py::arg("x"), py::arg("output"),
          py::arg("addend") = py::none(), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("tile") = "64x64x32_W16x32_S2");
}
