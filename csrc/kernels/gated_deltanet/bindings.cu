#include <torch/extension.h>

#include "gated_deltanet/gated_deltanet.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gated_deltanet_fwd", &gated_deltanet_fwd,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
        py::arg("eps") = 1e-6, py::arg("chunk") = 64,
        "Normalize q/k, transpose q/k/v to head-major, and pre-scan the gate"
    );
    m.def("gated_deltanet_bwd", &gated_deltanet_bwd,
        py::arg("q"), py::arg("k"), py::arg("v_new"), py::arg("h"),
        py::arg("g"), py::arg("do"), py::arg("scale"),
        "Backward of the GDN output stage"
    );
}
