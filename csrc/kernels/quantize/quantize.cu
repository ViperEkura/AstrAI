// CUDA bindings for the stateless FP8 quantize primitives. The launcher, the
// ring layout and the composed pass live in ``launch.cuh`` so the fp8-linear
// composition (``gemm/fp8_linear.cu``, compiled into the gemm module
// where the GEMM dispatch state lives) shares them instead of re-deriving
// them; this TU is only the pybind surface.

#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <torch/extension.h>

#include "launch.cuh"

using namespace astrai::quant;

namespace {

at::ScalarType scalar_type_of(const py::object& dtype) {
    TORCH_CHECK(!dtype.is_none(), "quantize output dtype is required");
    return dtype.cast<at::ScalarType>();
}

c10::optional<torch::Tensor> optional_tensor(const py::object& t,
                                             const char* name) {
    if (t.is_none()) return c10::nullopt;
    torch::Tensor tensor = t.cast<torch::Tensor>();
    TORCH_CHECK(tensor.defined(), name, " must be a defined tensor");
    return tensor;
}

// Shared binding body for the two quantize entry points: RowMajor /
// Transposed (single output) serve quantize(), Dual (both orientations from
// one read) serve quantize_dual(). A ring tensor switches on the in-kernel
// delayed-scaling fold (state layout owned by launch.cuh's RingView), and
// the returned amax is the ring's self-cleaned persistent slot — its only
// reducer. Without a ring the kernel runs a pure scale+cast (no fused amax)
// and the returned amax is None; callers that need one measure it
// themselves (dynamic scaling), matching the TE-delayed versus
// torchao-dynamic split. ``transposed_dtype`` (default: the row-major
// dtype) selects the transposed orientation's format independently — the
// hybrid training pair casts with the forward format on one side and the
// backward format on the other, both from a single read.
py::object quantize_impl(torch::Tensor x, torch::Tensor scale,
                         at::ScalarType out_dtype, QuantLayout layout,
                         py::object transposed_dtype, py::object ring,
                         int64_t hist_idx, py::object hist_len, double fp8_max,
                         double pow2_margin) {
    c10::optional<at::ScalarType> t_dtype = c10::nullopt;
    if (!transposed_dtype.is_none())
        t_dtype = transposed_dtype.cast<at::ScalarType>();
    c10::optional<int64_t> ring_hist_len = c10::nullopt;
    if (!hist_len.is_none()) ring_hist_len = hist_len.cast<int64_t>();
    const QuantizeOutputs outs =
        run_quantize(x, scale, layout, out_dtype, t_dtype,
                     optional_tensor(ring, "ring_state"), hist_idx, fp8_max,
                     pow2_margin, c10::nullopt, c10::nullopt, ring_hist_len);
    if (layout == QuantLayout::Dual)
        return py::make_tuple(outs.out, outs.out_t, outs.amax);
    return py::make_tuple(layout == QuantLayout::Transposed ? outs.out_t
                                                            : outs.out,
                          outs.amax);
}

// Single-orientation quantize binding: row-major x8, or its [cols][rows]
// transpose when transposed is set — the K-contiguous operand orientation
// NT GEMMs want. Returns (x8|x8T, amax); amax is the fold's raw-domain amax
// of the round when ring_state is given, else None (pure scale+cast).
py::object quantize(torch::Tensor x, torch::Tensor scale,
                    at::ScalarType dtype, bool transposed, py::object ring,
                    int64_t hist_idx, py::object hist_len, double fp8_max,
                    double pow2_margin) {
    const QuantLayout layout =
        transposed ? QuantLayout::Transposed : QuantLayout::RowMajor;
    return quantize_impl(x, scale, dtype, layout, py::none(), ring, hist_idx,
                         hist_len, fp8_max, pow2_margin);
}

// Dual-orientation quantize binding: one read of x produces both the
// row-major x8 and its transpose (plus amax), for tensors consumed by GEMMs
// in both orientations (backward g). ``transposed_dtype`` casts the
// transposed side in a different fp8 format from one read (hybrid training:
// E4M3 forward operand, E5M2 backward operand). Returns (x8, x8T, amax),
// amax as above.
py::object quantize_dual(torch::Tensor x, torch::Tensor scale,
                         at::ScalarType dtype, py::object transposed_dtype,
                         py::object ring, int64_t hist_idx, py::object hist_len,
                         double fp8_max, double pow2_margin) {
    return quantize_impl(x, scale, dtype, QuantLayout::Dual, transposed_dtype,
                         ring, hist_idx, hist_len, fp8_max, pow2_margin);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // The fold-scratch extent, for policy/tests that size the ring buffer
    // (the full layout is RingLayout, quantize/common.h).
    m.attr("K_FOLD_SLOTS") = kFoldSlots;
    m.def("quantize", &quantize, py::arg("x"), py::arg("scale"),
          py::arg("dtype"), py::arg("transposed") = false,
          py::arg("ring") = py::none(), py::arg("hist_idx") = 0,
          py::arg("hist_len") = py::none(), py::arg("fp8_max") = 448.0,
          py::arg("pow2_margin") = 1.0);
    m.def("quantize_dual", &quantize_dual, py::arg("x"), py::arg("scale"),
          py::arg("dtype"), py::arg("transposed_dtype") = py::none(),
          py::arg("ring") = py::none(), py::arg("hist_idx") = 0,
          py::arg("hist_len") = py::none(), py::arg("fp8_max") = 448.0,
          py::arg("pow2_margin") = 1.0);
}
