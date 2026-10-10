#pragma once
/* Internal boundary between the FP8 autograd path and its pybind runtime. */
#include <c10/util/Optional.h>
#include <torch/extension.h>

namespace astrai {
namespace fp8 {

torch::Tensor fp8_linear(const torch::Tensor& x,
                         const torch::Tensor& w,
                         const c10::optional<torch::Tensor>& bias,
                         bool update_rings,
                         bool need_bias_grad,
                         bool dynamic,
                         int64_t history_len,
                         int64_t margin,
                         at::ScalarType fmt_a,
                         at::ScalarType fmt_b,
                         int64_t slot);

void bind_fp8(py::module& m);

} // namespace fp8
} // namespace astrai
