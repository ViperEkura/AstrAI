#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <vector>

namespace astrai {
namespace muon_ns {

namespace {

torch::Tensor muon_ns_impl(torch::Tensor grad, const std::vector<double>& coefficients,
                           int64_t ns_steps, double eps) {
    TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
    TORCH_CHECK(grad.dim() == 2, "grad must be a 2D matrix");
    TORCH_CHECK(grad.is_floating_point(), "grad must have a floating-point dtype");
    TORCH_CHECK(coefficients.size() == 3, "ns_coefficients must contain exactly three values");
    TORCH_CHECK(ns_steps >= 1 && ns_steps < 100, "ns_steps must be in [1, 99]");
    TORCH_CHECK(eps > 0.0, "eps must be positive");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad));
    const double a = coefficients[0];
    const double b = coefficients[1];
    const double c = coefficients[2];

    auto x = grad.to(torch::kBFloat16);
    const bool tall = x.size(0) > x.size(1);
    if (tall) {
        x = x.transpose(0, 1);
    }

    x.div_(x.norm().clamp_min(eps));

    const auto m = x.size(0);
    auto gram = torch::empty({m, m}, x.options());
    auto gram_update = torch::empty_like(gram);
    auto next_x = torch::empty_like(x);

    for (int64_t step = 0; step < ns_steps; ++step) {
        at::mm_out(gram, x, x.transpose(0, 1));
        at::addmm_out(gram_update, gram, gram, gram, b, c);
        at::addmm_out(next_x, x, gram_update, x, a, 1.0);
        std::swap(x, next_x);
    }

    return tall ? x.transpose(0, 1) : x;
}

} // namespace

} // namespace muon_ns
} // namespace astrai

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("muon_ns", &astrai::muon_ns::muon_ns_impl, py::arg("grad"),
          py::arg("ns_coefficients"), py::arg("ns_steps"), py::arg("eps"),
          "Run Muon's Newton-Schulz orthogonalization in one extension call");
}
