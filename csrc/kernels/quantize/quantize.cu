// CUDA bindings for the stateless FP8 quantize primitives.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

#include "common/device.cuh"
#include "checks.h"
#include "quantize.cuh"

using namespace astrai::quant;

namespace {

// Dtype dispatch over the unified quantize launcher: one case per
// supported input dtype; the default is a hard error (entry-checked, so
// unreachable — never a silent bf16 re-route).
template <bool Tiled, typename Fp8T>
void launch_for_dtype(const torch::Tensor& x, const QuantParams& p,
                      cudaStream_t stream) {
    switch (x.scalar_type()) {
    case torch::kBFloat16:
        launch_fp8_quantize<Fp8T, __nv_bfloat16, Tiled>(p, stream);
        break;
    case torch::kHalf:
        launch_fp8_quantize<Fp8T, __half, Tiled>(p, stream);
        break;
    case torch::kFloat32:
        launch_fp8_quantize<Fp8T, float, Tiled>(p, stream);
        break;
    default:
        TORCH_CHECK(false, "unsupported quantize input dtype: ",
                    x.scalar_type());
    }
}

template <bool Tiled>
void launch_quantize_for(const torch::Tensor& x, const QuantParams& p,
                         bool e5m2, cudaStream_t stream) {
    if (e5m2)
        launch_for_dtype<Tiled, __nv_fp8_e5m2>(x, p, stream);
    else
        launch_for_dtype<Tiled, __nv_fp8_e4m3>(x, p, stream);
}

// Shared binding body for the two quantize entry points: RowMajor /
// Transposed (single output) serve quantize(), Dual (both orientations from
// one read) serves quantize_dual(). A ring tensor switches
// on the in-kernel delayed-scaling fold: state layout
// [hist n | scale | legacy | amax | done-as-int], and the returned amax is
// the (self-cleaned) persistent slot — its only reducer. Without a ring the
// kernel runs a pure scale+cast (no fused amax: p.amax stays null) and the
// returned amax is None; callers that need one measure it themselves
// (dynamic scaling), matching the TE-delayed versus torchao-dynamic split.
py::object quantize_impl(torch::Tensor x, torch::Tensor scale,
                         at::ScalarType out_dtype, QuantLayout layout,
                         py::object ring, int64_t hist_idx, double fp8_max,
                         double pow2_margin) {
    TORCH_CHECK(x.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 ||
                    x.scalar_type() == torch::kHalf ||
                    x.scalar_type() == torch::kFloat32,
                "x must be bf16, fp16 or fp32");
    TORCH_CHECK(out_dtype == torch::kFloat8_e4m3fn ||
                    out_dtype == torch::kFloat8_e5m2,
                "unsupported quantize output dtype: expected "
                "float8_e4m3fn or float8_e5m2");
    TORCH_CHECK(layout == QuantLayout::RowMajor || x.dim() >= 2,
                "transposed quantize layouts need a 2D+ tensor");
    TORCH_CHECK(scale.is_cuda() && scale.device() == x.device() &&
                    scale.scalar_type() == torch::kFloat32 && scale.numel() == 1,
                "scale must be a CUDA float32 scalar on the input device");
    check_fp8_device(x.device().index());
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto input = x.contiguous();
    auto out_opts = input.options().dtype(out_dtype);
    torch::Tensor amax;
    float *ring_hist = nullptr, *ring_scale_out = nullptr;
    unsigned int* ring_done = nullptr;
    int ring_len = 0;
    if (!ring.is_none()) {
        auto st = ring.cast<torch::Tensor>();
        TORCH_CHECK(st.is_cuda() && st.dim() == 1 &&
                        st.scalar_type() == torch::kFloat32,
                    "ring state must be a 1D float32 CUDA tensor");
        const int64_t n = st.numel() - 4;
        TORCH_CHECK(n > 0 && hist_idx >= 0 && hist_idx < n,
                    "ring state too small or hist_idx out of range");
        float* base = st.data_ptr<float>();
        amax = st.narrow(0, n + 2, 1);
        ring_hist = base;
        ring_scale_out = base + n;
        ring_done = reinterpret_cast<unsigned int*>(base + n + 3);
        ring_len = static_cast<int>(n);
    }

    QuantParams p;
    p.input_ptr = input.data_ptr();
    p.scale = scale.data_ptr<float>();
    p.amax = amax.defined() ? amax.data_ptr<float>() : nullptr;
    if (ring_hist) {
        p.fold_ring = true;
        p.hist = ring_hist;
        p.scale_out = ring_scale_out;
        p.done = ring_done;
        p.hist_len = ring_len;
        p.hist_idx = static_cast<int>(hist_idx);
        p.fp8_max = static_cast<float>(fp8_max);
        p.pow2_margin = static_cast<float>(pow2_margin);
    }
    p.total = static_cast<int>(input.numel());
    p.out_layout = layout;
    p.rows = static_cast<int>(input.size(-2));
    p.cols = static_cast<int>(input.size(-1));
    torch::Tensor output, output_t;
    if (layout != QuantLayout::Transposed) {
        output = torch::empty_like(input, out_opts);
        p.output_ptr = output.data_ptr();
    }
    if (layout != QuantLayout::RowMajor) {
        output_t = torch::empty({input.size(-1), input.size(-2)}, out_opts);
        p.output_transposed_ptr = output_t.data_ptr();
    }
    const bool e5m2 = out_dtype == torch::kFloat8_e5m2;
    if (layout == QuantLayout::RowMajor)
        launch_quantize_for<false>(input, p, e5m2, stream.stream());
    else
        launch_quantize_for<true>(input, p, e5m2, stream.stream());
    C10_CUDA_CHECK(cudaGetLastError());
    if (layout == QuantLayout::Dual)
        return py::make_tuple(output, output_t, amax);
    return py::make_tuple(
        layout == QuantLayout::Transposed ? output_t : output, amax);
}

}  // namespace

// Single-orientation quantize binding: row-major x8, or its [cols][rows]
// transpose when transposed is set — the K-contiguous operand orientation
// NT GEMMs want. Returns (x8|x8T, amax); amax is the ring's self-cleaned
// slot when ring_state is given, else None (pure scale+cast).
py::object quantize(torch::Tensor x, torch::Tensor scale,
                    at::ScalarType dtype, bool transposed, py::object ring,
                    int64_t hist_idx, double fp8_max, double pow2_margin) {
    const QuantLayout layout =
        transposed ? QuantLayout::Transposed : QuantLayout::RowMajor;
    return quantize_impl(x, scale, dtype, layout, ring, hist_idx, fp8_max,
                         pow2_margin);
}

// Dual-orientation quantize binding: one read of x produces both the
// row-major x8 and its transpose (plus amax), for tensors consumed by GEMMs
// in both orientations (backward g). Returns (x8, x8T, amax), amax as above.
py::object quantize_dual(torch::Tensor x, torch::Tensor scale,
                         at::ScalarType dtype, py::object ring,
                         int64_t hist_idx, double fp8_max,
                         double pow2_margin) {
    return quantize_impl(x, scale, dtype, QuantLayout::Dual, ring, hist_idx,
                         fp8_max, pow2_margin);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &quantize, py::arg("x"), py::arg("scale"),
          py::arg("dtype"), py::arg("transposed") = false,
          py::arg("ring") = py::none(), py::arg("hist_idx") = 0,
          py::arg("fp8_max") = 448.0, py::arg("pow2_margin") = 1.0);
    m.def("quantize_dual", &quantize_dual, py::arg("x"), py::arg("scale"),
          py::arg("dtype"), py::arg("ring") = py::none(),
          py::arg("hist_idx") = 0, py::arg("fp8_max") = 448.0,
          py::arg("pow2_margin") = 1.0);
}
