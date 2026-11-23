#include <ATen/cuda/CUDAContext.h>
#include <api/torch/gemm.h>
#include <c10/cuda/CUDAGuard.h>
#include <kernel/gemm/small_m.cuh>
#include <utils/launch.cuh>

namespace astrai {
namespace gemm {

torch::Tensor small_m_linear(const torch::Tensor& x, const torch::Tensor& weight) {
    if (!x.is_cuda() || x.device() != weight.device() || x.scalar_type() != torch::kBFloat16 ||
        weight.scalar_type() != x.scalar_type() || x.dim() == 0 || weight.dim() != 2 ||
        x.size(-1) != weight.size(1) || !x.is_contiguous() || !weight.is_contiguous() ||
        (at::GradMode::is_enabled() && (x.requires_grad() || weight.requires_grad())))
        return at::linear(x, weight);
    const auto n = weight.size(0), k = weight.size(1);
    if (k == 0 || k > INT_MAX / 16 || k % 8 || n == 0 || n > INT_MAX / 16)
        return at::linear(x, weight);
    const auto rows = x.numel() / k;
    const auto pointers =
        reinterpret_cast<uintptr_t>(x.data_ptr()) | reinterpret_cast<uintptr_t>(weight.data_ptr());
    if (rows == 0 || rows >= 16 || (pointers & 15))
        return at::linear(x, weight);
    c10::cuda::CUDAGuard guard(x.device());
    if (!capabilities().mma)
        return at::linear(x, weight);
    auto shape = x.sizes().vec();
    shape.back() = n;
    auto out = torch::empty(shape, x.options());
    small_m_dot<4><<<dim3(n, (rows + 3) / 4), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), rows, n, k);
    ASTRAI_LAUNCH_CHECK();
    return out;
}

} // namespace gemm
} // namespace astrai
