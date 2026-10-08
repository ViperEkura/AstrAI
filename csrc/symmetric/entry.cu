#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "entry.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace astrai::symmetric {
namespace {

bool dense_matrix(const torch::Tensor& x) {
    if (x.dim() != 2 && x.dim() != 3) return false;
    if (x.dim() == 3 && (x.size(0) < 1 || x.size(0) > 65535 ||
                        x.stride(0) != x.size(-2) * x.size(-1))) return false;
    return x.stride(-1) == 1 && x.stride(-2) == x.size(-1) ||
           x.stride(-2) == 1 && x.stride(-1) == x.size(-2);
}

void check_buffers(const torch::Tensor& x, const torch::Tensor& output) {
    TORCH_CHECK(x.is_cuda() && output.is_cuda(), "x and output must be CUDA tensors");
    TORCH_CHECK(x.device() == output.device(), "x and output must share a device");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
                "x and output must be bfloat16");
    TORCH_CHECK((x.dim() == 2 || x.dim() == 3) && output.dim() == x.dim(),
                "x and output must be matrices or matrix batches");
    TORCH_CHECK(x.dim() == 2 || x.size(0) == output.size(0), "batch size mismatch");
    const auto rows = x.size(-2);
    const auto reduction = x.size(-1);
    TORCH_CHECK(rows >= 64 && rows % 64 == 0 && reduction >= 64 && reduction % 64 == 0,
                "matrix dimensions must be positive multiples of 64");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "input must be 16-byte aligned");
    TORCH_CHECK(dense_matrix(x) && dense_matrix(output), "x and output must be dense row/column-major");
    TORCH_CHECK(output.size(-2) == rows && output.size(-1) == rows,
                "output must have matching square matrix shape");
    TORCH_CHECK(!x.is_alias_of(output), "input and output must not alias");
    TORCH_CHECK(rows * rows <= std::numeric_limits<int>::max() &&
                    reduction <= std::numeric_limits<int>::max(),
                "matrix dimensions exceed the kernel limit");
}


void check_addend(const torch::Tensor& output, const c10::optional<torch::Tensor>& addend,
                  float alpha, float beta) {
    TORCH_CHECK(std::isfinite(alpha) && std::isfinite(beta), "coefficients must be finite");
    TORCH_CHECK(beta == 0.0f || addend.has_value(), "nonzero beta requires an addend");
    if (!addend.has_value() || beta == 0.0f) return;
    TORCH_CHECK(addend->device() == output.device() &&
                addend->scalar_type() == output.scalar_type() &&
                addend->sizes() == output.sizes() && dense_matrix(*addend),
                "addend must match output device, dtype, shape and layout");
    TORCH_CHECK(!output.is_alias_of(*addend), "output must not alias addend");
}

} // namespace

void syrk_out(torch::Tensor x, torch::Tensor output,
              c10::optional<torch::Tensor> addend, float alpha, float beta,
              std::string tile) {
    check_buffers(x, output);
    check_addend(output, addend, alpha, beta);
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8, "BF16 tensor cores required");
    if (tile == "wmma64")
        TORCH_CHECK(x.is_contiguous(), "wmma64 requires row-major input");
    gemm::GemmParams p{};
    p.a_ptr = x.data_ptr(); p.b_ptr = x.data_ptr(); p.out_ptr = output.data_ptr();
    p.m = x.size(-2); p.n = x.size(-2); p.k = x.size(-1);
    p.batch = x.dim() == 3 ? x.size(0) : 1;
    p.a_batch_stride = p.b_batch_stride = x.size(-2) * x.size(-1);
    p.out_batch_stride = x.size(-2) * x.size(-2);
    p.a_ld = x.stride(-1) == 1 ? x.size(-1) : x.size(-2);
    p.b_ld = p.a_ld; p.out_ld = x.size(-2);
    launch_syrk(p, addend, alpha, beta, tile, !x.is_contiguous(),
                at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void symm_out(torch::Tensor symmetric, torch::Tensor x, torch::Tensor output,
              c10::optional<torch::Tensor> addend, float alpha, float beta,
              std::string tile, int raster) {
    TORCH_CHECK(symmetric.is_cuda() && x.is_cuda() && output.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(symmetric.device() == x.device() && x.device() == output.device(), "device mismatch");
    TORCH_CHECK(symmetric.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16 &&
                output.scalar_type() == at::kBFloat16, "all tensors must be bfloat16");
    TORCH_CHECK((x.dim() == 2 || x.dim() == 3) &&
                symmetric.dim() == x.dim() && output.dim() == x.dim(),
                "expected matrices or matrix batches");
    TORCH_CHECK(x.dim() == 2 || symmetric.size(0) == x.size(0), "batch size mismatch");
    const auto rows = x.size(-2), cols = x.size(-1);
    TORCH_CHECK(symmetric.size(-2) == rows && symmetric.size(-1) == rows &&
                output.sizes() == x.sizes(), "matrix shape mismatch");
    TORCH_CHECK(rows >= 64 && cols >= 64 && rows % 64 == 0 && cols % 64 == 0,
                "matrix dimensions must be positive multiples of 64");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(symmetric.data_ptr()) % 16 == 0,
                "inputs must be 16-byte aligned");
    TORCH_CHECK(dense_matrix(symmetric) && dense_matrix(x) && dense_matrix(output),
                "all tensors must be dense row/column-major");
    TORCH_CHECK(!output.is_alias_of(x) && !output.is_alias_of(symmetric),
                "output must not alias inputs");
    TORCH_CHECK(rows * cols <= std::numeric_limits<int>::max() &&
                rows * rows <= std::numeric_limits<int>::max(), "matrix exceeds kernel limit");
    TORCH_CHECK(raster >= -32 && raster <= 32, "raster must be in [-32, 32]");
    check_addend(output, addend, alpha, beta);
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8, "BF16 tensor cores required");
    gemm::GemmParams p{};
    // (S X)^T = X^T S: no materialized transpose, epilogue restores orientation.
    p.a_ptr = x.data_ptr(); p.b_ptr = symmetric.data_ptr(); p.out_ptr = output.data_ptr();
    p.m = cols; p.n = rows; p.k = rows;
    p.batch = x.dim() == 3 ? x.size(0) : 1;
    p.a_batch_stride = p.out_batch_stride = rows * cols;
    p.b_batch_stride = rows * rows;
    p.a_ld = x.is_contiguous() ? cols : rows;
    p.b_ld = rows; p.out_ld = output.is_contiguous() ? cols : rows; p.raster = raster;
    launch_symm(p, addend, alpha, beta, tile, !x.is_contiguous(),
                !output.is_contiguous(), at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


} // namespace astrai::symmetric
