#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

namespace astrai {
namespace muon_ns {

namespace {

// Match BF16 clamp_min followed by division, including rounding eps to the
// tensor dtype before comparison. Keep the norm reduction order unchanged.
__global__ void
normalize_bf16_kernel(at::BFloat16* x, const at::BFloat16* norm, int64_t count, at::BFloat16 eps) {
    float divisor = static_cast<float>(norm[0]);
    const float floor = static_cast<float>(eps);
    if (divisor < floor) {
        divisor = floor;
    }
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        x[index] = static_cast<at::BFloat16>(__fdiv_rn(static_cast<float>(x[index]), divisor));
    }
}

// One warp owns an output tile. The BF16 product stays in FP32 until the
// polynomial epilogue, so the Gram matrix is read only for MMA and the
// residual, and the rounded polynomial is written once.
__global__ void fused_gram_polynomial_kernel(
    const __nv_bfloat16* gram, __nv_bfloat16* polynomial, int dim, float b, float c) {
    using namespace nvcuda;
    constexpr int kTile = 16;
    const int row = blockIdx.y * kTile;
    const int col = blockIdx.x * kTile;
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __nv_bfloat16, wmma::row_major> lhs;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __nv_bfloat16, wmma::row_major> rhs;
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> product;
    wmma::fill_fragment(product, 0.0f);
    for (int k = 0; k < dim; k += kTile) {
        wmma::load_matrix_sync(lhs, gram + row * dim + k, dim);
        wmma::load_matrix_sync(rhs, gram + k * dim + col, dim);
        wmma::mma_sync(product, lhs, rhs, product);
    }
    __shared__ float tile[kTile * kTile];
    wmma::store_matrix_sync(tile, product, kTile, wmma::mem_row_major);
    __syncwarp();
    for (int index = threadIdx.x; index < kTile * kTile; index += warpSize) {
        const int offset = (row + index / kTile) * dim + col + index % kTile;
        const float residual = __bfloat162float(gram[offset]);
        polynomial[offset] = __float2bfloat16_rn(fmaf(c, tile[index], b * residual));
    }
}

// Four output MMA tiles share the same input fragments in each K iteration.
// This reduces redundant Gram loads once the matrix is too large for the
// one-MMA-tile variant.
__global__ void fused_gram_polynomial_32_kernel(
    const __nv_bfloat16* gram, __nv_bfloat16* polynomial, int dim, float b, float c) {
    using namespace nvcuda;
    constexpr int kMma = 16;
    constexpr int kTile = 32;
    const int row = blockIdx.y * kTile;
    const int col = blockIdx.x * kTile;
    wmma::fragment<wmma::matrix_a, kMma, kMma, kMma, __nv_bfloat16, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, kMma, kMma, kMma, __nv_bfloat16, wmma::row_major> b0, b1;
    wmma::fragment<wmma::accumulator, kMma, kMma, kMma, float> p00, p01, p10, p11;
    wmma::fill_fragment(p00, 0.0f);
    wmma::fill_fragment(p01, 0.0f);
    wmma::fill_fragment(p10, 0.0f);
    wmma::fill_fragment(p11, 0.0f);
    for (int k = 0; k < dim; k += kMma) {
        wmma::load_matrix_sync(a0, gram + row * dim + k, dim);
        wmma::load_matrix_sync(a1, gram + (row + kMma) * dim + k, dim);
        wmma::load_matrix_sync(b0, gram + k * dim + col, dim);
        wmma::load_matrix_sync(b1, gram + k * dim + col + kMma, dim);
        wmma::mma_sync(p00, a0, b0, p00);
        wmma::mma_sync(p01, a0, b1, p01);
        wmma::mma_sync(p10, a1, b0, p10);
        wmma::mma_sync(p11, a1, b1, p11);
    }
    __shared__ float tile[kTile * kTile];
    wmma::store_matrix_sync(tile, p00, kTile, wmma::mem_row_major);
    wmma::store_matrix_sync(tile + kMma, p01, kTile, wmma::mem_row_major);
    wmma::store_matrix_sync(tile + kMma * kTile, p10, kTile, wmma::mem_row_major);
    wmma::store_matrix_sync(tile + kMma * kTile + kMma, p11, kTile, wmma::mem_row_major);
    __syncwarp();
    for (int index = threadIdx.x; index < kTile * kTile; index += warpSize) {
        const int offset = (row + index / kTile) * dim + col + index % kTile;
        const float residual = __bfloat162float(gram[offset]);
        polynomial[offset] = __float2bfloat16_rn(fmaf(c, tile[index], b * residual));
    }
}

// cuBLASLt accepts distinct residual (C) and result (D) pointers. ATen's
// addmm_out copies C into D before GEMM when beta is nonzero; keeping them
// separate removes two full-matrix copies from every NS iteration.
struct LtMatrix {
    cublasLtMatrixLayout_t layout = nullptr;

    LtMatrix(int64_t rows, int64_t cols, int64_t stride0, int64_t stride1) {
        const bool column_major = stride0 == 1;
        const auto order = column_major ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
        const int64_t ld = column_major ? stride1 : stride0;
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&layout, CUDA_R_16BF, rows, cols, ld));
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                              &order, sizeof(order)));
    }

    ~LtMatrix() {
        if (layout != nullptr) {
            cublasLtMatrixLayoutDestroy(layout);
        }
    }

    LtMatrix(const LtMatrix&) = delete;
    LtMatrix& operator=(const LtMatrix&) = delete;
};

struct LtMatmul {
    cublasLtMatmulDesc_t desc = nullptr;

    LtMatmul() {
        TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    }

    ~LtMatmul() {
        if (desc != nullptr) {
            cublasLtMatmulDescDestroy(desc);
        }
    }

    LtMatmul(const LtMatmul&) = delete;
    LtMatmul& operator=(const LtMatmul&) = delete;
};

void lt_addmm(const LtMatmul& operation,
              const torch::Tensor& lhs,
              const LtMatrix& lhs_layout,
              const torch::Tensor& rhs,
              const LtMatrix& rhs_layout,
              const torch::Tensor& residual,
              const LtMatrix& residual_layout,
              torch::Tensor& result,
              const LtMatrix& result_layout,
              float alpha,
              float beta,
              cudaStream_t stream) {
    TORCH_CUDABLAS_CHECK(cublasLtMatmul(
        at::cuda::getCurrentCUDABlasLtHandle(), operation.desc, &alpha,
        lhs.data_ptr<at::BFloat16>(), lhs_layout.layout, rhs.data_ptr<at::BFloat16>(),
        rhs_layout.layout, &beta, residual.data_ptr<at::BFloat16>(), residual_layout.layout,
        result.data_ptr<at::BFloat16>(), result_layout.layout, nullptr, nullptr, 0, stream));
}

void direct_gram(const torch::Tensor& x, torch::Tensor& gram) {
    const int m = static_cast<int>(x.size(0));
    const int n = static_cast<int>(x.size(1));
    const bool column_major = x.stride(0) == 1;
    const int ld = column_major ? m : n;
    const auto lhs_op = column_major ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto rhs_op = column_major ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float one = 1.0f;
    const float zero = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(), lhs_op, rhs_op, m, m, n,
                                      &one, x.data_ptr<at::BFloat16>(), CUDA_R_16BF, ld,
                                      x.data_ptr<at::BFloat16>(), CUDA_R_16BF, ld, &zero,
                                      gram.data_ptr<at::BFloat16>(), CUDA_R_16BF, m,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

torch::Tensor muon_ns_impl(torch::Tensor grad,
                           const std::vector<double>& coefficients,
                           int64_t ns_steps,
                           double eps,
                           bool fused_polynomial) {
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

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (fused_polynomial && x.is_non_overlapping_and_dense() && x.numel() > 0) {
        auto norm = x.norm();
        const int blocks = static_cast<int>(std::min<int64_t>((x.numel() + 255) / 256, 4096));
        normalize_bf16_kernel<<<blocks, 256, 0, stream>>>(x.data_ptr<at::BFloat16>(),
                                                          norm.data_ptr<at::BFloat16>(), x.numel(),
                                                          static_cast<at::BFloat16>(eps));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        x.div_(x.norm().clamp_min(eps));
    }

    const auto m = x.size(0);
    auto gram = torch::empty({m, m}, x.options());
    auto gram_update = torch::empty_like(gram);
    auto next_x = torch::empty_like(x);
    // The size gates were measured on RTX 5090. Keep the ATen GEMM path on
    // other architectures until their crossover points are profiled.
    const bool default_reduction = at::globalContext().allowBF16ReductionCuBLAS() ==
                                   at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK;
    const bool use_wmma = fused_polynomial && default_reduction &&
                          !at::globalContext().deterministicAlgorithms() &&
                          at::cuda::getCurrentDeviceProperties()->major == 12;
    const bool use_fused_polynomial_16 = use_wmma && m >= 16 && m <= 256 && m % 16 == 0;
    const bool use_fused_polynomial_32 = use_wmma && m >= 288 && m <= 512 && m % 32 == 0;
    const bool dense_matrix =
        (x.stride(1) == 1 && x.stride(0) == x.size(1)) || (x.stride(0) == 1 && x.stride(1) == m);
    const bool use_lt = use_wmma && dense_matrix && m >= 16 && m % 16 == 0 && x.size(1) % 16 == 0 &&
                        m <= std::numeric_limits<int>::max() &&
                        x.size(1) <= std::numeric_limits<int>::max();
    std::optional<LtMatmul> lt_operation;
    std::optional<LtMatrix> lt_gram;
    std::optional<LtMatrix> lt_x;
    if (use_lt) {
        lt_operation.emplace();
        lt_gram.emplace(m, m, gram.stride(0), gram.stride(1));
        lt_x.emplace(x.size(0), x.size(1), x.stride(0), x.stride(1));
    }
    for (int64_t step = 0; step < ns_steps; ++step) {
        if (use_lt) {
            direct_gram(x, gram);
        } else {
            at::mm_out(gram, x, x.transpose(0, 1));
        }
        if (use_fused_polynomial_16) {
            const int tiles = static_cast<int>(m / 16);
            fused_gram_polynomial_kernel<<<dim3(tiles, tiles), 32, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(gram.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(gram_update.data_ptr<at::BFloat16>()),
                static_cast<int>(m), static_cast<float>(b), static_cast<float>(c));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (use_fused_polynomial_32) {
            const int tiles = static_cast<int>(m / 32);
            fused_gram_polynomial_32_kernel<<<dim3(tiles, tiles), 32, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(gram.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(gram_update.data_ptr<at::BFloat16>()),
                static_cast<int>(m), static_cast<float>(b), static_cast<float>(c));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (use_lt) {
            lt_addmm(*lt_operation, gram, *lt_gram, gram, *lt_gram, gram, *lt_gram, gram_update,
                     *lt_gram, static_cast<float>(c), static_cast<float>(b), stream);
        } else {
            at::addmm_out(gram_update, gram, gram, gram, b, c);
        }
        if (use_lt) {
            lt_addmm(*lt_operation, gram_update, *lt_gram, x, *lt_x, x, *lt_x, next_x, *lt_x, 1.0f,
                     static_cast<float>(a), stream);
        } else {
            at::addmm_out(next_x, x, gram_update, x, a, 1.0);
        }
        std::swap(x, next_x);
    }

    return tall ? x.transpose(0, 1) : x;
}

} // namespace

} // namespace muon_ns
} // namespace astrai

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void prepare_kernel(const scalar_t* grad,
                               scalar_t* momentum_buffer,
                               at::BFloat16* update,
                               float* partial_squares,
                               int64_t count,
                               float momentum,
                               float momentum_weight,
                               bool nesterov) {
    __shared__ float sums[kThreads];
    float local_sum = 0.0f;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float g = static_cast<float>(grad[index]);
        const float old_buffer = static_cast<float>(momentum_buffer[index]);
        const scalar_t new_buffer =
            static_cast<scalar_t>(old_buffer + (g - old_buffer) * momentum_weight);
        momentum_buffer[index] = new_buffer;
        const float value = nesterov ? g + (static_cast<float>(new_buffer) - g) * momentum
                                     : static_cast<float>(new_buffer);
        const scalar_t rounded_update = static_cast<scalar_t>(value);
        const at::BFloat16 bf16_update =
            static_cast<at::BFloat16>(static_cast<float>(rounded_update));
        update[index] = bf16_update;
        const float x = static_cast<float>(bf16_update);
        local_sum += x * x;
    }
    sums[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sums[threadIdx.x] += sums[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_squares[blockIdx.x] = sums[0];
    }
}

__global__ void
sum_partials_kernel(const float* partials, float* totals, int64_t count, int index) {
    __shared__ float sums[kThreads];
    float value = 0.0f;
    for (int64_t offset = threadIdx.x; offset < count; offset += kThreads) {
        value += partials[offset];
    }
    sums[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        totals[index] = sums[0];
    }
}

__global__ void
normalize_kernel(at::BFloat16* update, const float* norm_squared, int64_t count, float eps) {
    const at::BFloat16 rounded_norm = static_cast<at::BFloat16>(sqrtf(norm_squared[0]));
    const float divisor = fmaxf(static_cast<float>(rounded_norm), eps);
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        update[index] = static_cast<at::BFloat16>(static_cast<float>(update[index]) / divisor);
    }
}

__global__ void copy_residuals_kernel(const at::BFloat16* gram,
                                      at::BFloat16* polynomial,
                                      int64_t gram_count,
                                      const at::BFloat16* x,
                                      at::BFloat16* next_x,
                                      int64_t x_count) {
    const int64_t count = gram_count > x_count ? gram_count : x_count;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        if (index < gram_count) {
            polynomial[index] = gram[index];
        }
        if (index < x_count) {
            next_x[index] = x[index];
        }
    }
}

template <typename scalar_t>
__global__ void finish_kernel(scalar_t* param,
                              const at::BFloat16* update,
                              int64_t count,
                              float decay_factor,
                              float adjusted_lr) {
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float decayed_float = __fmul_rn(static_cast<float>(param[index]), decay_factor);
        const scalar_t decayed = static_cast<scalar_t>(decayed_float);
        const float next = __fadd_rn(static_cast<float>(decayed),
                                     __fmul_rn(-adjusted_lr, static_cast<float>(update[index])));
        param[index] = static_cast<scalar_t>(next);
    }
}

void check_local_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.dim() == 2 && tensor.is_contiguous(), name,
                " must be a contiguous CUDA matrix");
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16 || tensor.scalar_type() == at::kFloat, name,
                " must be BF16 or FP32");
}

int blocks_for(int64_t count) {
    return static_cast<int>(std::min<int64_t>(4096, (count + kThreads - 1) / kThreads));
}

std::tuple<torch::Tensor, torch::Tensor>
prepare(const torch::Tensor& grad, torch::Tensor momentum_buffer, double momentum, bool nesterov) {
    check_local_tensor(grad, "grad");
    check_local_tensor(momentum_buffer, "momentum_buffer");
    TORCH_CHECK(grad.sizes() == momentum_buffer.sizes() &&
                    grad.scalar_type() == momentum_buffer.scalar_type() &&
                    grad.device() == momentum_buffer.device(),
                "grad and momentum_buffer must have matching shape, dtype and device");
    TORCH_CHECK(grad.numel() > 0, "local shard must be nonempty");
    TORCH_CHECK(momentum >= 0.0 && momentum <= 1.0, "momentum must be in [0, 1]");
    const at::cuda::OptionalCUDAGuard guard(device_of(grad));
    auto update = torch::empty(grad.sizes(), grad.options().dtype(at::kBFloat16));
    const int blocks = blocks_for(grad.numel());
    auto partials = torch::empty({blocks}, grad.options().dtype(at::kFloat));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (grad.scalar_type() == at::kBFloat16) {
        prepare_kernel<<<blocks, kThreads, 0, stream>>>(
            grad.data_ptr<at::BFloat16>(), momentum_buffer.data_ptr<at::BFloat16>(),
            update.data_ptr<at::BFloat16>(), partials.data_ptr<float>(), grad.numel(),
            static_cast<float>(momentum), static_cast<float>(1.0 - momentum), nesterov);
    } else {
        prepare_kernel<<<blocks, kThreads, 0, stream>>>(
            grad.data_ptr<float>(), momentum_buffer.data_ptr<float>(),
            update.data_ptr<at::BFloat16>(), partials.data_ptr<float>(), grad.numel(),
            static_cast<float>(momentum), static_cast<float>(1.0 - momentum), nesterov);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {update, partials};
}

torch::Tensor reduce_partials(const std::vector<torch::Tensor>& partials) {
    TORCH_CHECK(!partials.empty(), "partials must contain at least one tensor");
    const auto& first = partials.front();
    TORCH_CHECK(first.is_cuda() && first.scalar_type() == at::kFloat,
                "partials must be CUDA FP32 tensors");
    const at::cuda::OptionalCUDAGuard guard(device_of(first));
    auto totals = torch::empty({static_cast<int64_t>(partials.size())}, first.options());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (size_t index = 0; index < partials.size(); ++index) {
        const auto& part = partials[index];
        TORCH_CHECK(part.is_cuda() && part.device() == first.device() &&
                        part.scalar_type() == at::kFloat && part.is_contiguous() &&
                        part.dim() == 1 && part.numel() > 0,
                    "each partial tensor must be a nonempty contiguous CUDA FP32 vector");
        sum_partials_kernel<<<1, kThreads, 0, stream>>>(part.data_ptr<float>(),
                                                        totals.data_ptr<float>(), part.numel(),
                                                        static_cast<int>(index));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return totals;
}

void normalize_(torch::Tensor update, const torch::Tensor& norm_squared, double eps) {
    TORCH_CHECK(update.is_cuda() && update.is_contiguous() && update.scalar_type() == at::kBFloat16,
                "update must be a contiguous CUDA BF16 tensor");
    TORCH_CHECK(norm_squared.is_cuda() && norm_squared.device() == update.device() &&
                    norm_squared.scalar_type() == at::kFloat && norm_squared.numel() == 1,
                "norm_squared must be a CUDA FP32 scalar on update's device");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
    const at::cuda::OptionalCUDAGuard guard(device_of(update));
    normalize_kernel<<<blocks_for(update.numel()), kThreads, 0,
                       at::cuda::getCurrentCUDAStream().stream()>>>(
        update.data_ptr<at::BFloat16>(), norm_squared.data_ptr<float>(), update.numel(),
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_ns_x(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 &&
                    x.stride(0) == 1 && x.stride(1) == x.size(0),
                "x must be a transposed contiguous CUDA BF16 matrix");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(0) <= std::numeric_limits<int>::max() &&
                    x.size(1) <= std::numeric_limits<int>::max(),
                "x dimensions exceed cuBLAS limits");
}

void gram_(const torch::Tensor& x, torch::Tensor gram) {
    check_ns_x(x);
    const int small = static_cast<int>(x.size(0));
    const int local_rows = static_cast<int>(x.size(1));
    TORCH_CHECK(gram.is_cuda() && gram.device() == x.device() && gram.is_contiguous() &&
                    gram.scalar_type() == at::kBFloat16 &&
                    gram.sizes() == at::IntArrayRef({small, small}),
                "gram must be a matching contiguous CUDA BF16 matrix");
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    const float one = 1.0f;
    const float zero = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        at::cuda::getCurrentCUDABlasHandle(), CUBLAS_OP_N, CUBLAS_OP_T, small, small, local_rows,
        &one, x.data_ptr<at::BFloat16>(), CUDA_R_16BF, small, x.data_ptr<at::BFloat16>(),
        CUDA_R_16BF, small, &zero, gram.data_ptr<at::BFloat16>(), CUDA_R_16BF, small,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void ns_update_(const torch::Tensor& x,
                const torch::Tensor& gram,
                torch::Tensor polynomial,
                torch::Tensor next_x,
                double a,
                double b,
                double c) {
    check_ns_x(x);
    check_ns_x(next_x);
    const int small = static_cast<int>(x.size(0));
    const int local_rows = static_cast<int>(x.size(1));
    TORCH_CHECK(next_x.sizes() == x.sizes() && next_x.device() == x.device(),
                "next_x must match x");
    for (const auto& tensor : {gram, polynomial}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == x.device() && tensor.is_contiguous() &&
                        tensor.scalar_type() == at::kBFloat16 &&
                        tensor.sizes() == at::IntArrayRef({small, small}),
                    "Gram scratch must be a matching contiguous CUDA BF16 matrix");
    }
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    copy_residuals_kernel<<<blocks_for(std::max(gram.numel(), x.numel())), kThreads, 0, stream>>>(
        gram.data_ptr<at::BFloat16>(), polynomial.data_ptr<at::BFloat16>(), gram.numel(),
        x.data_ptr<at::BFloat16>(), next_x.data_ptr<at::BFloat16>(), x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float b_value = static_cast<float>(b);
    const float c_value = static_cast<float>(c);
    const auto handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, small, small, small,
                                      &c_value, gram.data_ptr<at::BFloat16>(), CUDA_R_16BF, small,
                                      gram.data_ptr<at::BFloat16>(), CUDA_R_16BF, small, &b_value,
                                      polynomial.data_ptr<at::BFloat16>(), CUDA_R_16BF, small,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    const float one = 1.0f;
    const float a_value = static_cast<float>(a);
    TORCH_CUDABLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, small, local_rows, small,
                                      &one, polynomial.data_ptr<at::BFloat16>(), CUDA_R_16BF, small,
                                      x.data_ptr<at::BFloat16>(), CUDA_R_16BF, small, &a_value,
                                      next_x.data_ptr<at::BFloat16>(), CUDA_R_16BF, small,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void finish_(torch::Tensor param,
             const torch::Tensor& update,
             double lr,
             double weight_decay,
             double adjusted_lr) {
    check_local_tensor(param, "param");
    TORCH_CHECK(update.is_cuda() && update.device() == param.device() &&
                    update.scalar_type() == at::kBFloat16 && update.sizes() == param.sizes() &&
                    update.is_contiguous(),
                "update must be a matching contiguous CUDA BF16 matrix");
    const at::cuda::OptionalCUDAGuard guard(device_of(param));
    const float decay_factor = static_cast<float>(1.0 - lr * weight_decay);
    const float step = static_cast<float>(adjusted_lr);
    const int blocks = blocks_for(param.numel());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (param.scalar_type() == at::kBFloat16) {
        finish_kernel<<<blocks, kThreads, 0, stream>>>(param.data_ptr<at::BFloat16>(),
                                                       update.data_ptr<at::BFloat16>(),
                                                       param.numel(), decay_factor, step);
    } else {
        finish_kernel<<<blocks, kThreads, 0, stream>>>(param.data_ptr<float>(),
                                                       update.data_ptr<at::BFloat16>(),
                                                       param.numel(), decay_factor, step);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare", &prepare, "Fuse local Muon momentum and NS input preparation");
    m.def("reduce_partials", &reduce_partials, "Reduce local Muon norm partials");
    m.def("normalize_", &normalize_, "Normalize a local Muon BF16 shard");
    m.def("gram_", &gram_, "Compute a local BF16 Muon Gram matrix with cuBLAS");
    m.def("ns_update_", &ns_update_, "Apply one BF16 local Muon NS update");
    m.def("finish_", &finish_, "Fuse local Muon decay and parameter update");
    m.def("muon_ns", &astrai::muon_ns::muon_ns_impl, py::arg("grad"), py::arg("ns_coefficients"),
          py::arg("ns_steps"), py::arg("eps"), py::arg("fused_polynomial") = true,
          "Run Muon's Newton-Schulz orthogonalization in one extension call");
}
