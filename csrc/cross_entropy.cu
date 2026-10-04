#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <tuple>

namespace {
constexpr int kThreads = 256;

template <bool Maximum> __device__ float combine(float a, float b) {
    if constexpr (Maximum) {
        return isnan(a) || isnan(b) ? NAN : fmaxf(a, b);
    }
    return a + b;
}

template <bool Maximum> __device__ float reduce(float value) {
    __shared__ float warps[8];
    for (int offset = 16; offset; offset /= 2) {
        value = combine<Maximum>(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    if (threadIdx.x % 32 == 0) {
        warps[threadIdx.x / 32] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8 ? warps[threadIdx.x] : (Maximum ? -INFINITY : 0.0f);
    if (threadIdx.x < 32) {
        for (int offset = 16; offset; offset /= 2) {
            value = combine<Maximum>(value, __shfl_down_sync(0xffffffff, value, offset));
        }
        if (threadIdx.x == 0) {
            warps[0] = value;
        }
    }
    __syncthreads();
    value = warps[0];
    __syncthreads();
    return value;
}

template <typename scalar_t>
__global__ void forward_rows(scalar_t* logits,
                             const int64_t* targets,
                             float* maxima,
                             float* log_sums,
                             float* losses,
                             int64_t vocab,
                             int64_t ignore_index,
                             float smoothing) {
    const int64_t row = blockIdx.x;
    const int64_t target = targets[row];
    CUDA_KERNEL_ASSERT(target == ignore_index || (target >= 0 && target < vocab));
    scalar_t* values = logits + row * vocab;
    if (target == ignore_index) {
        if (threadIdx.x == 0) {
            losses[row] = maxima[row] = log_sums[row] = 0.0f;
        }
        return;
    }
    float maximum = -INFINITY;
    for (int64_t col = threadIdx.x; col < vocab; col += kThreads) {
        maximum = combine<true>(maximum, float(values[col]));
    }
    maximum = reduce<true>(maximum);
    float sum = 0.0f, shifted_sum = 0.0f;
    for (int64_t col = threadIdx.x; col < vocab; col += kThreads) {
        const float shifted = float(values[col]) - maximum;
        sum += expf(shifted);
        if (smoothing != 0.0f) {
            shifted_sum += shifted;
        }
    }
    const float log_sum = logf(reduce<false>(sum));
    if (smoothing != 0.0f) {
        shifted_sum = reduce<false>(shifted_sum);
    }
    if (threadIdx.x == 0) {
        maxima[row] = maximum;
        log_sums[row] = log_sum;
        losses[row] = (1.0f - smoothing) * (maximum - float(values[target])) + log_sum -
                      smoothing * shifted_sum / float(vocab);
    }
}

template <typename scalar_t>
__global__ void backward_rows(const scalar_t* logits,
                              const int64_t* targets,
                              const float* maxima,
                              const float* log_sums,
                              const float* grad_loss,
                              scalar_t* grad_logits,
                              int64_t vocab,
                              int64_t ignore_index,
                              float smoothing) {
    const int64_t row = blockIdx.x;
    const int64_t target = targets[row];
    for (int64_t col = threadIdx.x; col < vocab; col += kThreads) {
        float grad = 0.0f;
        if (target != ignore_index) {
            const float p = expf((float(logits[row * vocab + col]) - maxima[row]) - log_sums[row]);
            grad = (p - smoothing / float(vocab) - (col == target ? 1.0f - smoothing : 0.0f)) *
                   grad_loss[0];
        }
        grad_logits[row * vocab + col] = scalar_t(grad);
    }
}

void check_inputs(const torch::Tensor& logits, const torch::Tensor& targets, double smoothing) {
    TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.is_contiguous(),
                "logits must be a contiguous CUDA matrix");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16 || logits.scalar_type() == at::kHalf ||
                    logits.scalar_type() == at::kFloat,
                "logits must be BF16, FP16, or FP32");
    TORCH_CHECK(logits.size(0) > 0 && logits.size(1) > 0, "logits must be nonempty");
    TORCH_CHECK(targets.device() == logits.device() && targets.dim() == 1 &&
                    targets.is_contiguous() && targets.scalar_type() == at::kLong &&
                    targets.numel() == logits.size(0),
                "targets must be a matching CUDA int64 vector");
    TORCH_CHECK(std::isfinite(smoothing) && smoothing >= 0.0 && smoothing <= 1.0,
                "label_smoothing must be in [0, 1]");
}

void launch_forward(const torch::Tensor& logits,
                    const torch::Tensor& targets,
                    torch::Tensor& maxima,
                    torch::Tensor& log_sums,
                    torch::Tensor& losses,
                    int64_t ignore_index,
                    double smoothing) {
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(), "ce_forward", [&] {
            forward_rows<scalar_t><<<logits.size(0), kThreads, 0, stream>>>(
                logits.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(), maxima.data_ptr<float>(),
                log_sums.data_ptr<float>(), losses.data_ptr<float>(), logits.size(1), ignore_index,
                smoothing);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor& logits,
                                                                const torch::Tensor& targets,
                                                                int64_t ignore_index,
                                                                double smoothing) {
    check_inputs(logits, targets, smoothing);
    const c10::cuda::CUDAGuard guard(logits.device());
    auto options = logits.options().dtype(at::kFloat);
    auto maxima = torch::empty({logits.size(0)}, options);
    auto log_sums = torch::empty_like(maxima);
    auto losses = torch::empty_like(maxima);
    launch_forward(logits, targets, maxima, log_sums, losses, ignore_index, smoothing);
    return {losses.sum(), maxima, log_sums};
}

torch::Tensor backward(const torch::Tensor& logits,
                       const torch::Tensor& targets,
                       const torch::Tensor& maxima,
                       const torch::Tensor& log_sums,
                       const torch::Tensor& grad_loss,
                       int64_t ignore_index,
                       double smoothing) {
    check_inputs(logits, targets, smoothing);
    const c10::cuda::CUDAGuard guard(logits.device());
    for (const auto& t : {maxima, log_sums}) {
        TORCH_CHECK(t.device() == logits.device() && t.scalar_type() == at::kFloat &&
                        t.is_contiguous() && t.numel() == logits.size(0),
                    "invalid row statistics");
    }
    TORCH_CHECK(grad_loss.device() == logits.device() && grad_loss.scalar_type() == at::kFloat &&
                    grad_loss.is_contiguous() && grad_loss.numel() == 1,
                "invalid grad_loss");
    auto grad = torch::empty_like(logits);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(), "ce_backward", [&] {
            backward_rows<<<logits.size(0), kThreads, 0, stream>>>(
                logits.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(), maxima.data_ptr<float>(),
                log_sums.data_ptr<float>(), grad_loss.data_ptr<float>(), grad.data_ptr<scalar_t>(),
                logits.size(1), ignore_index, smoothing);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}

// Recompute vocabulary tiles in backward. Each dW tile reduces ALL tokens
// in one GEMM (FP32 accumulation, one final BF16/FP16 cast), so there is no
// [vocab, hidden] FP32 gradient buffer or repeated low-precision accumulation.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> linear_forward(const torch::Tensor& hidden,
                                                                       const torch::Tensor& weight,
                                                                       const torch::Tensor& targets,
                                                                       int64_t ignore_index,
                                                                       double smoothing,
                                                                       int64_t chunk_size) {
    check_inputs(hidden, targets, smoothing);
    TORCH_CHECK(weight.device() == hidden.device() &&
                    weight.scalar_type() == hidden.scalar_type() && weight.dim() == 2 &&
                    weight.is_contiguous() && weight.size(1) == hidden.size(1) &&
                    weight.size(0) > 0,
                "weight must be a matching contiguous [vocab, hidden] matrix");
    TORCH_CHECK(chunk_size > 0 && weight.size(0) <= INT_MAX && hidden.size(1) <= INT_MAX &&
                    hidden.size(0) <= INT_MAX && chunk_size <= INT_MAX,
                "invalid GEMM dimensions or chunk_size");
    const c10::cuda::CUDAGuard guard(hidden.device());
    const auto options = hidden.options().dtype(at::kFloat);
    const int64_t rows = hidden.size(0), chunk = std::min(chunk_size, rows);
    auto logits = torch::empty({chunk, weight.size(0)}, hidden.options());
    auto maxima = torch::empty({rows}, options);
    auto log_sums = torch::empty_like(maxima);
    auto losses = torch::empty_like(maxima);
    for (int64_t start = 0; start < rows; start += chunk) {
        const int64_t count = std::min(chunk, rows - start);
        auto x = hidden.narrow(0, start, count);
        auto z = logits.narrow(0, 0, count);
        auto y = targets.narrow(0, start, count);
        auto l = losses.narrow(0, start, count);
        auto m = maxima.narrow(0, start, count);
        auto s = log_sums.narrow(0, start, count);
        at::mm_out(z, x, weight.t());
        launch_forward(z, y, m, s, l, ignore_index, smoothing);
    }
    return {losses.sum(), maxima, log_sums};
}

template <typename scalar_t>
__global__ void tile_gradient(scalar_t* logits,
                              const int64_t* targets,
                              const float* maxima,
                              const float* log_sums,
                              const float* scale,
                              int64_t stride,
                              int64_t width,
                              int64_t offset,
                              int64_t vocab,
                              int64_t ignore_index,
                              float smoothing) {
    const int64_t row = blockIdx.x;
    const int64_t target = targets[row];
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        float grad = 0.0f;
        if (target != ignore_index) {
            const float p = expf((float(logits[row * stride + col]) - maxima[row]) - log_sums[row]);
            grad = (p - smoothing / float(vocab) -
                    (col + offset == target ? 1.0f - smoothing : 0.0f)) *
                   scale[0];
        }
        logits[row * stride + col] = scalar_t(grad);
    }
}

void accumulate_hidden(const torch::Tensor& dz,
                       const torch::Tensor& weight,
                       torch::Tensor& dx,
                       bool first) {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    const auto dtype = weight.scalar_type() == at::kBFloat16 ? CUDA_R_16BF
                       : weight.scalar_type() == at::kHalf   ? CUDA_R_16F
                                                             : CUDA_R_32F;
    const float alpha = 1.0f, beta = first ? 0.0f : 1.0f;
    // Column-major dx[H,T] = W[H,Vtile] @ dZ[Vtile,T]. Only [T,H]
    // accumulates across tiles in FP32 (12 MiB for T=2048,H=1536).
    const auto status =
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, weight.size(1), dz.size(0), weight.size(0),
                     &alpha, weight.data_ptr(), dtype, weight.size(1), dz.data_ptr(), dtype,
                     dz.stride(0), &beta, dx.data_ptr(), CUDA_R_32F, weight.size(1),
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CE hidden-gradient GEMM failed: ", int(status));
}

std::tuple<torch::Tensor, torch::Tensor> linear_backward(const torch::Tensor& hidden,
                                                         const torch::Tensor& weight,
                                                         const torch::Tensor& targets,
                                                         const torch::Tensor& maxima,
                                                         const torch::Tensor& log_sums,
                                                         const torch::Tensor& scale,
                                                         int64_t ignore_index,
                                                         double smoothing,
                                                         bool need_hidden,
                                                         bool need_weight) {
    check_inputs(hidden, targets, smoothing);
    const c10::cuda::CUDAGuard guard(hidden.device());
    TORCH_CHECK(weight.device() == hidden.device() &&
                    weight.scalar_type() == hidden.scalar_type() && weight.dim() == 2 &&
                    weight.is_contiguous() && weight.size(1) == hidden.size(1),
                "invalid saved weight");
    for (const auto& t : {maxima, log_sums}) {
        TORCH_CHECK(t.device() == hidden.device() && t.scalar_type() == at::kFloat &&
                        t.is_contiguous() && t.numel() == hidden.size(0),
                    "invalid saved row statistics");
    }
    TORCH_CHECK(scale.device() == hidden.device() && scale.scalar_type() == at::kFloat &&
                    scale.is_contiguous() && scale.numel() == 1,
                "invalid loss scale");
    const int64_t vocab = weight.size(0);
    // Target a 128 MiB private tile while keeping GEMMs wide enough.
    const int64_t budget = 128 * 1024 * 1024 / hidden.element_size() / hidden.size(0);
    const int64_t tile =
        std::min(vocab, std::max<int64_t>(256, std::min<int64_t>(16384, budget / 256 * 256)));
    auto scratch = torch::empty({hidden.size(0), tile}, hidden.options());
    auto dx = need_hidden ? torch::empty(hidden.sizes(), hidden.options().dtype(at::kFloat))
                          : torch::empty({0}, hidden.options().dtype(at::kFloat));
    auto dw = need_weight ? torch::empty_like(weight) : torch::empty({0}, weight.options());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (int64_t offset = 0; offset < vocab; offset += tile) {
        const int64_t width = std::min(tile, vocab - offset);
        auto w = weight.narrow(0, offset, width);
        auto z = scratch.narrow(1, 0, width);
        at::mm_out(z, hidden, w.t());
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, hidden.scalar_type(), "ce_tile", [&] {
                tile_gradient<<<hidden.size(0), kThreads, 0, stream>>>(
                    z.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(), maxima.data_ptr<float>(),
                    log_sums.data_ptr<float>(), scale.data_ptr<float>(), z.stride(0), width, offset,
                    vocab, ignore_index, smoothing);
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (need_hidden) {
            accumulate_hidden(z, w, dx, offset == 0);
        }
        if (need_weight) {
            auto dw_tile = dw.narrow(0, offset, width);
            at::mm_out(dw_tile, z.t(), hidden);
        }
    }
    return {dx.to(hidden.scalar_type()), dw};
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross entropy sum with FP32 reductions");
    m.def("backward", &backward, "Cross entropy backward from low-precision logits");
    m.def("linear_forward", &linear_forward, "Chunked linear CE sum and row statistics");
    m.def("linear_backward", &linear_backward, "Vocabulary-tiled linear CE gradients");
}
