#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <torch/extension.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <tuple>
#include <type_traits>

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

template <typename scalar_t, bool ComputeGradient = false>
__global__ void forward_rows_online(scalar_t* logits, const int64_t* targets, float* maxima,
                                    float* log_sums, float* losses, int64_t vocab,
                                    int64_t ignore_index, float smoothing) {
    const int64_t row = blockIdx.x;
    const int64_t target = targets[row];
    CUDA_KERNEL_ASSERT(target == ignore_index || (target >= 0 && target < vocab));
    scalar_t* values = logits + row * vocab;
    if (target == ignore_index) {
        if (threadIdx.x == 0) {
            losses[row] = maxima[row] = log_sums[row] = 0.0f;
        }
        if constexpr (ComputeGradient) {
            for (int64_t col = threadIdx.x; col < vocab; col += kThreads) {
                values[col] = scalar_t(0.0f);
            }
        }
        return;
    }
    // Each lane maintains a stable normalizer for its strided vocabulary slice.
    // One exponential and a predicated select avoid a divergent new-maximum
    // branch.
    float lane_maximum = -INFINITY;
    float lane_sum = 0.0f, lane_shifted_sum = 0.0f;
    int64_t lane_count = 0;
    if (threadIdx.x < vocab) {
        lane_maximum = float(values[threadIdx.x]);
        lane_sum = 1.0f;
        lane_count = 1;
    }
    for (int64_t col = threadIdx.x + kThreads; col < vocab; col += kThreads) {
        const float value = float(values[col]);
        const float difference = value - lane_maximum;
        const float factor = expf(-fabsf(difference));
        const bool higher = value > lane_maximum;
        lane_sum = higher ? fmaf(lane_sum, factor, 1.0f) : lane_sum + factor;
        if (smoothing != 0.0f) {
            lane_shifted_sum += higher ? -float(lane_count) * difference : difference;
        }
        lane_maximum = combine<true>(lane_maximum, value);
        ++lane_count;
    }
    const float maximum = reduce<true>(lane_maximum);
    const float sum = lane_count ? lane_sum * expf(lane_maximum - maximum) : 0.0f;
    const float log_sum = logf(reduce<false>(sum));
    float shifted_sum = 0.0f;
    if (smoothing != 0.0f) {
        if (lane_count) {
            shifted_sum = lane_shifted_sum + float(lane_count) * (lane_maximum - maximum);
        }
        shifted_sum = reduce<false>(shifted_sum);
    }
    if (threadIdx.x == 0) {
        maxima[row] = maximum;
        log_sums[row] = log_sum;
        losses[row] = (1.0f - smoothing) * (maximum - float(values[target])) + log_sum -
                      smoothing * shifted_sum / float(vocab);
    }
    if constexpr (ComputeGradient) {
        // Loss reads the target logit before any thread overwrites the row.
        __syncthreads();
        for (int64_t col = threadIdx.x; col < vocab; col += kThreads) {
            const float probability = expf((float(values[col]) - maximum) - log_sum);
            // Raise FP16 subnormal gradients into its normal exponent range.
            // A power of two is exact and 2^15 keeps gradients in [-1, 1] finite.
            const float projection_scale = std::is_same<scalar_t, at::Half>::value ? 32768.0f : 1.0f;
            values[col] = scalar_t((probability - smoothing / float(vocab) -
                                   (col == target ? 1.0f - smoothing : 0.0f)) * projection_scale);
        }
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

void launch_forward(const torch::Tensor& logits, const torch::Tensor& targets,
                    torch::Tensor& maxima, torch::Tensor& log_sums, torch::Tensor& losses,
                    int64_t ignore_index, double smoothing, bool compute_gradient = false) {
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t rows = logits.size(0);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(), "ce_forward", [&] {
            if (compute_gradient) {
                forward_rows_online<scalar_t, true><<<rows, kThreads, 0, stream>>>(
                    logits.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(),
                    maxima.data_ptr<float>(), log_sums.data_ptr<float>(), losses.data_ptr<float>(),
                    logits.size(1), ignore_index, smoothing);
            } else {
                forward_rows_online<scalar_t><<<rows, kThreads, 0, stream>>>(
                    logits.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(),
                    maxima.data_ptr<float>(), log_sums.data_ptr<float>(), losses.data_ptr<float>(),
                    logits.size(1), ignore_index, smoothing);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void accumulate_hidden(const torch::Tensor& dz, const torch::Tensor& weight, torch::Tensor& dx,
                       bool first) {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    const auto dtype = weight.scalar_type() == at::kBFloat16 ? CUDA_R_16BF
                       : weight.scalar_type() == at::kHalf   ? CUDA_R_16F
                                                             : CUDA_R_32F;
    const float alpha = 1.0f, beta = first ? 0.0f : 1.0f;
    // Column-major dx[H,T] = W[H,V] @ dZ[V,T], with FP32 output.
    const auto status =
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, weight.size(1), dz.size(0), weight.size(0),
                     &alpha, weight.data_ptr(), dtype, weight.size(1), dz.data_ptr(), dtype,
                     dz.stride(0), &beta, dx.data_ptr(), CUDA_R_32F, weight.size(1),
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CE hidden-gradient GEMM failed: ", int(status));
}

void accumulate_weight(const torch::Tensor& dz, const torch::Tensor& hidden, torch::Tensor& dw,
                       bool first) {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    const auto dtype = hidden.scalar_type() == at::kBFloat16 ? CUDA_R_16BF
                       : hidden.scalar_type() == at::kHalf   ? CUDA_R_16F
                                                             : CUDA_R_32F;
    const float alpha = 1.0f, beta = first ? 0.0f : 1.0f;
    // Row-major dW[V,H] is column-major dW[H,V] = X[H,T] @ dZ[T,V].
    const auto status =
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, hidden.size(1), dz.size(1), hidden.size(0),
                     &alpha, hidden.data_ptr(), dtype, hidden.size(1), dz.data_ptr(), dtype,
                     dz.stride(0), &beta, dw.data_ptr<float>(), CUDA_R_32F, hidden.size(1),
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CE weight-gradient GEMM failed: ", int(status));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
linear_forward(const torch::Tensor& hidden, const torch::Tensor& weight,
                           const torch::Tensor& targets, int64_t ignore_index, double smoothing,
                           int64_t chunk_size, bool need_hidden, bool need_weight) {
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
    auto selected_targets = targets;
    torch::Tensor valid_rows;
    // Large masked heads can avoid projecting ignored tokens. Dynamic nonzero
    // synchronizes its row count, so graph capture keeps the fixed-shape path.
    const double projection_work = double(hidden.size(0)) * hidden.size(1) * weight.size(0);
    if (projection_work >= double(uint64_t(1) << 39) &&
        c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None) {
        auto indices = torch::nonzero(targets.ne(ignore_index)).flatten();
        if (indices.numel() < hidden.size(0)) {
            valid_rows = indices;
            selected_targets = targets.index_select(0, indices);
        }
    }
    const int64_t rows = selected_targets.size(0), chunk = std::min(chunk_size, rows);
    auto dx = need_hidden ? (valid_rows.defined() ? torch::zeros(hidden.sizes(), options)
                                                 : torch::empty(hidden.sizes(), options))
                          : torch::empty({0}, options);
    auto dw = need_weight ? torch::empty(weight.sizes(), options) : torch::empty({0}, options);
    if (rows == 0) {
        if (need_weight) dw.zero_();
        return {torch::zeros({}, options), dx, dw};
    }
    auto selected_dx = need_hidden && valid_rows.defined()
                           ? torch::empty({chunk, hidden.size(1)}, options) : dx;
    auto logits = torch::empty({chunk, weight.size(0)}, hidden.options());
    auto maxima = torch::empty({rows}, options);
    auto log_sums = torch::empty_like(maxima);
    auto losses = torch::empty_like(maxima);
    for (int64_t start = 0; start < rows; start += chunk) {
        const int64_t count = std::min(chunk, rows - start);
        auto row_indices = valid_rows.defined() ? valid_rows.narrow(0, start, count)
                                                : torch::Tensor();
        auto x = row_indices.defined() ? hidden.index_select(0, row_indices)
                                       : hidden.narrow(0, start, count);
        auto z = logits.narrow(0, 0, count);
        auto y = selected_targets.narrow(0, start, count);
        auto l = losses.narrow(0, start, count);
        auto m = maxima.narrow(0, start, count);
        auto s = log_sums.narrow(0, start, count);
        at::mm_out(z, x, weight.t());
        launch_forward(z, y, m, s, l, ignore_index, smoothing, need_hidden || need_weight);
        if (need_hidden || need_weight) {
            if (need_hidden) {
                auto dx_chunk = selected_dx.narrow(0, valid_rows.defined() ? 0 : start, count);
                accumulate_hidden(z, weight, dx_chunk, true);
                if (row_indices.defined()) {
                    dx.index_copy_(0, row_indices, dx_chunk);
                }
            }
            if (need_weight) {
                accumulate_weight(z, x, dw, start == 0);
            }
        }
    }
    return {losses.sum(), dx, dw};
}

template <typename scalar_t>
__global__ void scale_precomputed(const float* __restrict__ raw, const float* __restrict__ scale,
                                  scalar_t* __restrict__ output, int64_t elements) {
    const float projection_scale = std::is_same<scalar_t, at::Half>::value ? 32768.0f : 1.0f;
    const float factor = scale[0] / projection_scale;
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < elements;
         index += int64_t(gridDim.x) * blockDim.x) {
        output[index] = scalar_t(raw[index] * factor);
    }
}

std::tuple<torch::Tensor, torch::Tensor> linear_backward(const torch::Tensor& dx_raw,
                                                                     const torch::Tensor& dw_raw,
                                                                     const torch::Tensor& scale,
                                                                     const torch::Tensor& hidden,
                                                                     const torch::Tensor& weight) {
    const c10::cuda::CUDAGuard guard(hidden.device());
    TORCH_CHECK(scale.device() == hidden.device() && scale.scalar_type() == at::kFloat &&
                    scale.is_contiguous() && scale.numel() == 1,
                "invalid loss scale");
    TORCH_CHECK(dx_raw.device() == hidden.device() && dw_raw.device() == hidden.device() &&
                    dx_raw.scalar_type() == at::kFloat && dw_raw.scalar_type() == at::kFloat &&
                    dx_raw.is_contiguous() && dw_raw.is_contiguous(),
                "invalid precomputed gradients");
    TORCH_CHECK((dx_raw.numel() == 0 || dx_raw.sizes() == hidden.sizes()) &&
                    (dw_raw.numel() == 0 || dw_raw.sizes() == weight.sizes()) &&
                    hidden.scalar_type() == weight.scalar_type(),
                "invalid precomputed gradient shapes or output dtype");
    auto dx = dx_raw.numel() ? torch::empty_like(hidden) : torch::empty({0}, hidden.options());
    auto dw = dw_raw.numel() ? torch::empty_like(weight) : torch::empty({0}, weight.options());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, hidden.scalar_type(), "ce_scale_precomputed", [&] {
            if (dx_raw.numel()) {
                const int blocks =
                    std::min<int64_t>(65535, (dx_raw.numel() + kThreads - 1) / kThreads);
                scale_precomputed<scalar_t><<<blocks, kThreads, 0, stream>>>(
                    dx_raw.data_ptr<float>(), scale.data_ptr<float>(), dx.data_ptr<scalar_t>(),
                    dx_raw.numel());
            }
            if (dw_raw.numel()) {
                const int blocks =
                    std::min<int64_t>(65535, (dw_raw.numel() + kThreads - 1) / kThreads);
                scale_precomputed<scalar_t><<<blocks, kThreads, 0, stream>>>(
                    dw_raw.data_ptr<float>(), scale.data_ptr<float>(), dw.data_ptr<scalar_t>(),
                    dw_raw.numel());
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dw};
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Chunked linear CE sum and projected gradients");
    m.def("linear_backward", &linear_backward, "Scale projected linear CE gradients");
}
