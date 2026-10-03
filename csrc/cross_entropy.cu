#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <tuple>

namespace {
constexpr int kThreads = 256;

template <bool Maximum> __device__ float block_reduce(float value, float* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other = shared[threadIdx.x + stride];
            if constexpr (Maximum) {
                shared[threadIdx.x] = (isnan(shared[threadIdx.x]) || isnan(other))
                                          ? NAN
                                          : fmaxf(shared[threadIdx.x], other);
            } else {
                shared[threadIdx.x] += other;
            }
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename scalar_t>
__global__ void forward_rows(const scalar_t* logits,
                             const int64_t* targets,
                             float* maxima,
                             float* log_sums,
                             float* losses,
                             int64_t vocab,
                             int64_t ignore_index) {
    const int64_t row = blockIdx.x;
    const int64_t target = targets[row];
    CUDA_KERNEL_ASSERT(target == ignore_index || (target >= 0 && target < vocab));
    if (target == ignore_index) {
        if (threadIdx.x == 0) {
            losses[row] = 0.0f;
            maxima[row] = 0.0f;
            log_sums[row] = 0.0f;
        }
        return;
    }
    __shared__ float shared[kThreads];
    float maximum = -INFINITY;
    for (int64_t col = threadIdx.x; col < vocab; col += kThreads) {
        const float value = static_cast<float>(logits[row * vocab + col]);
        maximum = (isnan(maximum) || isnan(value)) ? NAN : fmaxf(maximum, value);
    }
    maximum = block_reduce<true>(maximum, shared);
    __syncthreads();
    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < vocab; col += kThreads) {
        sum += expf(static_cast<float>(logits[row * vocab + col]) - maximum);
    }
    sum = block_reduce<false>(sum, shared);
    if (threadIdx.x == 0) {
        const float log_sum = logf(sum);
        maxima[row] = maximum;
        log_sums[row] = log_sum;
        losses[row] = (maximum - static_cast<float>(logits[row * vocab + target])) + log_sum;
    }
}

__global__ void reduce_loss(const float* losses,
                            const int64_t* targets,
                            float* loss,
                            int64_t* valid_count,
                            int64_t rows,
                            int64_t ignore_index,
                            bool mean) {
    __shared__ float shared[kThreads];
    __shared__ int64_t counts[kThreads];
    float total = 0.0f;
    int64_t count = 0;
    for (int64_t row = threadIdx.x; row < rows; row += kThreads) {
        total += losses[row];
        count += targets[row] != ignore_index;
    }
    total = block_reduce<false>(total, shared);
    counts[threadIdx.x] = count;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        valid_count[0] = counts[0];
        loss[0] = mean ? total / static_cast<float>(counts[0]) : total;
    }
}

template <typename scalar_t>
__global__ void backward_rows(const scalar_t* logits,
                              const int64_t* targets,
                              const float* maxima,
                              const float* log_sums,
                              const int64_t* valid_count,
                              const float* grad_loss,
                              scalar_t* grad_logits,
                              int64_t vocab,
                              int64_t ignore_index,
                              bool mean) {
    const int64_t row = blockIdx.x;
    const int64_t target = targets[row];
    const bool ignored = target == ignore_index;
    const float scale =
        ignored ? 0.0f : (mean ? grad_loss[0] / static_cast<float>(valid_count[0]) : grad_loss[0]);
    for (int64_t col = threadIdx.x; col < vocab; col += kThreads) {
        float grad = 0.0f;
        if (!ignored) {
            const float log_prob =
                (static_cast<float>(logits[row * vocab + col]) - maxima[row]) - log_sums[row];
            grad = __fmul_rn(expf(log_prob), scale);
            if (col == target) {
                grad = __fsub_rn(grad, scale);
            }
        }
        grad_logits[row * vocab + col] = static_cast<scalar_t>(grad);
    }
}

void check_inputs(const torch::Tensor& logits, const torch::Tensor& targets) {
    TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.is_contiguous(),
                "logits must be a contiguous CUDA matrix");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16 || logits.scalar_type() == at::kHalf ||
                    logits.scalar_type() == at::kFloat,
                "logits must be BF16, FP16, or FP32");
    TORCH_CHECK(logits.size(0) > 0 && logits.size(1) > 0, "logits must be nonempty");
    TORCH_CHECK(targets.device() == logits.device() && targets.dim() == 1 &&
                    targets.is_contiguous() && targets.scalar_type() == at::kLong &&
                    targets.numel() == logits.size(0),
                "targets must be a matching contiguous CUDA int64 vector");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
    const torch::Tensor& logits, const torch::Tensor& targets, int64_t ignore_index, bool mean) {
    check_inputs(logits, targets);
    const at::cuda::OptionalCUDAGuard guard(device_of(logits));
    const int64_t rows = logits.size(0);
    auto options = logits.options().dtype(at::kFloat);
    auto maxima = torch::empty({rows}, options);
    auto log_sums = torch::empty({rows}, options);
    auto losses = torch::empty({rows}, options);
    auto loss = torch::empty({}, options);
    auto count = torch::empty({}, logits.options().dtype(at::kLong));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(), "cross_entropy_forward", [&] {
            forward_rows<<<rows, kThreads, 0, stream>>>(
                logits.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(), maxima.data_ptr<float>(),
                log_sums.data_ptr<float>(), losses.data_ptr<float>(), logits.size(1), ignore_index);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    reduce_loss<<<1, kThreads, 0, stream>>>(losses.data_ptr<float>(), targets.data_ptr<int64_t>(),
                                            loss.data_ptr<float>(), count.data_ptr<int64_t>(), rows,
                                            ignore_index, mean);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {loss, maxima, log_sums, count};
}

torch::Tensor backward(const torch::Tensor& logits,
                       const torch::Tensor& targets,
                       const torch::Tensor& maxima,
                       const torch::Tensor& log_sums,
                       const torch::Tensor& count,
                       const torch::Tensor& grad_loss,
                       int64_t ignore_index,
                       bool mean) {
    check_inputs(logits, targets);
    for (const auto& tensor : {maxima, log_sums}) {
        TORCH_CHECK(tensor.device() == logits.device() && tensor.scalar_type() == at::kFloat &&
                        tensor.is_contiguous() && tensor.numel() == logits.size(0),
                    "invalid saved row statistics");
    }
    TORCH_CHECK(count.device() == logits.device() && count.scalar_type() == at::kLong &&
                    count.numel() == 1,
                "invalid valid_count");
    TORCH_CHECK(grad_loss.device() == logits.device() && grad_loss.scalar_type() == at::kFloat &&
                    grad_loss.numel() == 1,
                "grad_loss must be a CUDA FP32 scalar");
    const at::cuda::OptionalCUDAGuard guard(device_of(logits));
    auto grad_logits = torch::empty_like(logits);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(), "cross_entropy_backward", [&] {
            backward_rows<<<logits.size(0), kThreads, 0, stream>>>(
                logits.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(), maxima.data_ptr<float>(),
                log_sums.data_ptr<float>(), count.data_ptr<int64_t>(), grad_loss.data_ptr<float>(),
                grad_logits.data_ptr<scalar_t>(), logits.size(1), ignore_index, mean);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_logits;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "FP32-reduced cross entropy from low-precision logits");
    m.def("backward", &backward, "Cross entropy backward without FP32 logits storage");
}
