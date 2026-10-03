#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <tuple>
#include <vector>

namespace astrai {
namespace muon_ns {

namespace {

torch::Tensor muon_ns_impl(torch::Tensor grad,
                           const std::vector<double>& coefficients,
                           int64_t ns_steps,
                           double eps) {
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

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void prepare_kernel(const scalar_t* grad,
                               scalar_t* momentum_buffer,
                               at::BFloat16* update,
                               float* partial_squares,
                               int64_t count,
                               float momentum,
                               bool nesterov) {
    __shared__ float sums[kThreads];
    float local_sum = 0.0f;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float g = static_cast<float>(grad[index]);
        const float old_buffer = static_cast<float>(momentum_buffer[index]);
        const scalar_t new_buffer =
            static_cast<scalar_t>(old_buffer + (g - old_buffer) * (1.0f - momentum));
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
normalize_kernel(at::BFloat16* update, const float* norm_squared, int64_t count, float eps) {
    const at::BFloat16 rounded_norm = static_cast<at::BFloat16>(sqrtf(norm_squared[0]));
    const float divisor = fmaxf(static_cast<float>(rounded_norm), eps);
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        update[index] = static_cast<at::BFloat16>(static_cast<float>(update[index]) / divisor);
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
            static_cast<float>(momentum), nesterov);
    } else {
        prepare_kernel<<<blocks, kThreads, 0, stream>>>(
            grad.data_ptr<float>(), momentum_buffer.data_ptr<float>(),
            update.data_ptr<at::BFloat16>(), partials.data_ptr<float>(), grad.numel(),
            static_cast<float>(momentum), nesterov);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {update, partials};
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
    m.def("normalize_", &normalize_, "Normalize a local Muon BF16 shard");
    m.def("finish_", &finish_, "Fuse local Muon decay and parameter update");
    m.def("muon_ns", &astrai::muon_ns::muon_ns_impl, py::arg("grad"), py::arg("ns_coefficients"),
          py::arg("ns_steps"), py::arg("eps"),
          "Run Muon's Newton-Schulz orthogonalization in one extension call");
}
