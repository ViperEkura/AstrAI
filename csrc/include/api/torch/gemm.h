#pragma once
/* Tensor/ScalarType entry points; native planning remains Torch-free. */
#include <c10/util/Optional.h>
#include <torch/types.h>

#include <api/native/gemm_plan.h>

namespace astrai {
namespace gemm {

/*
 * The single quantized-GEMM entry (one kernel per cell); semantics, validation
 * and dispatch at the definition in gemm/entry.cu. pybind's quant_gemm is a thin
 * None-tolerant wrapper.
 */
torch::Tensor quant_gemm_impl(torch::Tensor a,
                              torch::Tensor b,
                              c10::optional<torch::Tensor> a_scale,
                              c10::optional<torch::Tensor> b_scale,
                              bool trans_a,
                              bool trans_b,
                              c10::optional<torch::Tensor> bias);

torch::Tensor small_m_linear(const torch::Tensor& x, const torch::Tensor& weight);

PlanProbe plan_probe(int64_t m,
                     int64_t n,
                     int64_t k,
                     at::ScalarType dt_a,
                     at::ScalarType dt_b,
                     bool trans_a,
                     bool trans_b,
                     int64_t batch);

} // namespace gemm
} // namespace astrai
