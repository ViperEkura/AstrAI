#pragma once
/* Cached occupancy and compiler resources for a typed kernel and device. */
#include <launcher/plan_types.h>
#include <utils/launch.cuh>

#include <map>

namespace astrai {
namespace gemm {

// Kernel metadata is queried once per typed kernel and device. No launch or
// timing is involved; CUDA accounts for registers and allocation granularity.
template <auto Kernel, typename Policy>
KernelResources kernel_resources(const PlanQuery& q) {
    using T = typename Policy::Traits;
    static thread_local std::map<int, KernelResources> cache;
    const auto found = cache.find(q.dev.ordinal);
    if (found != cache.end())
        return found->second;
    KernelResources result{};
    if (Policy::kSmemBytes > q.dev.smem_max)
        return result;
    if (Policy::kSmemBytes > 48 * 1024)
        ASTRAI_CUDA_CHECK(cudaFuncSetAttribute(
            Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Policy::kSmemBytes));
    cudaFuncAttributes attributes{};
    ASTRAI_CUDA_CHECK(cudaFuncGetAttributes(&attributes, Kernel));
    ASTRAI_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &result.resident, Kernel, T::kCtaThreads, Policy::kSmemBytes));
    result.registers = attributes.numRegs;
    result.local_bytes = static_cast<int>(attributes.localSizeBytes);
    cache.emplace(q.dev.ordinal, result);
    return result;
}

} // namespace gemm
} // namespace astrai
