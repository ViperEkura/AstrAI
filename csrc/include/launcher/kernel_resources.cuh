#pragma once
/* Cached occupancy and compiler resources for a typed kernel and device. */
#include <policy/gemm/config.h>
#include <utils/launch.cuh>

#include <map>

namespace astrai {
namespace gemm {

// Function attributes belong to the CUDA device. Each host thread can safely
// initialize its device on first use; repeated writes set the same kernel limit.
template <auto Kernel> inline void configure_kernel_smem(int bytes, int device) {
    if (bytes <= 48 * 1024)
        return;
    static thread_local std::map<int, int> configured;
    int& limit = configured[device];
    if (limit < bytes) {
        ASTRAI_CUDA_CHECK(
            cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        limit = bytes;
    }
}

// Kernel metadata is queried once per typed kernel and device. No launch or
// timing is involved; CUDA accounts for registers and allocation granularity.
template <auto Kernel, typename Policy> KernelResources kernel_resources(const PlanQuery& q) {
    using T = typename Policy::Traits;
    static thread_local std::map<int, KernelResources> cache;
    const auto found = cache.find(q.dev.ordinal);
    if (found != cache.end())
        return found->second;
    KernelResources result{};
    if (Policy::kSmemBytes > q.dev.smem_max)
        return result;
    configure_kernel_smem<Kernel>(Policy::kSmemBytes, q.dev.ordinal);
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
