/* Pure-CUDA device facts shared by kernels and out-of-tree harnesses.
 * Family capability checks and torch validation stay with their callers.
 */

#pragma once

#include <cuda_runtime.h>

#include <api/native/device.h>

namespace astrai {

// Geometry

/*
 * Cached device geometry used by the planner. smem_max is the per-block
 * opt-in limit; smem_per_sm and regs_per_sm bound CTA residency. cc gates TMA.
 * Query these values because resource limits vary by architecture.
 */
inline DeviceFacts device_facts() {
    static thread_local DeviceFacts cached[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    const bool cacheable = dev >= 0 && dev < 64;
    DeviceFacts facts = cacheable ? cached[dev] : DeviceFacts{};
    if (!facts.sms) {
        int l2 = 0, major = 0, minor = 0;
        facts.ordinal = dev;
        cudaDeviceGetAttribute(&facts.threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        cudaDeviceGetAttribute(&facts.sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&facts.smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cudaDeviceGetAttribute(&facts.smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                               dev);
        cudaDeviceGetAttribute(&facts.regs_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, dev);
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        facts.sms = facts.sms > 0 ? facts.sms : 1;
        facts.smem_max = facts.smem_max > 0 ? facts.smem_max : 48 * 1024;
        facts.smem_per_sm = facts.smem_per_sm > 0 ? facts.smem_per_sm : facts.smem_max;
        facts.regs_per_sm = facts.regs_per_sm > 0 ? facts.regs_per_sm : 65536;
        facts.l2_bytes = l2 > 0 ? l2 : (int64_t{4} << 20);
        facts.cc = major > 0 ? major * 10 + minor : 0;
        if (cacheable)
            cached[dev] = facts;
    }
    return facts;
}

} // namespace astrai
