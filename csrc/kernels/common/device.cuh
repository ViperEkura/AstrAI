// Cross-family device vocabulary — pure CUDA, no torch, so the pure
// kernel headers and the out-of-tree harnesses share the exact same device
// view. Device GEOMETRY (DeviceFacts — the planner layers price recipes
// against) is the whole of it. Capability checks specific to a family (e.g.
// the fp8 MMA minimum SM) live with that family (quantize/common.h);
// torch-bound tensor validation lives at the binding call sites.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace astrai {

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

// Device geometry the planner layers price recipes against (queried once
// per device and cached; benign init race — every writer stores the same
// facts). smem_max is the PER-BLOCK opt-in ceiling — what
// cudaFuncSetAttribute can raise dynamic shared memory to (about 1KB
// under the per-SM figure), the feasibility bound for staged kernels;
// consumers that fold smem residency into measured throughput scalars
// simply do not read it. cc is the numeric compute capability (120 =
// sm_120), the feature gate for the TMA staging path.
struct DeviceFacts {
    int sms;
    int smem_max;
    int64_t l2_bytes;
    int cc = 0;
};

inline DeviceFacts device_facts() {
    static DeviceFacts cached[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    const bool cacheable = dev >= 0 && dev < 64;
    DeviceFacts facts = cacheable ? cached[dev] : DeviceFacts{};
    if (!facts.sms) {
        int l2 = 0, major = 0, minor = 0;
        cudaDeviceGetAttribute(&facts.sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&facts.smem_max,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        facts.sms = facts.sms > 0 ? facts.sms : 1;
        facts.smem_max = facts.smem_max > 0 ? facts.smem_max : 48 * 1024;
        facts.l2_bytes = l2 > 0 ? l2 : (int64_t{4} << 20);
        facts.cc = major > 0 ? major * 10 + minor : 0;
        if (cacheable) cached[dev] = facts;
    }
    return facts;
}

}  // namespace astrai
