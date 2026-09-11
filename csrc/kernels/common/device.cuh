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
//
// smem_per_sm / regs_per_sm are the per-SM RESOURCE figures a plan's
// residency is the minimum of (plan_table.h prices rows against them).
// Queried rather than written down because they move with the SM generation,
// not just the SKU — smem per SM went 100KB (Ada) to 228KB (Hopper/Blackwell
// datacenter) while the register file stayed 64K — so a wave bound derived
// from one part's figures is wrong on the other. On the 512-thread tiles this
// repo instantiates it is the REGISTER FILE that binds, at two CTAs on every
// 64K part: 64 regs x 512 threads x 2 = 65536 exactly, so the accumulator
// alone leaves no room for a third. threads-per-SM is deliberately not a
// field: at 1536 (Ada) or 2048 (sm_90+) threads per SM the thread ceiling for
// a 512-thread tile (3 or 4) is never under the register floor, so the term
// could not bind and would only be a fourth thing to keep in step.
struct DeviceFacts {
    int sms;
    int smem_max;
    int smem_per_sm;
    int regs_per_sm;
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
        cudaDeviceGetAttribute(&facts.smem_per_sm,
                               cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
        cudaDeviceGetAttribute(&facts.regs_per_sm,
                               cudaDevAttrMaxRegistersPerMultiprocessor, dev);
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        facts.sms = facts.sms > 0 ? facts.sms : 1;
        facts.smem_max = facts.smem_max > 0 ? facts.smem_max : 48 * 1024;
        facts.smem_per_sm = facts.smem_per_sm > 0 ? facts.smem_per_sm : facts.smem_max;
        facts.regs_per_sm = facts.regs_per_sm > 0 ? facts.regs_per_sm : 65536;
        facts.l2_bytes = l2 > 0 ? l2 : (int64_t{4} << 20);
        facts.cc = major > 0 ? major * 10 + minor : 0;
        if (cacheable) cached[dev] = facts;
    }
    return facts;
}

}  // namespace astrai
