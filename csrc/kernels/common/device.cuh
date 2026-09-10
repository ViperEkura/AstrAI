// Cross-family device vocabulary — pure CUDA, no torch, so the pure
// kernel headers and the out-of-tree harnesses share the exact same device
// view. Two concerns live here: device GEOMETRY (DeviceFacts — the planner
// layers price recipes against) and device GENERATION (CUTLASS-style arch
// tags with feature gates — the compile-time selection surface for per-arch
// kernel specializations). Capability checks specific to a family (e.g. the
// fp8 MMA minimum SM) live with that family (quantize/common.h); torch-bound
// tensor validation lives at the binding call sites.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace astrai {

// ---------------------------------------------------------------------------
// Generation tags. A kernel template takes `typename Arch` and specializations
// key on the tag, so adding a generation means adding a specialization,
// never editing existing ones. Feature flags mirror the instruction-set
// boundaries:
//   sm_80  cp.async + mma.sync (base; sm_86 identical for our purposes)
//   sm_89  + native fp8 mma.sync (m16n8k32)
//   sm_90  + TMA, mbarrier, wgmma (Hopper)
//   sm_100 + tcgen05 (Blackwell); sm_120 inherits the sm_100 feature set
//           for the load/MMA vocabulary used here
// ---------------------------------------------------------------------------

struct ArchSm80 {
    static constexpr int kMajor = 8, kMinor = 0;
    static constexpr bool kHasFp8Mma = false;
    static constexpr bool kHasTma = false;
    static constexpr bool kHasMbarrier = false;
    static constexpr bool kHasWgmma = false;
};

struct ArchSm89 {
    static constexpr int kMajor = 8, kMinor = 9;
    static constexpr bool kHasFp8Mma = true;
    static constexpr bool kHasTma = false;
    static constexpr bool kHasMbarrier = false;
    static constexpr bool kHasWgmma = false;
};

struct ArchSm90 {
    static constexpr int kMajor = 9, kMinor = 0;
    static constexpr bool kHasFp8Mma = true;
    static constexpr bool kHasTma = true;
    static constexpr bool kHasMbarrier = true;
    static constexpr bool kHasWgmma = true;
};

struct ArchSm100 {
    static constexpr int kMajor = 10, kMinor = 0;
    static constexpr bool kHasFp8Mma = true;
    static constexpr bool kHasTma = true;
    static constexpr bool kHasMbarrier = true;
    static constexpr bool kHasWgmma = true;  // plus tcgen05, gated further out
};

// Host-side ladder: map a runtime (major, minor) to the nearest generation
// tag. The ladder is total: unknown future devices land on the newest known
// generation, pre-sm_80 devices are rejected by the bindings long before.
inline constexpr int arch_generation(int major, int minor) {
    if (major >= 10) return 100;
    if (major == 9) return 90;
    if (major == 8 && minor >= 9) return 89;
    return 80;
}

// Arch dispatch, CUTLASS ArchTag-style: run the generic lambda with the
// generation tag whose specialization should serve this device.
// `fn` must be callable as fn(ArchTag{}) for every tag and return a
// common type.
template <typename Fn>
decltype(auto) arch_dispatch(int major, int minor, Fn&& fn) {
    switch (arch_generation(major, minor)) {
        case 100: return fn(ArchSm100{});
        case 90: return fn(ArchSm90{});
        case 89: return fn(ArchSm89{});
        default: return fn(ArchSm80{});
    }
}

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
