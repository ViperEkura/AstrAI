// Async data-movement vocabulary: the raw cp.async / mbarrier PTX sites
// plus the stage-pipeline abstractions built on them. Two generations of
// backing primitive, one producer/consumer surface (selected with
// `if constexpr (Arch::kHasMbarrier)`):
//   PipelineSync<Stages>      sm_80/89: cp.async wait_group + __syncthreads
//   PipelineMbarrier<Stages>  sm_90+:   per-stage mbarrier with expect_tx
//                             (CUTLASS PipelineTmaAsync semantics)
// The mbarrier body compiles only on sm_90+; earlier passes leave the class
// inert so the template can still name it. Phase protocol: stage s uses
// barrier s % Stages; the consumer tracks a per-barrier parity bit that
// flips when the barrier's arrival count trips; producers arrive_expect_tx
// before issuing the stage's copies; consumers wait_parity then read.

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace astrai {

// ---------------------------------------------------------------------------
// Raw cp.async primitives
// ---------------------------------------------------------------------------

// Raw emitter: read src_size bytes (<= 16) from gmem into the shared
// offset. src_size = 0 reads nothing, so a predicated-off call zero-fills
// its destination without touching the (possibly out-of-range) source.
// BypassL1 selects .cg (L2 only, default) vs .ca (L1 + L2).
template <bool BypassL1 = true>
__device__ __forceinline__ void cp_async_16_raw(unsigned smem_addr,
                                                const void* gmem_ptr,
                                                int src_size) {
    if constexpr (BypassL1) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                     :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_size));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                     :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_size));
    }
}

// Unconditional 16-byte copy to a generic shared pointer.
// `T` is the smem element type; only the destination pointer's type matters.
template <typename T, bool BypassL1 = true>
__device__ __forceinline__ void cp_async_16(T* smem_ptr,
                                            const void* gmem_ptr) {
    cp_async_16_raw<BypassL1>(__cvta_generic_to_shared(smem_ptr), gmem_ptr,
                              16);
}

// Predicated: full copy when `pred`, zero-fill otherwise.
template <typename T, bool BypassL1 = true>
__device__ __forceinline__ void cp_async_16(T* smem_ptr, const void* gmem_ptr,
                                            bool pred) {
    cp_async_16_raw<BypassL1>(__cvta_generic_to_shared(smem_ptr), gmem_ptr,
                              pred ? 16 : 0);
}

// Partial prefix: copy `src_bytes` of the 16B chunk, hardware zero-fill the
// rest — the k-tail / OOB-row predication form (CUTLASS's zfill iterators):
// boundary chunks ride the same LDGSTS instead of a scalar fallback loop.
template <typename T, bool BypassL1 = true>
__device__ __forceinline__ void cp_async_16(T* smem_ptr, const void* gmem_ptr,
                                            int src_bytes) {
    cp_async_16_raw<BypassL1>(__cvta_generic_to_shared(smem_ptr), gmem_ptr,
                              src_bytes);
}

// Predicated raw-offset form: the destination is an already-converted
// shared-memory offset (e.g. a loop-carried swizzled stage address), so
// steady-state prefetch sites issue one LDGSTS straight from the register.
template <bool BypassL1 = true>
__device__ __forceinline__ void cp_async_16(unsigned smem_addr,
                                            const void* gmem_ptr, bool pred) {
    cp_async_16_raw<BypassL1>(smem_addr, gmem_ptr, pred ? 16 : 0);
}

// Commit all outstanding cp.async ops of this thread as one group.
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

// Wait for every committed group (pipeline drain).
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

// Wait until at most KeepGroups committed groups are still in flight.
// PTX requires an immediate operand; keep it as a template argument so the
// stage policy stays compile-time configurable.
template <int KeepGroups>
__device__ __forceinline__ void cp_async_wait_group() {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7,
                  "cp.async.wait_group supports immediates in [0, 7]");
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
}

// ---------------------------------------------------------------------------
// Stage pipelines
// ---------------------------------------------------------------------------

// sm_80/89 ring: the family's existing discipline — producers issue the
// stage's cp.async chunks and commit one group per stage; a consumer that
// needs tile i waits until at most Stages-1 groups remain in flight, then
// __syncthreads to publish the stage slot across the CTA.
template <int Stages>
struct PipelineSync {
    static_assert(Stages >= 1, "a pipeline needs at least one stage");

    __device__ __forceinline__ void producer_commit() const {
        cp_async_commit_group();
    }

    // Block until stage slots up to (group_of(tile) - (Stages-1)) landed.
    __device__ __forceinline__ void consumer_wait() const {
        cp_async_wait_group<Stages - 1>();
        __syncthreads();
    }

    // Full drain, epilogue-side.
    __device__ __forceinline__ void drain() const {
        cp_async_wait_all();
        __syncthreads();
    }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define ASTRAI_MBAR_ENABLED 1
#else
#define ASTRAI_MBAR_ENABLED 0
#endif

// mbarrier PTX sites (sm_90+). Shared addresses travel as 32-bit smem
// offsets per the PTX spec. Declarations stay visible on every pass
// (only the asm bodies are guarded) so __global__ templates can name
// them; the stubs must never execute pre-sm_90.
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
#if ASTRAI_MBAR_ENABLED
    const uint32_t addr = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(addr), "r"(count));
#else
    (void)bar;
    (void)count;
#endif
}

// Plain arrival (no transaction expectation): the TMA pipeline's
// consumer-side release — every thread arrives on the stage's empty
// barrier after its last fragment read, and the producer waits that
// barrier's phase before overwriting the slot.
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
#if ASTRAI_MBAR_ENABLED
    const uint32_t addr = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr));
#else
    (void)bar;
#endif
}

// Arrive with a transaction-count expectation: the barrier trips only
// after `bytes` of async copies (TMA) have landed in addition to the
// arrival itself. The TMA-issuing producer thread calls this once per
// stage; expect_tx accumulates, so per-operand barriers can be fed
// separately.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar,
                                                          uint32_t bytes) {
#if ASTRAI_MBAR_ENABLED
    const uint32_t addr = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 ::"r"(addr), "r"(bytes));
#else
    (void)bar;
    (void)bytes;
#endif
}

// Phase flip wait: blocks while the barrier's phase bit still equals
// `parity` (0 on first use of the barrier). Returns once the phase has
// advanced past it, i.e. the awaited trip completed.
__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar,
                                                     uint32_t parity) {
#if ASTRAI_MBAR_ENABLED
    const uint32_t addr = __cvta_generic_to_shared(bar);
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}"
        ::"r"(addr), "r"(parity));
#else
    (void)bar;
    (void)parity;
#endif
}

// sm_90+ ring: one mbarrier per stage slot. Arrive count 1 (the single
// TMA-issuing producer thread arrives with expect_tx; CTA-wide consumers
// only wait — the compute-side release for stage reuse is the consumer's
// own arrive, kept out of this minimal surface until warp specialization
// lands, at which point it mirrors PipelineTmaAsync's full handshake).
template <int Stages>
struct PipelineMbarrier {
    static_assert(Stages >= 1, "a pipeline needs at least one stage");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint64_t barriers[Stages];

    __device__ __forceinline__ void init(uint32_t producer_count = 1) {
        if (threadIdx.x == 0) {
            for (int s = 0; s < Stages; ++s)
                mbarrier_init(&barriers[s], producer_count);
        }
        __syncthreads();
    }

    // Producer, once per stage: arm the byte count, then issue the TMA
    // copies targeting this stage's slot.
    __device__ __forceinline__ void producer_commit(int stage, uint32_t bytes) {
        mbarrier_arrive_expect_tx(&barriers[stage % Stages], bytes);
    }

    // Consumer: wait for trip `use` of stage slot (use = tile / Stages);
    // parity alternates each reuse.
    __device__ __forceinline__ void consumer_wait(int stage, int use) {
        mbarrier_wait_parity(&barriers[stage % Stages],
                             static_cast<uint32_t>(use & 1));
    }
#endif
};

}  // namespace astrai
