/*
 * Async data-movement vocabulary: the raw cp.async / mbarrier PTX sites
 * plus the stage-pipeline abstraction built on cp.async.
 *   PipelineSync<Stages>      sm_80/89: cp.async wait_group + __syncthreads
 * The mbarrier sites (init / arrive_expect_tx / wait_parity) are the shared
 * vocabulary the gemm TMA path builds its per-stage ring from. Phase
 * protocol: stage s uses barrier s % Stages; the consumer tracks a
 * per-barrier parity bit that flips when the barrier's arrival count trips;
 * producers arrive_expect_tx before issuing the stage's copies; consumers
 * wait_parity then read.
 */

#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <utils/define.cuh>

namespace astrai {

// Copy the valid prefix into shared memory and zero-fill the rest of the vector.
// PTX permits .cg only for 16-byte copies. Use .ca for 8-byte copies.
template <int Bytes>
DEVICE_FORCEINLINE void cp_async(unsigned smem_addr, const void* src, int valid_bytes = Bytes) {
    static_assert(Bytes == 8 || Bytes == 16, "expected an 8- or 16-byte copy");
    if constexpr (Bytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_addr), "l"(src),
                     "r"(valid_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(smem_addr), "l"(src),
                     "r"(valid_bytes));
    }
}

template <int Bytes, typename T>
DEVICE_FORCEINLINE void cp_async(T* dst, const void* src, int valid_bytes = Bytes) {
    cp_async<Bytes>(static_cast<unsigned>(__cvta_generic_to_shared(dst)), src, valid_bytes);
}

// A false predicate zero-fills the entire vector.
template <int Bytes, typename T>
DEVICE_FORCEINLINE void cp_async(T* dst, const void* src, bool valid) {
    cp_async<Bytes>(static_cast<unsigned>(__cvta_generic_to_shared(dst)), src, valid ? Bytes : 0);
}

// Commit all outstanding cp.async ops of this thread as one group.
DEVICE_FORCEINLINE void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); }

// Wait for every committed group (pipeline drain).
DEVICE_FORCEINLINE void cp_async_wait_all() { asm volatile("cp.async.wait_all;"); }

/*
 * Wait until at most KeepGroups committed groups are still in flight.
 * PTX requires an immediate operand; keep it as a template argument so the
 * stage policy stays compile-time configurable.
 */
template <int KeepGroups> DEVICE_FORCEINLINE void cp_async_wait_group() {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7,
                  "cp.async.wait_group supports immediates in [0, 7]");
    asm volatile("cp.async.wait_group %0;" ::"n"(KeepGroups));
}

/*
 * Stage pipelines
 */

/*
 * sm_80/89 ring: the family's existing discipline — producers issue the
 * stage's cp.async chunks and commit one group per stage; a consumer that
 * needs tile i waits until at most Stages-1 groups remain in flight, then
 * __syncthreads to publish the stage slot across the CTA.
 */
template <int Stages> struct PipelineSync {
    static_assert(Stages >= 1, "a pipeline needs at least one stage");

    DEVICE_FORCEINLINE void producer_commit() const { cp_async_commit_group(); }

    // Block until stage slots up to (group_of(tile) - (Stages-1)) landed.
    DEVICE_FORCEINLINE void consumer_wait() const {
        cp_async_wait_group<Stages - 1>();
        __syncthreads();
    }

    // Full drain, epilogue-side.
    DEVICE_FORCEINLINE void drain() const {
        cp_async_wait_all();
        __syncthreads();
    }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define ASTRAI_MBAR_ENABLED 1
#else
#define ASTRAI_MBAR_ENABLED 0
#endif

/*
 * mbarrier PTX sites (sm_90+). Shared addresses travel as 32-bit smem
 * offsets per the PTX spec. Declarations stay visible on every pass
 * (only the asm bodies are guarded) so __global__ templates can name
 * them; the stubs must never execute pre-sm_90.
 */
DEVICE_FORCEINLINE void mbarrier_init(uint64_t* bar, uint32_t count) {
#if ASTRAI_MBAR_ENABLED
    const uint32_t addr = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(addr), "r"(count));
#else
    (void)bar;
    (void)count;
#endif
}

/*
 * Plain arrival (no transaction expectation): the TMA pipeline's
 * consumer-side release — every thread arrives on the stage's empty
 * barrier after its last fragment read, and the producer waits that
 * barrier's phase before overwriting the slot.
 */
DEVICE_FORCEINLINE void mbarrier_arrive(uint64_t* bar) {
#if ASTRAI_MBAR_ENABLED
    const uint32_t addr = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr));
#else
    (void)bar;
#endif
}

/*
 * Arrive with a transaction-count expectation: the barrier trips only
 * after `bytes` of async copies (TMA) have landed in addition to the
 * arrival itself. The TMA-issuing producer thread calls this once per
 * stage; expect_tx accumulates, so per-operand barriers can be fed
 * separately.
 */
DEVICE_FORCEINLINE void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
#if ASTRAI_MBAR_ENABLED
    const uint32_t addr = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr), "r"(bytes));
#else
    (void)bar;
    (void)bytes;
#endif
}

/*
 * Phase flip wait: blocks while the barrier's phase bit still equals
 * `parity` (0 on first use of the barrier). Returns once the phase has
 * advanced past it, i.e. the awaited trip completed.
 */
DEVICE_FORCEINLINE void mbarrier_wait_parity(uint64_t* bar, uint32_t parity) {
#if ASTRAI_MBAR_ENABLED
    const uint32_t addr = __cvta_generic_to_shared(bar);
    asm volatile("{\n"
                 ".reg .pred P;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
                 "@P bra DONE;\n"
                 "bra LAB_WAIT;\n"
                 "DONE:\n"
                 "}" ::"r"(addr),
                 "r"(parity));
#else
    (void)bar;
    (void)parity;
#endif
}

} // namespace astrai
