#pragma once
/* GEMM device entry kernels: cp.async and TMA staging. */
#include <cuda_runtime.h>

#include <api/gemm_common.h>
#include <epilogue/writer.cuh>
#include <kernel/gemm/mainloop.cuh>
#include <memory/pipeline.cuh>
#include <policy.cuh>
#include <scheduler.cuh>

namespace astrai {
namespace gemm {

// The ONE quantized-GEMM orchestrator (cp.async staging).
template <typename Policy>
__global__ void __launch_bounds__(Policy::kCtaThreads, Policy::kMinCtas) gemm_kernel(GemmParams p) {
    using Mainloop = GemmCollectiveMainloop<Policy>;
    using Epilogue = GemmCollectiveEpilogue<Policy>;
    /*
     * Stages live in dynamic shared memory so deep pipelines (> 48KB static
     * limit) opt in via cudaFuncSetAttribute in the launcher.
     */
    extern __shared__ __align__(16) char gemm_smem[];

    /*
     * Batch slice (grid.z): broadcast operands carry a 0 stride, so the
     * same pointer serves every batch.
     */
    using ElemA = typename Mainloop::ElemA;
    using ElemB = typename Mainloop::ElemB;
    using OutT = typename Policy::OutT;
    const ElemA* a =
        reinterpret_cast<const ElemA*>(p.a_ptr) + (int64_t)blockIdx.z * p.a_batch_stride;
    const ElemB* b =
        reinterpret_cast<const ElemB*>(p.b_ptr) + (int64_t)blockIdx.z * p.b_batch_stride;
    auto* out = reinterpret_cast<OutT*>(p.out_ptr) + (int64_t)blockIdx.z * p.out_batch_stride;

    static_assert(Mainloop::kOutputReclaimsRings,
                  "output tile must fit the reclaimed operand smem");
    const int2 blk = GemmTileScheduler::tile(blockIdx, gridDim, p.raster);
    Mainloop mainloop(gemm_smem, a, b, p.m, p.n, p.k, p.a_ld, p.b_ld, threadIdx.x, blk);
    typename Mainloop::AccTensor acc = {}; // C cells on the (mt, nt) grid
    mainloop.prologue();
    mainloop.accumulate(acc);
    /*
     * Drain the pipeline before the epilogue reclaims the operand rings:
     * cp_async_wait_all drains only the CALLING thread's copies and the
     * last mainloop iteration carries no trailing barrier — without this,
     * a thread racing into the epilogue scatters the output tile over
     * peers' still-in-flight staging writes. One barrier closes both.
     */
    astrai::PipelineSync<Mainloop::kStages>{}.drain();
    Epilogue(gemm_smem, p, blk.x, blk.y, threadIdx.x).run(acc, out);
}

/*
 * TMA orchestrator (sm_90+, dual-congruous staging): identical rings,
 * layouts and epilogue; the staging discipline changes — one elected
 * thread arms a per-slot mbarrier and issues the operand boxes
 * (cp.async.bulk.tensor), consumers wait the slot's phase. The rings sit
 * on a 1024B-aligned base because TMA swizzles the ABSOLUTE smem address
 * (the pad is budgeted in Policy::kSmemBytes), and the mbarriers live
 * right past the B ring.
 */
template <typename Policy, bool kRank3A, bool kRank3B>
__global__ void __launch_bounds__(Policy::kCtaThreads, Policy::kMinCtas)
    gemm_kernel_tma(GemmParams p,
                    const __grid_constant__ CUtensorMap tma_a,
                    const __grid_constant__ CUtensorMap tma_b) {
    using Traits = typename Policy::Traits;
    using Mainloop = GemmCollectiveMainloop<Policy>;
    using Epilogue = GemmCollectiveEpilogue<Policy>;
    static_assert(!Mainloop::kDirectA && !Mainloop::kDirectB,
                  "TMA staging requires dual-congruous operands");
    extern __shared__ __align__(16) char gemm_smem[];
    /*
     * Round the ring base up to its 1024B pattern period. Two's-complement
     * form: already-aligned bases pad 0 (~p would pad 1023 and misalign).
     */
    char* smem = gemm_smem + ((-reinterpret_cast<uintptr_t>(gemm_smem)) & 1023u);

    using OutT = typename Policy::OutT;
    auto* out = reinterpret_cast<OutT*>(p.out_ptr) + (int64_t)blockIdx.z * p.out_batch_stride;

    static_assert(Mainloop::kOutputReclaimsRings,
                  "output tile must fit the reclaimed operand smem");

    GemmTmaContext<kRank3A, kRank3B> tma;
    tma.map_a = &tma_a;
    tma.map_b = &tma_b;
    tma.bars = reinterpret_cast<uint64_t*>(smem + Mainloop::RingA::Layout::kTotalBytes +
                                           Mainloop::RingB::Layout::kTotalBytes);
    tma.depth = Mainloop::kARing;
    tma.z = blockIdx.z;
    if (threadIdx.x == 0) {
        for (int s = 0; s < Mainloop::kARing; ++s) {
            astrai::mbarrier_init(tma.full(s), 1); // producer expect_tx
            astrai::mbarrier_init(tma.empty(s), Policy::kCtaThreads);
        }
    }
    __syncthreads();

    const int2 blk = GemmTileScheduler::tile(blockIdx, gridDim, p.raster);
    Mainloop mainloop(smem, static_cast<const typename Mainloop::ElemA*>(p.a_ptr),
                      static_cast<const typename Mainloop::ElemB*>(p.b_ptr), p.m, p.n, p.k, p.a_ld,
                      p.b_ld, threadIdx.x, blk);
    typename Mainloop::AccTensor acc = {};
    mainloop.prologue(tma);
    mainloop.accumulate(acc, tma);
    /*
     * No cp.async groups on this path; the CTA join alone releases the
     * rings for the epilogue's reclaim.
     */
    __syncthreads();
    Epilogue(smem, p, blk.x, blk.y, threadIdx.x).run(acc, out);
}

} // namespace gemm
} // namespace astrai
