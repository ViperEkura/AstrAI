#pragma once
/* Direct 8-bit crosswise staging and its two-phase carry. */
#include <utils/define.cuh>
#include <utils/tensor.cuh>

namespace astrai {
namespace gemm {

/*
 * The two arms one 16-row crosswise chunk can take, shared by the general
 * loader and the register carry — the pair must never diverge on the perm
 * sequence or the predication, or the same operand would stage differently
 * depending on bus width. Fast arm: four contract runs (uint4, already in
 * the register file) -> one PRMT pass. Slow arm (row tail / misaligned
 * base): element-granular gather, contract-tail columns zero-fill.
 */
template <typename TileT>
DEVICE_FORCEINLINE void crosswise_perm_span(TileT tile, int rg, int span, const uint4* v) {
    const unsigned* bytes = reinterpret_cast<const unsigned*>(v);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        /*
         * Word i = row r0+i's span: byte i of each of the four runs
         * [v0.b(i), v1.b(i), v2.b(i), v3.b(i)]; run s is one uint4 (16
         * bytes = 16 rows), so word i>>2 of run s is bytes[4*s + (i>>2)].
         */
        const unsigned nib = i & 3;
        const unsigned sel = nib | ((nib + 4) << 4);
        const unsigned w01 = __byte_perm(bytes[0 + (i >> 2)], bytes[4 + (i >> 2)], sel);
        const unsigned w23 = __byte_perm(bytes[8 + (i >> 2)], bytes[12 + (i >> 2)], sel);
        *reinterpret_cast<unsigned*>(tile(rg * 16 + i, span * 4)) = __byte_perm(w01, w23, 0x5410u);
    }
}

template <typename TileT, typename ElemT>
DEVICE_FORCEINLINE void crosswise_gather_span(TileT tile,
                                              const ElemT* __restrict__ operand,
                                              int64_t rows,
                                              int64_t contract,
                                              int64_t ld,
                                              int64_t k_base,
                                              int64_t r0,
                                              int rg,
                                              int span) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int col = span * 4 + s;
        if (k_base + col >= contract) {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                *tile(rg * 16 + i, col) = ElemT(0.0f);
            continue;
        }
#pragma unroll
        for (int i = 0; i < 16; ++i)
            *tile(rg * 16 + i, col) =
                r0 + i < rows ? operand[(k_base + col) * ld + r0 + i] : ElemT(0.0f);
    }
}

/*
 * Direct (synchronous) crosswise load into a canonical rotating stage:
 * LDG.128 runs (4 x 16B of the non-contract dim) + in-register transpose
 * (PRMT) + 16 STS.32 — crosswise operands cannot cp.async into the
 * canonical tile (a 16B global run holds contract positions for a run of
 * the other dim). One chunk = 64B staging one 16-row group (4 runs of 16
 * rows x 4 contract positions); the PRMT gathers one 32-bit word per row
 * across the four runs. The GENERAL grid-stride form: any geometry stages
 * correctly however few threads; the instantiated path's two-phase sibling
 * (CrosswiseCarry) uses this fallback for
 * the under-subscribed bus.
 */
template <typename SmemLayout, typename ElemT, int kThreads>
DEVICE_FORCEINLINE void load_crosswise_direct_general(Tensor<PtrEngine<ElemT>, SmemLayout> tile,
                                                      const ElemT* __restrict__ operand,
                                                      int64_t rows,
                                                      int64_t contract,
                                                      int64_t ld,
                                                      int tid,
                                                      int64_t k_base,
                                                      int64_t block_row) {
    static_assert(sizeof(ElemT) == 1, "crosswise LDG+PRMT staging requires 1-byte elements");
    constexpr int kRowsTile = SmemLayout::kRows;
    constexpr int kK = SmemLayout::kChunks * 16;
    constexpr int kCw = 4;           // contract elems per chunk
    constexpr int kSpans = kK / kCw; // contract spans per tile
    constexpr int kGroups = kRowsTile / 16;
    constexpr int kTChunks = kSpans * kGroups; // 64B chunks per tile
    /*
     * r0 is a multiple of 16 and p*ld preserves alignment whenever ld has
     * it, so every run of a chunk shares one alignment verdict.
     */
    const bool run_aligned =
        ((reinterpret_cast<uintptr_t>(operand) | (ld * (int64_t)sizeof(ElemT))) & 15) == 0;
    for (int chunk = tid; chunk < kTChunks; chunk += kThreads) {
        const int span = chunk / kGroups;
        const int rg = chunk % kGroups;
        const int64_t r0 = block_row + rg * 16;
        const bool rows_full = r0 + 15 < rows;
        if (rows_full && run_aligned) {
            const int64_t p0 = k_base + span * kCw;
            uint4 v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                /*
                 * Contract tail: a run past k carries zero bytes; they flow
                 * through the transpose like any other value. v[i] = 16 rows
                 * at contract p0+i.
                 */
                if (p0 + i < contract)
                    v[i] = __ldg(reinterpret_cast<const uint4*>(operand + (p0 + i) * ld + r0));
                else
                    v[i] = make_uint4(0u, 0u, 0u, 0u);
            }
            crosswise_perm_span(tile, rg, span, v);
        } else {
            crosswise_gather_span(tile, operand, rows, contract, ld, k_base, r0, rg, span);
        }
    }
}

/*
 * Two-phase register staging for the 8-bit crosswise operand: the general
 * loader's synchronous round trip sits on the MMA phase's critical path, so
 * the carry splits it across the phase boundary — issue() fires the global
 * runs one phase ahead, commit() does PRMT + STS after the MMA phase, the
 * latency hiding behind tensor-pipe work. One thread owns the whole 64B
 * chunk: the four contract runs must meet in one register file for the
 * byte-perm (also why the 1-byte operand cannot ride the cp.async trans
 * staging the 2-byte side uses; the path is instruction-throughput bound —
 * spreading the chunk over more threads lost both times it was tried, so
 * do not retry). The ladder's tiles keep the chunk count at or below the
 * thread count, so one chunk per thread is the whole carried state. A
 * thinner bus, 2-byte elements and the predicated boundary chunks fall
 * back to load_crosswise_direct_general; the carry owns the interior,
 * aligned chunks. kOn false (a non-8-bit-crosswise operand) collapses
 * every method to a no-op through the same constexpr guards.
 */
template <typename SmemLayout, typename ElemT, int kThreads, bool kOn> struct CrosswiseCarry {
    static_assert(!kOn || sizeof(ElemT) == 1, "the register carry stages the 8-bit crosswise path");
    static constexpr int kCw = 4;         // contract elems per chunk
    static constexpr int kRowsChunk = 16; // rows per chunk (4 contract runs)
    static constexpr int kK = kOn ? SmemLayout::kChunks * 16 : 0;
    static constexpr int kGroups = kOn ? SmemLayout::kRows / kRowsChunk : 0;
    static constexpr int kTChunks = kOn ? (kK / kCw) * kGroups : 0;
    static constexpr bool kFits = kOn && kTChunks <= kThreads;

    uint4 v[4];   // the chunk's four 16-row contract runs
    int span = 0; // contract span this thread owns
    int rg = 0;   // its 16-row group
    bool active = false;
    bool fast = false; // interior + aligned: the arm the carry can stage

    DEVICE_FORCEINLINE void issue(const ElemT* __restrict__ operand,
                                  int64_t rows,
                                  int64_t contract,
                                  int64_t ld,
                                  int tid,
                                  int64_t k_base,
                                  int64_t block_row) {
        if constexpr (!kFits)
            return;
        active = tid < kTChunks;
        if (!active)
            return;
        span = tid / kGroups;
        rg = tid % kGroups;
        const int64_t r0 = block_row + rg * kRowsChunk;
        /*
         * r0 is a multiple of 16 and p*ld preserves alignment whenever ld has
         * it, so every run of a chunk shares one verdict (same rule as the
         * general loader).
         */
        const bool run_aligned =
            ((reinterpret_cast<uintptr_t>(operand) | (ld * (int64_t)sizeof(ElemT))) & 15) == 0;
        fast = (r0 + kRowsChunk - 1 < rows) && run_aligned;
        const int64_t p0 = k_base + span * kCw;
        if (fast) {
#pragma unroll
            for (int i = 0; i < kCw; ++i)
                v[i] = p0 + i < contract
                           ? __ldg(reinterpret_cast<const uint4*>(operand + (p0 + i) * ld + r0))
                           : make_uint4(0u, 0u, 0u, 0u);
        } else {
            v[0] = make_uint4(0u, 0u, 0u, 0u);
            v[1] = make_uint4(0u, 0u, 0u, 0u);
            v[2] = make_uint4(0u, 0u, 0u, 0u);
            v[3] = make_uint4(0u, 0u, 0u, 0u);
        }
    }

    /*
     * PRMT + STS for the chunk issue() fetched, through the shared span
     * arms — the element-granular fallback (row tail, misaligned base)
     * keeps the synchronous gather so the carry never has to hold
     * predicated state.
     */
    template <typename TileT>
    DEVICE_FORCEINLINE void commit(TileT tile,
                                   const ElemT* __restrict__ operand,
                                   int64_t rows,
                                   int64_t contract,
                                   int64_t ld,
                                   int tid,
                                   int64_t k_base,
                                   int64_t block_row) const {
        if constexpr (!kFits) {
            /*
             * Off (kOn false): nothing to stage. Thin bus (kOn true but not
             * kFits): issue() staged nothing, the round trip stays synchronous
             * here.
             */
            if constexpr (kOn)
                load_crosswise_direct_general<SmemLayout, ElemT, kThreads>(
                    tile, operand, rows, contract, ld, tid, k_base, block_row);
            return;
        }
        if (!active)
            return;
        if (fast) {
            crosswise_perm_span(tile, rg, span, v);
            return;
        }
        crosswise_gather_span(tile, operand, rows, contract, ld, k_base,
                              block_row + rg * kRowsChunk, rg, span);
    }
};

} // namespace gemm
} // namespace astrai
