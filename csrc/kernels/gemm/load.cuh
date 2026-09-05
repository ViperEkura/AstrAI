#pragma once
// Operand loaders: swizzled shared-memory staging for congruous operands
// (cp.async, predicated and interior variants, plus the loop-carried
// prefetch state) and the direct LDG+PRMT path for crosswise operands.
// The staging invariants and the swizzle derivation live in
// docs/developer/cuda_kernels.md.

#include "common/cp_async.cuh"
#include "gemm/common.h"
#include "policy.cuh"

namespace astrai {
namespace gemm {

// log2 of a compile-time power of two (for the swizzle shifts).
template <int N, int Acc = 0>
struct log2_const : log2_const<(N >> 1), Acc + 1> {};
template <int Acc>
struct log2_const<1, Acc> {
    static constexpr int value = Acc;
};

// Swizzled address inside a flat [rows * K] staging tile: the 16B chunk
// index is XORed with the row bits at [3, 3+log2(kChunks)) so a warp's
// ldmatrix fragment load (8 consecutive rows x 16B) hits all 32 banks
// exactly once; chunks stay contiguous, so cp.async staging is unaffected.
// All chunk math is in BYTES (16B granularity); element counts enter only
// through kChunkElems = 16 / sizeof(T).
template <int K, typename ElemT>
__device__ __forceinline__ ElemT* tile_at(ElemT* tile, int row, int col) {
    constexpr int kChunkElems = 16 / sizeof(ElemT);  // elems per 16B chunk
    constexpr int kChunks = K / kChunkElems;      // 16B chunks per row
    static_assert(kChunks >= 1 && (kChunks & (kChunks - 1)) == 0,
                  "swizzle needs a power-of-two 16B-chunk count");
    constexpr int kShift = 3 - log2_const<kChunks>::value;
    constexpr int kChunkShift = log2_const<kChunkElems>::value;
    return tile + row * K +
           ((((col >> kChunkShift) ^ ((row >> kShift) & (kChunks - 1)))
                  << kChunkShift) +
             (col & (kChunkElems - 1)));
}

// Trans-tile mirror of tile_at for crosswise 16-bit operands: the tile is
// [K][RowsT] (k rows, the non-contract dim contiguous), staged by cp.async
// directly from the crosswise global layout and read through
// ldmatrix.trans. The chunk XOR runs the other way — swizzled by the K-row
// bits — so the 8 k-rows one ldmatrix matrix addresses at a fixed column
// window land on distinct chunks (conflict-free). Only the low 3 row bits
// can join the XOR (the LDSM contract gives 8 rows per matrix), so tiles
// wider than 8 chunks leave the upper chunk bits unswizzled — each
// matrix's rows stay conflict-free either way.
template <int RowsT, typename ElemT>
__device__ __forceinline__ ElemT* tile_at_trans(ElemT* tile, int k, int col) {
    constexpr int kChunkElems = 16 / sizeof(ElemT);
    constexpr int kChunks = RowsT / kChunkElems;
    static_assert(kChunks >= 1 && (kChunks & (kChunks - 1)) == 0,
                  "swizzle needs a power-of-two 16B-chunk count");
    constexpr int kSwz = kChunks < 8 ? kChunks : 8;
    constexpr int kChunkShift = log2_const<kChunkElems>::value;
    return tile + (int64_t)k * RowsT +
           ((((col >> kChunkShift) ^ (k & (kSwz - 1))) << kChunkShift) +
             (col & (kChunkElems - 1)));
}

// Stage-load a CONGRUOUS operand (contract-contiguous storage — the only
// cp.async-able shape) into the flat [rows * K] swizzled tile. kInterior
// drops all predication: valid only for a fully interior CTA (whole rows,
// 16B-aligned base|ld, k_base + K <= contract); the address math then folds
// to one immediate XOR per chunk (see the design notes). Crosswise operands
// go through load_operand_tile_trans (16-bit) or load_crosswise_direct
// (8-bit) instead.
template <typename ElemT, int K, int RowsTile,
          int kThreads, bool kInterior = false>
__device__ __forceinline__ void
load_operand_tile(ElemT* tile, const ElemT* __restrict__ operand, int64_t rows,
                  int64_t contract, int64_t ld, int tid, int64_t k_base,
                  int64_t block_row) {
    constexpr int kChunkElems = 16 / sizeof(ElemT);
    constexpr int kChunks = K / kChunkElems;
    static_assert(RowsTile * kChunks % kThreads == 0,
                  "tile chunks must divide evenly across threads");
    constexpr int kCpt = RowsTile * kChunks / kThreads;  // chunks per thread
    constexpr int kCpr = kChunks / kCpt;  // chunks per row slice
    const int r = tid / kCpr;
    const int c0 = (tid % kCpr) * kCpt * kChunkElems;
    if constexpr (kInterior) {
        const char* src = reinterpret_cast<const char*>(
            operand + (block_row + r) * ld + k_base + c0);
        const uintptr_t dst =
            reinterpret_cast<uintptr_t>(tile_at<K>(tile, r, c0));
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(reinterpret_cast<ElemT*>(dst ^ (j << 4)),
                                src + j * 16);
    } else {
        const int64_t row = block_row + r;
        const bool row_ok = row < rows;
        // k_base and every c are multiples of 16, so all chunks share the
        // row base's alignment verdict.
        const auto* src = operand + row * ld + k_base;
        const bool chunk_aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
#pragma unroll
        for (int j = 0; j < kCpt; ++j) {
            const int c = c0 + j * kChunkElems;
            ElemT* dst = tile_at<K>(tile, r, c);
            if (row_ok && chunk_aligned &&
                k_base + c + kChunkElems - 1 < contract) {
                astrai::cp_async_16(dst, src + c);
            } else {
                // Tail chunk / misaligned base / OOB row: scalar fill.
# pragma unroll
                for (int i = 0; i < kChunkElems; ++i)
                    dst[i] =
                        row_ok && k_base + c + i < contract ? src[c + i] : ElemT(0.0f);
            }
        }
    }
}

// Stage-load a CROSSWISE 16-bit operand by cp.async: the global runs along
// the non-contract dim (row-contiguous 16B) copy straight into the
// transposed [K][RowsT] tile, and ldmatrix.trans does the matrix turn at
// fragment-extraction time (b16-only instruction — 8-bit crosswise
// operands cannot take this path and keep the LDG+PRMT staging).
// Role-swapped mirror of load_operand_tile: the tile's row dim is the
// contract dim here, so the predication axes trade places.
template <typename ElemT, int RowsTile, int kK, int kThreads,
          bool kInterior = false>
__device__ __forceinline__ void
load_operand_tile_trans(ElemT* tile, const ElemT* __restrict__ operand,
                        int64_t rows, int64_t contract, int64_t ld, int tid,
                        int64_t k_base, int64_t block_row) {
    constexpr int kChunkElems = 16 / sizeof(ElemT);
    constexpr int kChunks = RowsTile / kChunkElems;
    static_assert(sizeof(ElemT) == 2, "trans staging is 16-bit only");
    static_assert(kK * kChunks % kThreads == 0,
                  "tile chunks must divide evenly across threads");
    constexpr int kCpt = kK * kChunks / kThreads;
    constexpr int kCpr = kChunks / kCpt;  // n-chunk slices per k row
    const int kr = tid / kCpr;            // k row within the tile
    const int c0 = (tid % kCpr) * kCpt * kChunkElems;
    if constexpr (kInterior) {
        const char* src = reinterpret_cast<const char*>(
            operand + (k_base + kr) * ld + block_row + c0);
        const uintptr_t dst =
            reinterpret_cast<uintptr_t>(tile_at_trans<RowsTile>(tile, kr, c0));
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(reinterpret_cast<ElemT*>(dst ^ (j << 4)),
                                src + j * 16);
    } else {
        // block_row and every c are multiples of kChunkElems; the
        // alignment verdict is shared across each row's chunks only when
        // ld is (see the scalar fallback).
        const bool row_ok = k_base + kr < contract;
#pragma unroll
        for (int j = 0; j < kCpt; ++j) {
            const int c = c0 + j * kChunkElems;
            const int64_t col = block_row + c;
            ElemT* dst = tile_at_trans<RowsTile>(tile, kr, c);
            const bool col_ok = col < rows;
            const auto* src = operand + (k_base + kr) * ld + col;
            if (row_ok && col_ok &&
                (reinterpret_cast<uintptr_t>(src) & 15) == 0 &&
                col + kChunkElems <= rows) {
                astrai::cp_async_16(dst, src);
            } else {
# pragma unroll
                for (int i = 0; i < kChunkElems; ++i)
                    dst[i] = row_ok && col + i < rows ? src[i] : ElemT(0.0f);
            }
        }
    }
}

// Loop-carried prefetch state for one congruous-or-trans operand ring:
// per-thread (r, c0) mapping with the swizzled stage destination and global
// source pointer carried across k-tiles, so each prefetch chunk is one
// LDGSTS issued straight from registers. The guard is a property of the
// operand's layout, so it lives in the type: the false specialization
// (synchronous 8-bit crosswise operand) is an empty no-op. kTrans selects
// the crosswise 16-bit geometry: the tile is [kK][kRowsTile] (k rows), so
// the thread's row is a k line and the per-tile source advance is
// kK * ld instead of kK.
template <bool kAsync, typename ElemT, int kK, int kRowsTile, int kThreads,
          bool kTrans = false>
struct PrefetchCarry;

template <typename ElemT, int kK, int kRowsTile, int kThreads, bool kTrans>
struct PrefetchCarry<true, ElemT, kK, kRowsTile, kThreads, kTrans> {
    static constexpr int kChunkElems = 16 / sizeof(ElemT);
    static constexpr int kCpt =
        (kTrans ? kK * (kRowsTile / kChunkElems)
                : kRowsTile * (kK / kChunkElems)) /
        kThreads;
    static constexpr int kCpr =
        (kTrans ? kRowsTile / kChunkElems : kK / kChunkElems) / kCpt;
    // All carried state is in BYTES: the smem write ring and the global
    // source pointer both advance by the byte-sized stage stride.
    static constexpr unsigned kKBytes = (unsigned)kK * sizeof(ElemT);
    unsigned wr = 0;    // current stage's swizzled destination offset
    unsigned wr0 = 0;   // slot-0 wrap base
    unsigned wrEnd = 0; // one-past-the-ring sentinel
    const char* src = nullptr;      // current tile's global source bytes
    int64_t srcStep = 0;            // per-tile source advance (bytes)

    __device__ __forceinline__ PrefetchCarry(
        const ElemT* ring, int ringSlots, int stageBytes, const ElemT* operand,
        int64_t ld, int64_t blockRow, int tid, int firstTile) {
        const int r = tid / kCpr;
        const int c0 = (tid % kCpr) * kCpt * kChunkElems;
        const int stageElems = stageBytes / (int)sizeof(ElemT);
        const ElemT* slot0 =
            ring + (int64_t)(firstTile % ringSlots) * stageElems;
        const unsigned laneOff = static_cast<unsigned>(
            (const char*)(kTrans ? tile_at_trans<kRowsTile>(slot0, r, c0)
                                 : tile_at<kK>(slot0, r, c0)) -
            (const char*)slot0);
        const unsigned base = __cvta_generic_to_shared(ring) + laneOff;
        wr = base + (unsigned)((int64_t)(firstTile % ringSlots) * stageBytes);
        wr0 = base;
        wrEnd = base + (unsigned)((int64_t)ringSlots * stageBytes);
        if constexpr (kTrans) {
            src = reinterpret_cast<const char*>(
                operand + ((int64_t)firstTile * kK + r) * ld + blockRow + c0);
            srcStep = (int64_t)kK * ld * sizeof(ElemT);  // k advances rows
        } else {
            src = reinterpret_cast<const char*>(
                      operand + (blockRow + r) * ld + c0) +
                  (int64_t)firstTile * kKBytes;
            srcStep = kKBytes;  // k advances the contiguous columns
        }
    }

    // Emit this thread's chunks for the current tile; pf false (loop tail)
    // zero-fills into the slot compute(i-1) already released.
    __device__ __forceinline__ void emit(bool pf) const {
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(wr ^ (unsigned)(j << 4), src + j * 16, pf);
    }

    __device__ __forceinline__ void advance(int stageBytes) {
        wr += (unsigned)stageBytes;
        if (wr == wrEnd) wr = wr0;
        src += srcStep;
    }
};

template <typename ElemT, int kK, int kRowsTile, int kThreads, bool kTrans>
struct PrefetchCarry<false, ElemT, kK, kRowsTile, kThreads, kTrans> {
    __device__ __forceinline__ PrefetchCarry(
        const ElemT*, int, int, const ElemT*, int64_t, int64_t, int, int) {}
    __device__ __forceinline__ void emit(bool) const {}
    __device__ __forceinline__ void advance(int) {}
};

// Direct (synchronous) crosswise load into a canonical rotating stage:
// LDG.128 runs (4 x 16B of the non-contract dim) + in-register transpose
// (PRMT) + 16 STS.32. Crosswise operands cannot cp.async into the
// canonical tile (a 16B global run holds contract positions for a run of
// the other dim), so they take this path; a staged smem->smem variant
// measured 15-20% slower and was removed (see git history).
//
// One chunk = 64B of global memory staging one 16-row group:
//   1-byte elements: 4 runs of 16 rows x 4 contract positions; the PRMT
//     byte-perm gathers one 32-bit word per row across the four runs;
//   2-byte elements: 2 contract positions x 2 eight-row halves; a 16B run
//     covers only 8 rows, and the transpose selects halfwords — one
//     PRMT per output word (the byte selector already spans both source
//     words: 0x5410 low pair, 0x7632 high pair).
template <typename ElemT, int K, int RowsTile, int kThreads>
__device__ __forceinline__ void
load_crosswise_direct(ElemT* tile, const ElemT* __restrict__ operand, int64_t rows,
                      int64_t contract, int64_t ld, int tid, int64_t k_base,
                      int64_t block_row) {
    static_assert(sizeof(ElemT) == 1 || sizeof(ElemT) == 2,
                  "crosswise LDG+PRMT staging requires 1- or 2-byte elements");
    constexpr int kCw = sizeof(ElemT) == 1 ? 4 : 2;  // contract elems per chunk
    constexpr int kSpans = K / kCw;    // contract spans per tile
    constexpr int kGroups = RowsTile / 16;
    constexpr int kTChunks = kSpans * kGroups;  // 64B chunks per tile
    // r0 is a multiple of 16 and p*ld preserves alignment whenever ld has
    // it, so every run of a chunk shares one alignment verdict.
    const bool run_aligned =
        ((reinterpret_cast<uintptr_t>(operand) | (ld * (int64_t)sizeof(ElemT))) &
         15) == 0;
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
                // Contract tail: a run past k carries zero bytes; they flow
                // through the transpose like any other value.
                if constexpr (sizeof(ElemT) == 1) {
                    // v[s] = 16 rows at contract p0+s (run index i = s).
                    if (p0 + i < contract)
                        v[i] = *reinterpret_cast<const uint4*>(
                            operand + (p0 + i) * ld + r0);
                    else
                        v[i] = make_uint4(0u, 0u, 0u, 0u);
                } else {
                    // v[s][h] = 8 rows at contract p0+s (flat i = s*2+h).
                    const int s = i >> 1, h = i & 1;
                    if (p0 + s < contract)
                        v[i] = *reinterpret_cast<const uint4*>(
                            operand + (p0 + s) * ld + r0 + h * 8);
                    else
                        v[i] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            const unsigned* bytes = reinterpret_cast<const unsigned*>(v);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                unsigned w;
                if constexpr (sizeof(ElemT) == 1) {
                    // word i = row r0+i's span: byte i of each of the four
                    // runs [v0.b(i), v1.b(i), v2.b(i), v3.b(i)].
                    const unsigned nib = i & 3;
                    const unsigned sel = nib | ((nib + 4) << 4);
                    const unsigned w01 =
                        __byte_perm(bytes[0 + (i >> 2)], bytes[4 + (i >> 2)], sel);
                    const unsigned w23 =
                        __byte_perm(bytes[8 + (i >> 2)], bytes[12 + (i >> 2)], sel);
                    w = __byte_perm(w01, w23, 0x5410u);
                } else {
                    // word i = row r0+i's element pair (contracts p0, p0+1):
                    // uint4 v[s*2+h] holds 8 rows of contract p0+s (h =
                    // i>>3), word (i>>1)&3, halfword i&1 — one selector
                    // spans both source words: 0x5410 low pair, 0x7632 high.
                    const unsigned* v0 =
                        bytes + ((i >> 3) * 4 + ((i >> 1) & 3));
                    const unsigned* v1 = v0 + 8;  // p0+1 run, same h/j
                    w = __byte_perm(*v0, *v1, (i & 1) ? 0x7632u : 0x5410u);
                }
                *reinterpret_cast<unsigned*>(
                    tile_at<K>(tile, rg * 16 + i, span * kCw)) = w;
            }
        } else {
            // Row-tail or misaligned chunk: element-granular gather with
            // per-row predication; contract-tail columns zero-fill.
#pragma unroll
            for (int s = 0; s < kCw; ++s) {
                const int col = span * kCw + s;
                if (k_base + col >= contract) {
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                        *tile_at<K>(tile, rg * 16 + i, col) = ElemT(0.0f);
                    continue;
                }
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int64_t r_idx = r0 + i;
                    *tile_at<K>(tile, rg * 16 + i, col) =
                        r_idx < rows
                            ? operand[(k_base + col) * ld + r_idx]
                            : ElemT(0.0f);
                }
            }
        }
    }
}

}  // namespace gemm
}  // namespace astrai
