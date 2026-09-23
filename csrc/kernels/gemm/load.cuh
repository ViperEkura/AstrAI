#pragma once
// Operand loaders: swizzled shared-memory staging for congruous operands
// (cp.async, predicated and interior variants, plus the loop-carried
// prefetch state) and the direct LDG+PRMT path for crosswise operands.
// The staging invariants and the swizzle derivation live in
// docs/developer/cuda_kernels.md.

#include "common/pipeline.cuh"
#include "common/tensor.cuh"

namespace astrai {
namespace gemm {

// One operand element's raw byte. The 8-bit storages are class types (fp8) as
// often as integers, and a numeric conversion would round the bit pattern away
// — every packed-grid elementwise path reads bytes through here.
template <typename ElemT>
__device__ __forceinline__ unsigned raw_byte(const ElemT& e) {
    return *reinterpret_cast<const unsigned char*>(&e);
}

// Stage-load one operand tile from global memory into its swizzled ring
// slot. Two geometries are the role-swapped mirror of the SAME loop, so one
// template bit composes the whole function instead of a parallel copy:
//
//   kTransposed = false — CONGRUOUS operand (contract-contiguous storage,
//     the only cp.async-able shape) into the canonical [rows][kK] tile;
//   kTransposed = true  — CROSSWISE 16-bit operand (row-contiguous 16B
//     runs) into the transposed [kK][rows] tile, where ldmatrix.trans does
//     the matrix turn at fragment-extraction time (b16-only — 8-bit
//     crosswise operands keep the LDG+PRMT staging).
//
// Under transposed staging the tile's line axis is the contract dim, so the
// predication axes trade places with a canonical run's. The staged tile is
// the TRANS layout instance: chunks swizzled by the k-row bits (custom XOR,
// source is the row field) so the 8 k-rows one ldmatrix.trans matrix
// addresses at a fixed column window land on distinct chunks (conflict-
// free); only the low 3 row bits can join the XOR (LDSM gives 8 rows per
// matrix), so tiles wider than 8 chunks leave the upper bits unswizzled.
//
// The tile arrives as a Tensor over the staged layout: the swizzled address
// is the tensor's operator(), dispatching to the layout op. kInterior drops
// all predication — valid only for a fully interior CTA (whole lines,
// 16B-aligned base|ld, k_base + kK <= contract); the address math then folds
// to one immediate XOR per chunk (see the design notes).
template <typename SmemLayout, typename ElemT,
          int kThreads, bool kTransposed = false, bool kInterior = false>
__device__ __forceinline__ void
load_operand_tile(Tensor<PtrEngine<ElemT>, SmemLayout> tile,
                  const ElemT* __restrict__ operand, int64_t rows,
                  int64_t contract, int64_t ld, int tid, int64_t k_base,
                  int64_t block_row) {
    static_assert(!kTransposed || sizeof(ElemT) == 2,
                  "trans staging is 16-bit only");
    constexpr int kChunkElems = 16 / sizeof(ElemT);
    constexpr int kTileLines = SmemLayout::kRows;  // lines staged per tile
    constexpr int kChunks = SmemLayout::kChunks;   // chunks per line
    constexpr int kTotalChunks = kTileLines * kChunks;
    // The bus may be under-subscribed: a 1-byte operand halves the chunks a
    // line carries, so a narrow tile can hold fewer chunks than there are
    // threads (a 16-warp small CTA against a 1-byte B, or any kK=32 1-byte
    // side). Threads then take one chunk each and the rest stage nothing —
    // the same true-skip shape load_crosswise_direct's grid-stride loop has
    // always had. What cannot relax is the aligned run: the XOR chunk
    // stepping below (dst ^ (j << 4)) is the swizzle of c0c + j only
    // because a thread's chunks are one aligned power-of-two run inside
    // the line — j's bits never reach c0c's.
    static_assert(kTotalChunks % kThreads == 0 || kTotalChunks < kThreads,
                  "tile chunks must divide the threads or under-subscribe the bus");
    constexpr int kCpt =  // chunks per thread
        kTotalChunks < kThreads ? 1 : kTotalChunks / kThreads;
    static_assert(kCpt > 0 && (kCpt & (kCpt - 1)) == 0,
                  "XOR chunk stepping needs a power-of-two chunks-per-thread");
    // A thread's run must stay inside ONE line: the r/c0 decomposition
    // (kCpr below) divides kChunks BY kCpt, and a run wider than the line
    // sends kCpr to zero — tid/0 garbage addresses and a wedged cp.async
    // (observed 2026-09-16 on 128x256x32 with 4 warps: kernel hangs at
    // 100% GPU on the first launch). Fully subscribed this says the staged
    // extent (bm, bn) must not exceed the thread count; the under-subscribed
    // arm keeps kCpt 1 and cannot violate it.
    static_assert(kCpt <= kChunks,
                  "a thread's 16B run must fit one staged line: the staged "
                  "extent cannot exceed the thread count");
    constexpr int kCpr = kChunks / kCpt;  // chunks per line slice
    const int r = tid / kCpr;             // line within the tile
    const int c0 = (tid % kCpr) * kCpt * kChunkElems;
    if (r >= kTileLines) return;  // no chunks for this thread on this bus
    // The mirror is three axes: the tile line sources from block_row
    // (canonical) or k_base (transposed) rows; the 16B run starts at
    // k_base (canonical) or block_row (transposed); and each axis is cut
    // by the extent that is NOT the one the run walks — the line's own
    // extent, then the other (non-contract vs contract).
    const int64_t line0 = kTransposed ? k_base : block_row;
    const int64_t run0 = kTransposed ? block_row : k_base;
    const int64_t line_ext = kTransposed ? contract : rows;
    const int64_t run_ext = kTransposed ? rows : contract;
    if constexpr (kInterior) {
        const char* src = reinterpret_cast<const char*>(
            operand + (line0 + r) * ld + run0 + c0);
        const uintptr_t dst =
            reinterpret_cast<uintptr_t>(tile(r, c0));
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(reinterpret_cast<ElemT*>(dst ^ (j << 4)),
                                src + j * 16);
    } else {
        const int64_t line = line0 + r;
        const bool line_ok = line < line_ext;
        // line0, run0, c0 and every j step are multiples of 16, so all
        // chunks share the run's alignment verdict (verdicts only differ
        // ACROSS lines, when ld is not 16B — see the scalar fallback).
        const auto* src = operand + line * ld + run0 + c0;
        const bool chunk_aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
        const uintptr_t dst =
            reinterpret_cast<uintptr_t>(tile(r, c0));
#pragma unroll
        for (int j = 0; j < kCpt; ++j) {
            const int c = j * kChunkElems;
            const int64_t col = run0 + c0 + c;
            ElemT* dstj = reinterpret_cast<ElemT*>(dst ^ (unsigned)(j << 4));
            if (chunk_aligned) {
                // CUTLASS-style zero-fill predication: one cp.async whose
                // runtime src-size loads the valid prefix (whole chunk,
                // the extent tail cut or nothing for an OOB line) and the
                // hardware zero-fills the remainder.
                const int64_t room = line_ok ? run_ext - col : 0;
                const int bytes = room >= kChunkElems
                                      ? 16
                                      : room > 0
                                            ? (int)(room * (int64_t)sizeof(ElemT))
                                            : 0;
                astrai::cp_async_16(dstj, src + c, bytes);
            } else {
                // Misaligned base only: element-granular fallback.
# pragma unroll
                for (int i = 0; i < kChunkElems; ++i)
                    dstj[i] = line_ok && col + i < run_ext ? src[c + i]
                                                           : ElemT(0.0f);
            }
        }
    }
}

// Loop-carried prefetch state for one congruous-or-trans operand ring:
// per-thread (r, c0) mapping with the swizzled stage destination and global
// source pointer carried across k-tiles, so each prefetch chunk is one
// LDGSTS issued straight from registers. The geometry rides the operand's
// Ring tensor (slots, stage stride, staged layout). kTrans selects the
// crosswise 16-bit geometry on the SOURCE side: the tile's rows are k
// lines, so the per-tile source advance is kK * ld instead of kK.
// kAsync=false (synchronous 8-bit crosswise operand) is an empty no-op.
template <bool kAsync, typename RingT, int kThreads, bool kTrans = false>
struct PrefetchCarry;

template <typename RingT, int kThreads, bool kTrans>
struct PrefetchCarry<true, RingT, kThreads, kTrans> {
    using ElemT = typename RingT::Elem;  // Ring = Tensor<PtrEngine, RingLayout>
    using SmemLayout = typename RingT::Layout::Stage;  // per-stage layout
    static constexpr int kChunkElems = 16 / sizeof(ElemT);
    // The layout carries the tile geometry (rows x chunks per row),
    // whichever way round the staging runs. Same bus rule as
    // load_operand_tile: an under-subscribed bus (fewer chunks than
    // threads) leaves the surplus threads inactive rather than illegal.
    static constexpr int kTotalChunks =
        SmemLayout::kRows * SmemLayout::kChunks;
    static_assert(kTotalChunks % kThreads == 0 || kTotalChunks < kThreads,
                  "tile chunks must divide the threads or under-subscribe the bus");
    static constexpr int kCpt =
        kTotalChunks < kThreads ? 1 : kTotalChunks / kThreads;
    static_assert(kCpt > 0 && (kCpt & (kCpt - 1)) == 0,
                  "XOR chunk stepping needs a power-of-two chunks-per-thread");
    static constexpr int kCpr = SmemLayout::kChunks / kCpt;
    // All carried state is in BYTES: the smem write ring and the global
    // source pointer both advance by the byte-sized stage stride.
    static constexpr int kK =
        kTrans ? SmemLayout::kRows : SmemLayout::kChunks * kChunkElems;
    static constexpr unsigned kKBytes = (unsigned)kK * sizeof(ElemT);
    unsigned wr = 0;    // current stage's swizzled destination offset
    unsigned wr0 = 0;   // slot-0 wrap base
    unsigned wrEnd = 0; // one-past-the-ring sentinel
    const char* src = nullptr;      // current tile's global source bytes
    int64_t srcStep = 0;            // per-tile source advance (bytes)
    bool active = true;             // false: bus under-subscribed, no chunks

    __device__ __forceinline__ PrefetchCarry(
        const RingT& ring, const ElemT* operand,
        int64_t ld, int64_t blockRow, int tid, int firstTile) {
        const int rRaw = tid / kCpr;
        active = rRaw < SmemLayout::kRows;
        // An inactive thread's (r, c0) maps to no staged chunk; the carried
        // offsets are computed at a clamped r and never dereferenced.
        const int r = active ? rRaw : 0;
        const int c0 = (tid % kCpr) * kCpt * kChunkElems;
        const ElemT* slot0 = astrai::stage_of(ring, firstTile).engine.ptr;
        const unsigned laneOff = static_cast<unsigned>(
            (const char*)astrai::stage_of(ring, firstTile)(r, c0) -
            (const char*)slot0);
        const unsigned base =
            __cvta_generic_to_shared(ring.engine.ptr) + laneOff;
        wr = base + (unsigned)((int64_t)(firstTile % RingT::Layout::kSlots) *
                               RingT::Layout::kStageBytes);
        wr0 = base;
        wrEnd = base + (unsigned)RingT::Layout::kTotalBytes;
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
    // zero-fills into the slot compute(i-1) already released. An inactive
    // thread owns no chunks, so it emits nothing at all.
    __device__ __forceinline__ void emit(bool pf) const {
        if (!active) return;
#pragma unroll
        for (int j = 0; j < kCpt; ++j)
            astrai::cp_async_16(wr ^ (unsigned)(j << 4), src + j * 16, pf);
    }

    __device__ __forceinline__ void advance() {
        wr += (unsigned)RingT::Layout::kStageBytes;
        if (wr == wrEnd) wr = wr0;
        src += srcStep;
    }
};

template <typename RingT, int kThreads, bool kTrans>
struct PrefetchCarry<false, RingT, kThreads, kTrans> {
    __device__ __forceinline__ PrefetchCarry(
        const RingT&, const typename RingT::Elem*, int64_t, int64_t, int,
        int) {}
    __device__ __forceinline__ void emit(bool) const {}
    __device__ __forceinline__ void advance() {}
};

// The two arms one 16-row crosswise chunk can take, shared by the general
// grid-stride loader and the register carry below — the pair must never
// diverge on the perm sequence or the predication, or the same operand
// would stage differently depending on bus width.
//
// Fast arm: four contract runs (uint4 each, already fetched into the
// register file) -> one PRMT pass -> 16 packed span words in the staging
// tile. Slow arm (row tail / misaligned base): element-granular gather
// with per-row predication; contract-tail columns zero-fill.
template <typename TileT>
__device__ __forceinline__ void crosswise_perm_span(TileT tile, int rg,
                                                    int span, const uint4* v) {
    const unsigned* bytes = reinterpret_cast<const unsigned*>(v);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        // Word i = row r0+i's span: byte i of each of the four runs
        // [v0.b(i), v1.b(i), v2.b(i), v3.b(i)]; run s is one uint4 (16
        // bytes = 16 rows), so word i>>2 of run s is bytes[4*s + (i>>2)].
        const unsigned nib = i & 3;
        const unsigned sel = nib | ((nib + 4) << 4);
        const unsigned w01 =
            __byte_perm(bytes[0 + (i >> 2)], bytes[4 + (i >> 2)], sel);
        const unsigned w23 =
            __byte_perm(bytes[8 + (i >> 2)], bytes[12 + (i >> 2)], sel);
        *reinterpret_cast<unsigned*>(tile(rg * 16 + i, span * 4)) =
            __byte_perm(w01, w23, 0x5410u);
    }
}

template <typename TileT, typename ElemT>
__device__ __forceinline__ void crosswise_gather_span(
    TileT tile, const ElemT* __restrict__ operand, int64_t rows,
    int64_t contract, int64_t ld, int64_t k_base, int64_t r0, int rg,
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
            *tile(rg * 16 + i, col) = r0 + i < rows
                                           ? operand[(k_base + col) * ld + r0 + i]
                                           : ElemT(0.0f);
    }
}

// Direct (synchronous) crosswise load into a canonical rotating stage:
// LDG.128 runs (4 x 16B of the non-contract dim) + in-register transpose
// (PRMT) + 16 STS.32. Crosswise operands cannot cp.async into the
// canonical tile (a 16B global run holds contract positions for a run of
// the other dim), so they take the register route; a staged smem->smem
// variant measured 15-20% slower and was removed (see git history).
//
// One chunk = 64B of global memory staging one 16-row group: 4 runs of
// 16 rows x 4 contract positions; the PRMT byte-perm gathers one 32-bit
// word per row across the four runs.
//
// This is the GENERAL form: a grid-stride loop over the tile's chunks, so
// any tile geometry stages correctly however few threads it has. The path
// the ladders actually instantiate has a two-phase sibling (CrosswiseCarry,
// below) which load_crosswise_direct selects; this one covers the bus
// under-subscription case.
template <typename SmemLayout, typename ElemT, int kThreads>
__device__ __forceinline__ void
load_crosswise_direct_general(Tensor<PtrEngine<ElemT>, SmemLayout> tile,
                              const ElemT* __restrict__ operand, int64_t rows,
                              int64_t contract, int64_t ld, int tid,
                              int64_t k_base, int64_t block_row) {
    static_assert(sizeof(ElemT) == 1,
                  "crosswise LDG+PRMT staging requires 1-byte elements");
    constexpr int kRowsTile = SmemLayout::kRows;
    constexpr int kK = SmemLayout::kChunks * 16;
    constexpr int kCw = 4;  // contract elems per chunk
    constexpr int kSpans = kK / kCw;    // contract spans per tile
    constexpr int kGroups = kRowsTile / 16;
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
                // through the transpose like any other value. v[i] = 16 rows
                // at contract p0+i.
                if (p0 + i < contract)
                    v[i] = __ldg(reinterpret_cast<const uint4*>(
                        operand + (p0 + i) * ld + r0));
                else
                    v[i] = make_uint4(0u, 0u, 0u, 0u);
            }
            crosswise_perm_span(tile, rg, span, v);
        } else {
            crosswise_gather_span(tile, operand, rows, contract, ld, k_base,
                                  r0, rg, span);
        }
    }
}

// ---------------------------------------------------------------------------
// Two-phase register staging for the 8-bit crosswise operand.
//
// The general loader is a synchronous round trip in front of the k-tile's MMA
// phase, so its global latency sits on the critical path of every iteration.
// At the 64x64 CTA that costs 243 TF against 350 TF for the same tile staged
// congruously (interleaved A/B 2026-09-18, qkv fp8) — the sync staging's
// whole share, and the kernel's top stall.
//
// The carry splits it across the phase boundary: issue() fires the global runs
// one phase ahead of the transpose, commit() does PRMT + STS after the MMA
// phase, so the latency hides behind tensor-pipe work (ncu long-scoreboard
// 1.18 -> 0.42, barrier 5.35 -> 3.53, the 64x64 kernel 1.54 -> 1.34 ms).
// One thread owns the whole 64B chunk: the four contract runs have to meet in
// one register file for the byte-perm, which is also why the 1-byte operand
// cannot ride the cp.async trans staging the 2-byte side uses.
//
// Spreading that chunk over more threads was tried TWICE and lost both times,
// which is what pins this path as instruction-throughput bound rather than
// latency or parallelism bound (profile: issue 39%, not_selected 0.63, i.e.
// the scheduler is not saturated and the CTA waits on its two loader warps):
//
//   * 8-row chunks (32B, LDG.64, 2x the loader threads): qkv fp8 TT -1.7%,
//     NN -6.2%, TN -7.6% — the LDG count doubles and the loads narrow.
//   * lane pair per chunk (2 runs each + one shfl_xor per row, LDG.128 kept,
//     identical LDG/PRMT/STS totals): qkv fp8 TT -7.6%, NN -13%, TN -15%,
//     square -13..-18% — the exchange costs more than the balance buys.
//
// The ladder's tiles all keep the chunk count at or below the thread count,
// so one chunk per thread is the whole carried state. A thinner bus (more
// chunks than threads), 2-byte elements and the predicated boundary chunks
// fall back to load_crosswise_direct_general; the carry owns the interior,
// aligned chunks.
template <typename SmemLayout, typename ElemT, int kThreads, bool kOn>
struct CrosswiseCarry;

template <typename SmemLayout, typename ElemT, int kThreads>
struct CrosswiseCarry<SmemLayout, ElemT, kThreads, false> {
    __device__ __forceinline__ void issue(const ElemT*, int64_t, int64_t,
                                          int64_t, int, int64_t, int64_t) {}
    template <typename TileT>
    __device__ __forceinline__ void commit(TileT, const ElemT*, int64_t,
                                           int64_t, int64_t, int, int64_t,
                                           int64_t) const {}
};

template <typename SmemLayout, typename ElemT, int kThreads>
struct CrosswiseCarry<SmemLayout, ElemT, kThreads, true> {
    static_assert(sizeof(ElemT) == 1,
                  "the register carry stages the 8-bit crosswise path");
    static constexpr int kCw = 4;          // contract elems per chunk
    static constexpr int kRowsChunk = 16;  // rows per chunk (4 contract runs)
    static constexpr int kK = SmemLayout::kChunks * 16;
    static constexpr int kGroups = SmemLayout::kRows / kRowsChunk;
    static constexpr int kTChunks = (kK / kCw) * kGroups;
    static constexpr bool kFits = kTChunks <= kThreads;

    uint4 v[4];      // the chunk's four 16-row contract runs
    int span = 0;    // contract span this thread owns
    int rg = 0;      // its 16-row group
    bool active = false;
    bool fast = false;  // interior + aligned: the arm the carry can stage

    __device__ __forceinline__ void issue(const ElemT* __restrict__ operand,
                                          int64_t rows, int64_t contract,
                                          int64_t ld, int tid, int64_t k_base,
                                          int64_t block_row) {
        if constexpr (!kFits) return;
        active = tid < kTChunks;
        if (!active) return;
        span = tid / kGroups;
        rg = tid % kGroups;
        const int64_t r0 = block_row + rg * kRowsChunk;
        // r0 is a multiple of 16 and p*ld preserves alignment whenever ld has
        // it, so every run of a chunk shares one verdict (same rule as the
        // general loader).
        const bool run_aligned =
            ((reinterpret_cast<uintptr_t>(operand) |
              (ld * (int64_t)sizeof(ElemT))) &
             15) == 0;
        fast = (r0 + kRowsChunk - 1 < rows) && run_aligned;
        const int64_t p0 = k_base + span * kCw;
        if (fast) {
#pragma unroll
            for (int i = 0; i < kCw; ++i)
                v[i] = p0 + i < contract
                           ? __ldg(reinterpret_cast<const uint4*>(
                                 operand + (p0 + i) * ld + r0))
                           : make_uint4(0u, 0u, 0u, 0u);
        } else {
            v[0] = make_uint4(0u, 0u, 0u, 0u);
            v[1] = make_uint4(0u, 0u, 0u, 0u);
            v[2] = make_uint4(0u, 0u, 0u, 0u);
            v[3] = make_uint4(0u, 0u, 0u, 0u);
        }
    }

    // PRMT + STS for the chunk issue() fetched, through the shared span
    // arms — the element-granular fallback (row tail, misaligned base)
    // keeps the synchronous gather so the carry never has to hold
    // predicated state.
    template <typename TileT>
    __device__ __forceinline__ void commit(TileT tile,
                                           const ElemT* __restrict__ operand,
                                           int64_t rows, int64_t contract,
                                           int64_t ld, int tid, int64_t k_base,
                                           int64_t block_row) const {
        if constexpr (!kFits) {
            // Thin bus: issue() staged nothing, so the whole round trip stays
            // synchronous here.
            load_crosswise_direct_general<SmemLayout, ElemT, kThreads>(
                tile, operand, rows, contract, ld, tid, k_base, block_row);
            return;
        }
        if (!active) return;
        if (fast) {
            crosswise_perm_span(tile, rg, span, v);
            return;
        }
        crosswise_gather_span(tile, operand, rows, contract, ld, k_base,
                              block_row + rg * kRowsChunk, rg, span);
    }
};

// ---------------------------------------------------------------------------
// k-pair packed staging for the 8-bit crosswise operand.
//
// The canonical-tile carry above has to gather FOUR contract runs into one
// register file before its byte-perm; a packed unit is a 16-bit (row, k-pair)
// cell instead, so the two ADJACENT runs (2j, 2j+1) suffice and one thread
// owns its unit outright — no lane pairing, no shfl_xor, one PRMT per output
// word. The tile it stages is the transposed [kK/2][rows] grid the 16-bit
// crosswise path already reads: ldmatrix.trans turns 16 packed rows (one mma
// k-segment = 32 contract bytes) into m16n8k32 fragments, so that reader's
// lane offsets and per-segment steps carry over unchanged.
//
// Per 32B unit: 2 LDG.128 + 8 PRMT + 2 STS.128 — against the canonical
// carry's 4 LDG + 48 PRMT + 16 STS + 16 shfl per 64B staged.
//
// The element-granular sibling covers the row tail, a misaligned base and a
// bus thinner than the unit count: any geometry, at element resolution, so
// the fast carry itself stays predication-free.
template <typename SmemLayout, typename ElemT, int kThreads>
__device__ __forceinline__ void
pack_crosswise_general(Tensor<PtrEngine<ElemT>, SmemLayout> tile,
                       const ElemT* __restrict__ operand, int64_t rows,
                       int64_t contract, int64_t ld, int tid, int64_t k_base,
                       int64_t block_row) {
    static_assert(sizeof(ElemT) == 1, "the packed grid stages 8-bit elements");
    constexpr int kRowsTile = SmemLayout::kChunks * 8;  // 8 units per 16B chunk
    const int units = SmemLayout::kRows * kRowsTile;
    for (int u = tid; u < units; u += kThreads) {
        const int j = u / kRowsTile;
        const int row = u % kRowsTile;
        const int64_t r = block_row + row;
        const bool row_ok = r < rows;
        const int64_t k0 = k_base + 2 * (int64_t)j;
        // One unit = the two contract elements (k0, k0+1) of one row; a
        // contract tail or row tail contributes zero bytes. The bytes are
        // RAW (fp8 has no integer conversion — a cast through float rounds
        // the bit pattern away).
        const unsigned lo = row_ok && k0 < contract
                                ? raw_byte(operand[k0 * ld + r])
                                : 0u;
        const unsigned hi = row_ok && k0 + 1 < contract
                                ? raw_byte(operand[(k0 + 1) * ld + r])
                                : 0u;
        *reinterpret_cast<unsigned short*>(tile(j, row * 2)) =
            (unsigned short)(lo | (hi << 8));
    }
}

template <typename SmemLayout, typename ElemT, int kThreads, bool kOn>
struct PairPackCarry;

template <typename SmemLayout, typename ElemT, int kThreads>
struct PairPackCarry<SmemLayout, ElemT, kThreads, false> {
    __device__ __forceinline__ void issue(const ElemT*, int64_t, int64_t,
                                          int64_t, int, int64_t, int64_t) {}
    template <typename TileT>
    __device__ __forceinline__ void commit(TileT, const ElemT*, int64_t,
                                           int64_t, int64_t, int, int64_t,
                                           int64_t) const {}
};

template <typename SmemLayout, typename ElemT, int kThreads>
struct PairPackCarry<SmemLayout, ElemT, kThreads, true> {
    static_assert(sizeof(ElemT) == 1,
                  "the packed carry stages the 8-bit crosswise path");
    static constexpr int kRowsTile = SmemLayout::kChunks * 8;  // 8 units/chunk
    static constexpr int kGroups = kRowsTile / 16;  // 16-row groups per tile
    static constexpr int kUnits = SmemLayout::kRows * kGroups;  // 32B units
    static constexpr bool kFits = kUnits <= kThreads;

    uint4 v[2];   // this unit's two runs: k = 2j and 2j+1, 16 rows each
    int jrow = 0;  // the packed row this thread owns
    int grp = 0;   // its 16-row group
    bool active = false;
    bool fast = false;

    __device__ __forceinline__ void issue(const ElemT* __restrict__ operand,
                                          int64_t rows, int64_t contract,
                                          int64_t ld, int tid, int64_t k_base,
                                          int64_t block_row) {
        if constexpr (!kFits) return;
        active = tid < kUnits;
        if (!active) return;
        jrow = tid / kGroups;
        grp = tid % kGroups;
        const int64_t r0 = block_row + grp * 16;
        // r0 is a multiple of 16 and p*ld preserves alignment whenever ld has
        // it, so both runs of a unit share one verdict (same rule as the
        // general loader). No shuffle pairs this thread with another, so a
        // boundary unit may simply take the elementwise arm below.
        const bool run_aligned =
            ((reinterpret_cast<uintptr_t>(operand) |
              (ld * (int64_t)sizeof(ElemT))) &
             15) == 0;
        fast = (r0 + 15 < rows) && run_aligned;
        const int64_t k0 = k_base + 2 * (int64_t)jrow;
        if (fast) {
#pragma unroll
            for (int i = 0; i < 2; ++i)
                v[i] = k0 + i < contract
                           ? __ldg(reinterpret_cast<const uint4*>(
                                 operand + (k0 + i) * ld + r0))
                           : make_uint4(0u, 0u, 0u, 0u);
        } else {
            v[0] = make_uint4(0u, 0u, 0u, 0u);
            v[1] = make_uint4(0u, 0u, 0u, 0u);
        }
    }

    // PRMT + STS for the unit issue() fetched. Each output word packs two
    // adjacent rows' k-pair bytes — [a.b(2p), b.b(2p), a.b(2p+1), b.b(2p+1)]
    // out of run words a = 2j and b = 2j+1 — so one byte-perm per word turns
    // the two runs into the packed row's 32 bytes, no exchange needed.
    template <typename TileT>
    __device__ __forceinline__ void commit(TileT tile,
                                           const ElemT* __restrict__ operand,
                                           int64_t rows, int64_t contract,
                                           int64_t ld, int tid, int64_t k_base,
                                           int64_t block_row) const {
        if constexpr (!kFits) {
            // Thin bus: issue() staged nothing, so the whole round trip stays
            // synchronous here (the packed grid, still one thread per unit).
            pack_crosswise_general<SmemLayout, ElemT, kThreads>(
                tile, operand, rows, contract, ld, tid, k_base, block_row);
            return;
        }
        if (!active) return;
        if (fast) {
            const unsigned* a = reinterpret_cast<const unsigned*>(v);
            const unsigned* b = a + 4;
            unsigned w[8];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                w[2 * q] = __byte_perm(a[q], b[q], 0x5140u);
                w[2 * q + 1] = __byte_perm(a[q], b[q], 0x7362u);
            }
            // The 32B unit spans exactly two 16B chunks of the packed row;
            // each is swizzled on its own, so they are two stores.
            *reinterpret_cast<uint4*>(tile(jrow, grp * 32)) =
                make_uint4(w[0], w[1], w[2], w[3]);
            *reinterpret_cast<uint4*>(tile(jrow, grp * 32 + 16)) =
                make_uint4(w[4], w[5], w[6], w[7]);
            return;
        }
        // Boundary unit (row tail / misaligned base): elementwise pack.
        const int64_t r0 = block_row + grp * 16;
        const int64_t k0 = k_base + 2 * (int64_t)jrow;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int64_t r = r0 + i;
            const bool row_ok = r < rows;
            const unsigned lo = (row_ok && k0 < contract)
                                    ? raw_byte(operand[k0 * ld + r])
                                    : 0u;
            const unsigned hi = (row_ok && k0 + 1 < contract)
                                    ? raw_byte(operand[(k0 + 1) * ld + r])
                                    : 0u;
            *reinterpret_cast<unsigned short*>(tile(jrow, grp * 32 + i * 2)) =
                (unsigned short)(lo | (hi << 8));
        }
    }
};

// The packed-grid route: the two-phase carry when the bus fits it, the
// elementwise packed grid otherwise.
template <typename SmemLayout, typename ElemT, int kThreads>
__device__ __forceinline__ void
load_crosswise_paired(Tensor<PtrEngine<ElemT>, SmemLayout> tile,
                      const ElemT* __restrict__ operand, int64_t rows,
                      int64_t contract, int64_t ld, int tid, int64_t k_base,
                      int64_t block_row) {
    PairPackCarry<SmemLayout, ElemT, kThreads, true> carry;
    carry.issue(operand, rows, contract, ld, tid, k_base, block_row);
    carry.commit(tile, operand, rows, contract, ld, tid, k_base, block_row);
}

// The 1-byte route the ladders instantiate: the two-phase carry when the bus
// fits it, the general grid-stride loader otherwise.
template <typename SmemLayout, typename ElemT, int kThreads>
__device__ __forceinline__ void
load_crosswise_direct(Tensor<PtrEngine<ElemT>, SmemLayout> tile,
                      const ElemT* __restrict__ operand, int64_t rows,
                      int64_t contract, int64_t ld, int tid, int64_t k_base,
                      int64_t block_row) {
    if constexpr (sizeof(ElemT) == 1 &&
                  CrosswiseCarry<SmemLayout, ElemT, kThreads, true>::kFits) {
        CrosswiseCarry<SmemLayout, ElemT, kThreads, true> carry;
        carry.issue(operand, rows, contract, ld, tid, k_base, block_row);
        carry.commit(tile, operand, rows, contract, ld, tid, k_base,
                     block_row);
    } else {
        load_crosswise_direct_general<SmemLayout, ElemT, kThreads>(
            tile, operand, rows, contract, ld, tid, k_base, block_row);
    }
}

}  // namespace gemm
}  // namespace astrai
