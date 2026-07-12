#pragma once
#include <cfloat>
#include <cuda_bf16.h>
#include "attn_common.h"
#include "attn_mma_utils.cuh"

using bf16 = __nv_bfloat16;

// Tensor-core prefill flash attention (raw mma.sync PTX).
// One warp owns BR=16 query rows. S = Q@K^T and O = P@V run on bf16 tensor
// cores via mma.sync.m16n8k16 (f32 accumulate). Q fragments are loaded once
// straight from global into the mma A-operand layout (no smem staging) and
// kept resident in registers across the tile loop. S, O, and the online-softmax
// stats (m, l) also live in registers.
// Shared memory is statically sized via template parameters — no dynamic
// allocation. The mma fragment layout is used directly: the S accumulator
// (f32) maps element-for-element onto the P matrix_a (bf16) operand, so
// softmax needs no shuffle repack; row reductions fold across the 4-lane
// thread group. Templated on <HEAD_DIM, WARPS, BC> with BC a multiple of 16.
//
// Software pipeline: K/V are double-buffered and loaded via cp.async one tile
// ahead, so the next tile streams from global memory while the current tile's
// tensor-core math runs — hiding load latency (long_scoreboard). A single
// __syncthreads per tile both publishes the freshly loaded tile cross-warp and
// (because it runs before the next prefetch) guards the buffer being refilled,
// so no second barrier is needed. Predicated cp.async (cp_async_16_pred)
// zero-fills rows past kv_len, unifying full and partial tiles on one path.
// BC=32 (D<=128) amortizes the per-tile wait+barrier+loop overhead over more
// tensor-core work — this kernel is latency-bound (low occupancy from high
// register pressure), so fewer, larger tiles beat many tiny ones.
//
// Optimizations: load Q fragments directly from global in mma A-operand layout
// (no sQ staging, no prologue barriers); post-multiply scale in float after
// S=Q@K^T to avoid bf16 precision loss; packed bf16x2 output stores;
// causal tile skipping (block-level prefetch bound + warp-level compute skip);
// XOR swizzle (swiz_col) → eliminates ldmatrix bank conflicts without LD
// padding (LD=HEAD_DIM).

template <int HEAD_DIM, int WARPS, int BC>
__global__ void attn_prefill_split_q_mma_kernel(AttentionParams<bf16> p) {
    constexpr int BR = 16;
    constexpr int KD = HEAD_DIM / 16;  // Q/K k-tiles
    constexpr int NC8 = BC / 8;        // S n-tiles (N=8 each)
    constexpr int KT2 = BC / 16;       // P k-tiles (K=16 each)
    constexpr int DN8 = HEAD_DIM / 8;  // O n-tiles (N=8 each)
    constexpr int LD = HEAD_DIM;   // XOR swizzle (swiz_col) handles bank conflicts
    constexpr int SWIZ_MASK = (HEAD_DIM >= 64) ? 7 : (HEAD_DIM / 8 - 1);  // chunk bits, stay within LD

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gid = lane >> 2;   // 0..7  → rows gid, gid+8
    const int tid4 = lane & 3;   // 0..3
    const int nthreads = WARPS * 32;

    const int q_head = blockIdx.y;
    const int batch = blockIdx.z;
    const int kv_head = q_head / (p.q_head / p.kv_head);
    const int qrow0 = (blockIdx.x * WARPS + warp) * BR;

    // Static shared memory — sized by template parameters at compile time.
    // K/V are double-buffered (STAGES=2): the next tile's cp.async load runs
    // while the current tile's tensor-core math executes, hiding global-load
    // latency (FA2-style software pipeline). No dynamic smem / carveout opt-in.
    constexpr int STAGES = 2;
    __shared__ __align__(16) bf16 sK[STAGES * BC * LD];
    __shared__ __align__(16) bf16 sV[STAGES * BC * LD];

    // Load the Q fragments straight from global into the mma A-operand layout
    // (m16n8k16, row-major): no sQ staging area and no serialized per-warp
    // prologue barriers. Each lane reads exactly the 8 Q elements ldmatrix
    // would have produced, pre-scaled by the attention scale. Kept resident in
    // registers across the tile loop.
    //   frag[0]/[2]: row = qrow0 + gid ;  frag[1]/[3]: row = qrow0 + gid + 8
    //   frag[0]/[1]: cols kt*16 + tid4*2 + {0,1} ; frag[2]/[3]: + 8
    const int q_base = ((batch * p.q_head + q_head) * p.q_len) * HEAD_DIM;
    const int qra = qrow0 + gid;
    const int qrb = qrow0 + gid + 8;
    const bool va = qra < p.q_len, vb = qrb < p.q_len;
    unsigned Qa[KD][4];
#pragma unroll
    for (int kt = 0; kt < KD; kt++) {
        int c = kt * 16 + tid4 * 2;
        const unsigned* pau = reinterpret_cast<const unsigned*>(
            &p.q[q_base + qra * HEAD_DIM + c]);
        const unsigned* pbu = reinterpret_cast<const unsigned*>(
            &p.q[q_base + qrb * HEAD_DIM + c]);
        Qa[kt][0] = va ? pau[0] : 0u;
        Qa[kt][1] = vb ? pbu[0] : 0u;
        Qa[kt][2] = va ? pau[4] : 0u;
        Qa[kt][3] = vb ? pbu[4] : 0u;
    }

    float Oacc[DN8][4];
#pragma unroll
    for (int j = 0; j < DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    const int kv_base = ((batch * p.kv_head + kv_head) * p.kv_len) * HEAD_DIM;
    const int tiles = (p.kv_len + BC - 1) / BC;
    const int qr0 = qrow0 + gid;      // row for c0/c1
    const int qr1 = qrow0 + gid + 8;  // row for c2/c3

    // Causal tile-skip bounds (no-op when is_causal == 0)
    const int use_skip = p.is_causal;
    const int max_kv = qrow0 + BR - 1 + p.causal_offset;
    const int block_max_kv =
        blockIdx.x * WARPS * BR + WARPS * BR - 1 + p.causal_offset;
    const int has_mask = p.use_mask && p.mask;
    const int mb = batch * p.kv_len;

    // Last active tile: block-level causal bound (all warps in the block share
    // the K/V load, so the prefetch range is the block max, not per-warp).
    int t_end = tiles - 1;
    if (use_skip) {
        int bt = block_max_kv / BC;
        if (bt < t_end) t_end = bt;
    }

    constexpr int VEC = 8;  // bf16 per cp.async unit (16 bytes)
    constexpr int TOTAL = BC * HEAD_DIM;

    // Issue cp.async loads for tile `ti` into shared buffer `buf`. Predicated
    // loads zero-fill rows past kv_len, so partial tiles need no scalar path.
    auto load_tile = [&](int ti, int buf) {
        int kv0 = ti * BC;
        bf16* dK = sK + buf * BC * LD;
        bf16* dV = sV + buf * BC * LD;
#pragma unroll
        for (int i = threadIdx.x * VEC; i < TOTAL; i += nthreads * VEC) {
            int r = i / HEAD_DIM, d = i % HEAD_DIM;
            int kc = kv0 + r;
            bool valid = kc < p.kv_len;
            int off = r * LD + swiz_col(d, r, SWIZ_MASK);
            cp_async_16_pred(&dK[off], &p.k[kv_base + kc * HEAD_DIM + d], valid);
            cp_async_16_pred(&dV[off], &p.v[kv_base + kc * HEAD_DIM + d], valid);
        }
        cp_async_commit();
    };

    // Prologue: kick off the first tile's load.
    load_tile(0, 0);

    for (int ti = 0; ti <= t_end; ti++) {
        int buf = ti & 1;

        // Wait for the current tile's async copies, then a single barrier: it
        // both publishes this tile's data cross-warp AND guarantees the prior
        // compute on the buffer we are about to refill has finished. Issuing
        // the next tile's load *after* this barrier lets one barrier cover both
        // hazards (vs two), while the load still overlaps this tile's math.
        cp_async_wait_group<0>();
        __syncthreads();
        if (ti < t_end) load_tile(ti + 1, (ti + 1) & 1);

        const bf16* bK = sK + buf * BC * LD;
        const bf16* bV = sV + buf * BC * LD;
        int kv0 = ti * BC;

        // Warp-level causal skip
        if (!use_skip || kv0 <= max_kv) {

        // S = Q @ K^T + scale + online softmax + O += P @ V
        float Sacc[NC8][4];
        mma_compute_scores<KD, NC8>(Qa, bK, LD, SWIZ_MASK, lane, Sacc);

        // post-multiply scale in float (no bf16 precision loss from pre-scaling Q)
        #pragma unroll
        for (int n8 = 0; n8 < NC8; n8++)
            Sacc[n8][0] *= p.scale, Sacc[n8][1] *= p.scale,
            Sacc[n8][2] *= p.scale, Sacc[n8][3] *= p.scale;

        int maxc0 = p.is_causal ? min(p.kv_len, qr0 + p.causal_offset + 1)
                                : p.kv_len;
        int maxc1 = p.is_causal ? min(p.kv_len, qr1 + p.causal_offset + 1)
                                : p.kv_len;
        mma_softmax_tile<NC8, DN8>(kv0, maxc0, maxc1,
                                    mb, p.mask, has_mask,
                                    Sacc, Oacc, m0, m1, l0, l1, lane);

        mma_pv_accumulate<DN8, KT2>(Sacc, bV, LD, SWIZ_MASK, lane, Oacc);
        }  // if active (warp-level causal skip)
    }

    // ---- write output ---- (packed bf16x2 stores: one 32-bit STG per pair,
    // halves store count and removes the uncoalesced scalar-store penalty)
    float rl0 = (l0 > 1e-20f) ? (1.0f / l0) : 0.0f;
    float rl1 = (l1 > 1e-20f) ? (1.0f / l1) : 0.0f;
    const int o_base = ((batch * p.q_head + q_head) * p.q_len) * HEAD_DIM;
#pragma unroll
    for (int dn8 = 0; dn8 < DN8; dn8++) {
        int d = dn8 * 8 + 2 * tid4;
        if (qr0 < p.q_len) {
            __nv_bfloat162 v = __floats2bfloat162_rn(Oacc[dn8][0] * rl0,
                                                     Oacc[dn8][1] * rl0);
            *reinterpret_cast<__nv_bfloat162*>(&p.o[o_base + qr0 * HEAD_DIM + d]) = v;
        }
        if (qr1 < p.q_len) {
            __nv_bfloat162 v = __floats2bfloat162_rn(Oacc[dn8][2] * rl1,
                                                     Oacc[dn8][3] * rl1);
            *reinterpret_cast<__nv_bfloat162*>(&p.o[o_base + qr1 * HEAD_DIM + d]) = v;
        }
    }
}
