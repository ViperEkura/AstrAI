#pragma once
#include "gqa_common.cuh"
#include "gqa_mma_utils.cuh"

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
// thread group. Templated on <HEAD_DIM, WARPS, BC, MIN_BLOCKS> with BC a
// multiple of 16.
//
// Occupancy: __launch_bounds__ forces the compiler to fit MIN_BLOCKS blocks/SM,
// spilling to local memory as needed. MIN_BLOCKS is tuned per HEAD_DIM to the
// double-buffered smem footprint (2*BC*LD for each of K/V).
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
// (no sQ staging, no prologue barriers); pre-scale Q by attention scale during Q load; packed bf16x2 output stores;
// causal tile skipping (block-level prefetch bound + warp-level compute skip);
// XOR swizzle (swiz_col) → eliminates ldmatrix bank conflicts without LD
// padding (LD=HEAD_DIM).

template <int HEAD_DIM, int WARPS, int BC, int MIN_BLOCKS>
__global__ __launch_bounds__(WARPS * 32, MIN_BLOCKS)
void gqa_prefill_attn_mma_kernel(GQAParams p) {
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
        const bf16* pa = &p.q[q_base + qra * HEAD_DIM + c];
        const bf16* pb = &p.q[q_base + qrb * HEAD_DIM + c];
        Qa[kt][0] = va ? pk2(__bfloat162float(pa[0]) * p.scale,
                             __bfloat162float(pa[1]) * p.scale) : 0u;
        Qa[kt][1] = vb ? pk2(__bfloat162float(pb[0]) * p.scale,
                             __bfloat162float(pb[1]) * p.scale) : 0u;
        Qa[kt][2] = va ? pk2(__bfloat162float(pa[8]) * p.scale,
                             __bfloat162float(pa[9]) * p.scale) : 0u;
        Qa[kt][3] = vb ? pk2(__bfloat162float(pb[8]) * p.scale,
                             __bfloat162float(pb[9]) * p.scale) : 0u;
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

        // S = Q @ K^T  → Sacc[n8][0..3]   (n8: 8 kv cols each)
        float Sacc[NC8][4];
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            Sacc[n8][0] = Sacc[n8][1] = Sacc[n8][2] = Sacc[n8][3] = 0.0f;
            int krow_l = n8 * 8 + (lane & 7);
            int kcol_h = (lane & 8) ? 8 : 0;
#pragma unroll
            for (int kt = 0; kt < KD; kt++) {
                unsigned b[2];
                ldmatrix_x2(b, &bK[krow_l * LD + swiz_col(kt * 16 + kcol_h, krow_l, SWIZ_MASK)]);
                mma16816(Sacc[n8], Qa[kt], b, Sacc[n8]);
            }
        }

        // ---- online softmax (in registers) ----
        // Q is pre-scaled, so Sacc already includes the attention scale.
        int maxc0 = p.is_causal ? min(p.kv_len, qr0 + p.causal_offset + 1)
                                : p.kv_len;
        int maxc1 = p.is_causal ? min(p.kv_len, qr1 + p.causal_offset + 1)
                                : p.kv_len;
        float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            int cc = kv0 + n8 * 8 + 2 * tid4;
            int c1 = cc + 1;
            bool b0 = (cc >= maxc0) || (has_mask && !p.mask[mb + cc]);
            bool b1 = (c1 >= maxc0) || (has_mask && !p.mask[mb + c1]);
            bool b2 = (cc >= maxc1) || (has_mask && !p.mask[mb + cc]);
            bool b3 = (c1 >= maxc1) || (has_mask && !p.mask[mb + c1]);
            float s0 = b0 ? -FLT_MAX : Sacc[n8][0];
            float s1 = b1 ? -FLT_MAX : Sacc[n8][1];
            float s2 = b2 ? -FLT_MAX : Sacc[n8][2];
            float s3 = b3 ? -FLT_MAX : Sacc[n8][3];
            Sacc[n8][0] = s0; Sacc[n8][1] = s1;
            Sacc[n8][2] = s2; Sacc[n8][3] = s3;
            rmax0 = fmaxf(rmax0, fmaxf(s0, s1));
            rmax1 = fmaxf(rmax1, fmaxf(s2, s3));
        }
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

        float nm0 = fmaxf(m0, rmax0), nm1 = fmaxf(m1, rmax1);
        float corr0 = (nm0 == -FLT_MAX) ? 1.0f : __expf(m0 - nm0);
        float corr1 = (nm1 == -FLT_MAX) ? 1.0f : __expf(m1 - nm1);

        float rsum0 = 0.0f, rsum1 = 0.0f;
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            float p0 = (Sacc[n8][0] == -FLT_MAX) ? 0.0f
                                                 : __expf(Sacc[n8][0] - nm0);
            float p1 = (Sacc[n8][1] == -FLT_MAX) ? 0.0f
                                                 : __expf(Sacc[n8][1] - nm0);
            float p2 = (Sacc[n8][2] == -FLT_MAX) ? 0.0f
                                                 : __expf(Sacc[n8][2] - nm1);
            float p3 = (Sacc[n8][3] == -FLT_MAX) ? 0.0f
                                                 : __expf(Sacc[n8][3] - nm1);
            Sacc[n8][0] = p0; Sacc[n8][1] = p1;
            Sacc[n8][2] = p2; Sacc[n8][3] = p3;
            rsum0 += p0 + p1;
            rsum1 += p2 + p3;
        }
        rsum0 += __shfl_xor_sync(0xFFFFFFFF, rsum0, 1);
        rsum0 += __shfl_xor_sync(0xFFFFFFFF, rsum0, 2);
        rsum1 += __shfl_xor_sync(0xFFFFFFFF, rsum1, 1);
        rsum1 += __shfl_xor_sync(0xFFFFFFFF, rsum1, 2);
        l0 = l0 * corr0 + rsum0;
        l1 = l1 * corr1 + rsum1;
        m0 = nm0; m1 = nm1;

        // rescale O accumulator by per-row correction
#pragma unroll
        for (int j = 0; j < DN8; j++) {
            Oacc[j][0] *= corr0; Oacc[j][1] *= corr0;
            Oacc[j][2] *= corr1; Oacc[j][3] *= corr1;
        }

        // O += P @ V
#pragma unroll
        for (int kt2 = 0; kt2 < KT2; kt2++) {
            unsigned Pa[4];
            Pa[0] = pk2(Sacc[kt2 * 2][0], Sacc[kt2 * 2][1]);
            Pa[1] = pk2(Sacc[kt2 * 2][2], Sacc[kt2 * 2][3]);
            Pa[2] = pk2(Sacc[kt2 * 2 + 1][0], Sacc[kt2 * 2 + 1][1]);
            Pa[3] = pk2(Sacc[kt2 * 2 + 1][2], Sacc[kt2 * 2 + 1][3]);
            int vrow_l = kt2 * 16 + (lane & 15);
#pragma unroll
            for (int dn8 = 0; dn8 < DN8; dn8++) {
                unsigned b[2];
                ldmatrix_x2_trans(b, &bV[vrow_l * LD + swiz_col(dn8 * 8, vrow_l, SWIZ_MASK)]);
                mma16816(Oacc[dn8], Pa, b, Oacc[dn8]);
            }
        }
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
