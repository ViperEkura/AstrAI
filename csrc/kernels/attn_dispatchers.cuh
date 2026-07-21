#pragma once
// Shared attention dispatchers — used by both production .cu and test .cu.
// No torch dependency; pure CUDA.

#include <cuda_runtime.h>
#include <algorithm>
#include "attn_prefill_split_q.cuh"
#include "attn_decode_split_kv.cuh"
#include "attn_paged_decode_split_kv.cuh"
#ifndef ASTRAI_NO_MMA
#include "attn_prefill_split_q_mma.cuh"
#include "attn_decode_split_kv_mma.cuh"
#include "attn_paged_decode_split_kv_mma.cuh"
#endif

// Split-KV: compute number of splits to fill all SMs for small-batch decode.
inline int compute_num_splits(int base_blocks, int tiles_total) {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int n = (2 * sm_count + base_blocks - 1) / base_blocks;
    return std::max(1, std::min(n, std::min(tiles_total, 32)));
}

// ======================================================================
// Prefill
// ======================================================================

#ifndef ASTRAI_NO_MMA
template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_prefill_mma(AttentionParams<bf16>& p) {
    constexpr int WARPS = 4;
    constexpr int BC = (HEAD_DIM <= 128) ? 32 : 16;
    using Traits = KernelTraits<HEAD_DIM, BC, WARPS, 2>;
    dim3 grid((p.q_len + Traits::BR * WARPS - 1) / (Traits::BR * WARPS), p.q_head, p.batch);
    dim3 block(Traits::NUM_THREADS);
    attn_prefill_split_q_mma_kernel<Traits, IsCausal, HasMask><<<grid, block>>>(p);
}
#endif

template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_prefill_scalar(AttentionParams<bf16>& p) {
    constexpr int G = 8, ROWS = 32, P_BC = 32;
    dim3 grid((p.q_len + ROWS - 1) / ROWS, p.q_head, p.batch);
    dim3 block(G, ROWS);
    attn_prefill_split_q_kernel_t<HEAD_DIM, G, ROWS, P_BC, IsCausal, HasMask><<<grid, block>>>(p);
}

template <int HEAD_DIM>
static inline void dispatch_prefill(AttentionParams<bf16>& p) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);

#ifndef ASTRAI_NO_MMA
    if (is_causal) {
        if (has_mask)      launch_prefill_mma<HEAD_DIM, true, true>(p);
        else               launch_prefill_mma<HEAD_DIM, true, false>(p);
    } else {
        if (has_mask)      launch_prefill_mma<HEAD_DIM, false, true>(p);
        else               launch_prefill_mma<HEAD_DIM, false, false>(p);
    }
#else
    if (is_causal) {
        if (has_mask)      launch_prefill_scalar<HEAD_DIM, true, true>(p);
        else               launch_prefill_scalar<HEAD_DIM, true, false>(p);
    } else {
        if (has_mask)      launch_prefill_scalar<HEAD_DIM, false, true>(p);
        else               launch_prefill_scalar<HEAD_DIM, false, false>(p);
    }
#endif
}

// ======================================================================
// Decode
// ======================================================================

#ifndef ASTRAI_NO_MMA
template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_decode_mma(AttentionParams<bf16>& p, int group_size) {
    int G = p.q_head / p.kv_head;
    if (G >= 1 && G <= 16) {
        int tiles_total = (p.kv_len + 32 - 1) / 32;
        p.num_splits = compute_num_splits(p.batch * p.kv_head, tiles_total);
        constexpr int STAGES = (HEAD_DIM <= 128) ? 2 : 1;
        using Traits = KernelTraits<HEAD_DIM, 32, 1, STAGES>;
        dim3 grid(p.kv_head, p.batch, p.num_splits);
        attn_decode_split_kv_mma_kernel<Traits, IsCausal, HasMask><<<grid, 32>>>(p);
    } else {
        int chunks_total = (p.kv_len + DC_CHUNK - 1) / DC_CHUNK;
        p.num_splits = compute_num_splits(p.batch * p.kv_head, chunks_total);
        size_t smem = DC_CHUNK * p.head_dim * sizeof(bf16);
        dim3 grid(p.batch * p.kv_head, 1, p.num_splits);
        dim3 block(32, group_size);
        attn_decode_split_kv_kernel<HEAD_DIM, IsCausal, HasMask><<<grid, block, smem>>>(p);
    }
}
#endif

template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_decode_scalar(AttentionParams<bf16>& p, int group_size) {
    int chunks_total = (p.kv_len + DC_CHUNK - 1) / DC_CHUNK;
    p.num_splits = compute_num_splits(p.batch * p.kv_head, chunks_total);
    size_t smem = DC_CHUNK * p.head_dim * sizeof(bf16);
    dim3 grid(p.batch * p.kv_head, 1, p.num_splits);
    dim3 block(32, group_size);
    attn_decode_split_kv_kernel<HEAD_DIM, IsCausal, HasMask><<<grid, block, smem>>>(p);
}

template <int HEAD_DIM>
static inline void dispatch_decode(AttentionParams<bf16>& p) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);
    int group_size = p.q_head / p.kv_head;

#ifndef ASTRAI_NO_MMA
    if (is_causal) {
        if (has_mask)      launch_decode_mma<HEAD_DIM, true, true>(p, group_size);
        else               launch_decode_mma<HEAD_DIM, true, false>(p, group_size);
    } else {
        if (has_mask)      launch_decode_mma<HEAD_DIM, false, true>(p, group_size);
        else               launch_decode_mma<HEAD_DIM, false, false>(p, group_size);
    }
#else
    if (is_causal) {
        if (has_mask)      launch_decode_scalar<HEAD_DIM, true, true>(p, group_size);
        else               launch_decode_scalar<HEAD_DIM, true, false>(p, group_size);
    } else {
        if (has_mask)      launch_decode_scalar<HEAD_DIM, false, true>(p, group_size);
        else               launch_decode_scalar<HEAD_DIM, false, false>(p, group_size);
    }
#endif

    attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}

// ======================================================================
// Paged Decode
// ======================================================================

#ifndef ASTRAI_NO_MMA
template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_paged_decode_mma(PagedAttentionParams<bf16>& p, int group_size) {
    int G = p.q_head / p.kv_head;
    if (G >= 1 && G <= 16 && p.page_size >= 32) {
        int tiles_total = (p.kv_len + 32 - 1) / 32;
        p.num_splits = compute_num_splits(p.batch * p.kv_head, tiles_total);
        constexpr int STAGES = (HEAD_DIM <= 128) ? 2 : 1;
        using Traits = KernelTraits<HEAD_DIM, 32, 1, STAGES>;
        dim3 grid(p.kv_head, p.batch, p.num_splits);
        paged_attn_decode_split_kv_mma_kernel<Traits, IsCausal, HasMask><<<grid, 32>>>(p);
    } else {
        int chunks_total = (p.kv_len + PDC_CHUNK - 1) / PDC_CHUNK;
        p.num_splits = compute_num_splits(p.batch * p.kv_head, chunks_total);
        size_t smem = PDC_CHUNK * p.head_dim * sizeof(bf16);
        dim3 grid(p.batch * p.kv_head, 1, p.num_splits);
        dim3 block(32, group_size);
        paged_attn_decode_split_kv_kernel<HEAD_DIM, IsCausal, HasMask><<<grid, block, smem>>>(p);
    }
}
#endif

template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_paged_decode_scalar(PagedAttentionParams<bf16>& p, int group_size) {
    int chunks_total = (p.kv_len + PDC_CHUNK - 1) / PDC_CHUNK;
    p.num_splits = compute_num_splits(p.batch * p.kv_head, chunks_total);
    size_t smem = PDC_CHUNK * p.head_dim * sizeof(bf16);
    dim3 grid(p.batch * p.kv_head, 1, p.num_splits);
    dim3 block(32, group_size);
    paged_attn_decode_split_kv_kernel<HEAD_DIM, IsCausal, HasMask><<<grid, block, smem>>>(p);
}

template <int HEAD_DIM>
static inline void dispatch_paged_decode(PagedAttentionParams<bf16>& p) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);
    int group_size = p.q_head / p.kv_head;

#ifndef ASTRAI_NO_MMA
    if (is_causal) {
        if (has_mask)      launch_paged_decode_mma<HEAD_DIM, true, true>(p, group_size);
        else               launch_paged_decode_mma<HEAD_DIM, true, false>(p, group_size);
    } else {
        if (has_mask)      launch_paged_decode_mma<HEAD_DIM, false, true>(p, group_size);
        else               launch_paged_decode_mma<HEAD_DIM, false, false>(p, group_size);
    }
#else
    if (is_causal) {
        if (has_mask)      launch_paged_decode_scalar<HEAD_DIM, true, true>(p, group_size);
        else               launch_paged_decode_scalar<HEAD_DIM, true, false>(p, group_size);
    } else {
        if (has_mask)      launch_paged_decode_scalar<HEAD_DIM, false, true>(p, group_size);
        else               launch_paged_decode_scalar<HEAD_DIM, false, false>(p, group_size);
    }
#endif

    paged_attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}
