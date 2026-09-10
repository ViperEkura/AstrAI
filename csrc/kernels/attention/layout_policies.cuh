#pragma once
#include <cuda_bf16.h>
#include "common.h"

// ============================================================================
// Attention layout policies keep Q scheduling independent from K/V storage.
// DenseQSchedule / PackedQSchedule map blocks to Q tiles; ContigKV / PagedKV
// resolve logical K/V positions to physical addresses. This lets the shared
// kernels compose Q layout and K/V storage without coupling the two concerns.
//
//   ContigKV:  K/V are dense [batch, kv_head, kv_len, head_dim] tensors.
//              Params fields used: k, v, kv_stride_*, kv_len, q_len,
//              q_b_stride, causal_offset.
//   PagedKV:   K/V live in a flat pool [size, kv_head, head_dim] indexed via
//              req_to_token.  Params fields used: k_cache, v_cache,
//              req_to_token, req_pool_indices, kv_indptr, qo_indptr,
//              max_context_len, q_l_stride.
//
// Addressing state that is constant across a whole kernel invocation for one
// (batch, kv_head) pair is captured once by make_ctx<HEAD_DIM>() and passed
// to kv_addr, so the load loops never redo the hoistable base computation
// (e.g. the req_pool_indices global read) element-by-element.
// ============================================================================

#define HOST_FORCEINLINE static __host__ __forceinline__
#define DEVICE_FORCEINLINE static __device__ __forceinline__
#define HOST_DEV_FORCEINLINE static __host__ __device__ __forceinline__

namespace astrai {
namespace attention {

using bf16 = __nv_bfloat16;

// ============================================================================
// Q scheduling policies
//
// Map CUDA blocks to request-local Q tiles independently of K/V storage.
// Dense tensors encode the request in blockIdx.z; packed ragged tensors use
// a compact precomputed work map indexed by blockIdx.x.
// ============================================================================

struct DenseQSchedule {
    HOST_FORCEINLINE int host_q_blocks(
        const AttentionParams<bf16>& p, int rows) {
        return (p.q_len + rows - 1) / rows;
    }

    HOST_FORCEINLINE int host_grid_batch(
        const AttentionParams<bf16>& p) {
        return p.batch;
    }

    DEVICE_FORCEINLINE void map_block(
        const AttentionParams<bf16>&, int& batch, int& q_tile) {
        batch = blockIdx.z;
        q_tile = blockIdx.x;
    }

    // GQA-packed prefill mapping: HB q-heads of one kv-head group share a
    // block's K/V stream, each head owning `rows` = BR*WPH consecutive q rows
    // per block.  Dense tensors tile q_len directly, one block per range.
    HOST_FORCEINLINE int packed_grid_x(
        const AttentionParams<bf16>& p, int rows) {
        return (p.q_len + rows - 1) / rows;
    }

    DEVICE_FORCEINLINE void map_packed_block(
        const AttentionParams<bf16>&, int rows, int& batch, int& row_base) {
        batch = blockIdx.z;
        row_base = blockIdx.x * rows;
    }

    DEVICE_FORCEINLINE int q_len(
        const AttentionParams<bf16>& p, int) {
        return p.q_len;
    }

    DEVICE_FORCEINLINE int q_base(
        const AttentionParams<bf16>& p, int batch, int q_head) {
        return batch * p.q_b_stride + q_head * p.q_h_stride;
    }
};

struct PackedQSchedule {
    HOST_FORCEINLINE int host_q_blocks(
        const AttentionParams<bf16>& p, int) {
        return p.num_q_tiles;
    }

    HOST_FORCEINLINE int host_grid_batch(
        const AttentionParams<bf16>&) {
        return 1;
    }

    DEVICE_FORCEINLINE void map_block(
        const AttentionParams<bf16>& p, int& batch, int& q_tile) {
        batch = p.q_tile_to_batch[blockIdx.x];
        q_tile = p.q_tile_to_index[blockIdx.x];
    }

    // GQA-packed prefill mapping: the host tile maps are built in
    // HOST_Q_TILE_ROWS granularity, so each host tile splits into
    // HOST_Q_TILE_ROWS / rows packed blocks along blockIdx.x.
    HOST_FORCEINLINE int packed_grid_x(
        const AttentionParams<bf16>& p, int rows) {
        return p.num_q_tiles * (HOST_Q_TILE_ROWS / rows);
    }

    DEVICE_FORCEINLINE void map_packed_block(
        const AttentionParams<bf16>& p, int rows, int& batch, int& row_base) {
        const int hb = HOST_Q_TILE_ROWS / rows;
        const int host_tile = blockIdx.x / hb;
        batch = p.q_tile_to_batch[host_tile];
        row_base = p.q_tile_to_index[host_tile] * HOST_Q_TILE_ROWS
            + (blockIdx.x - host_tile * hb) * rows;
    }

    DEVICE_FORCEINLINE int q_len(
        const AttentionParams<bf16>& p, int batch) {
        return p.qo_indptr[batch + 1] - p.qo_indptr[batch];
    }

    DEVICE_FORCEINLINE int q_base(
        const AttentionParams<bf16>& p, int batch, int q_head) {
        return p.qo_indptr[batch] * p.q_l_stride + q_head * p.q_h_stride;
    }
};

// Hoisted per-(batch, kv_head) addressing context.
struct KVContext {
    int kv_base;          // contig: batch*kv_b_stride + kv_head*kv_h_stride
    int req_idx;          // paged: req_pool_indices[batch]
    int64_t rtt_stride;   // paged: max_context_len
    int64_t pool_stride;  // paged: kv_head * HEAD_DIM
    int64_t head_off;     // paged: kv_head * HEAD_DIM
};

// Per-element K/V global addresses for one (kc, d) position of a K/V tile.
// The pointers are ALWAYS the computed addresses (never nullptr) — callers
// gate on `valid` (cp.async src_size=0, or a guarded scalar deref).  `valid`
// starts as "within the request's seq_len"; the paged policy further degrades
// it when req_to_token maps the position to a negative slot (empty padding).
// This matches the original hand-rolled load loops, where the address was
// always formed and the predicate decided whether anything was read.
struct KVAddr {
    const void* k;
    const void* v;
    bool valid;
};

// ---- Contiguous K/V ----
struct ContigKV {
    static constexpr bool kPaged = false;

    HOST_FORCEINLINE int host_kv_len(const AttentionParams<bf16>& p) {
        return p.kv_len;
    }

    // decode: same offset (q_len == 1, so there is no row stride component)
    DEVICE_FORCEINLINE int q_decode_base(
        const AttentionParams<bf16>& p, int batch, int q_head) {
        return batch * p.q_b_stride + q_head * p.q_h_stride;
    }

    DEVICE_FORCEINLINE int kv_len(const AttentionParams<bf16>& p, int) {
        return p.kv_len;
    }
    DEVICE_FORCEINLINE int causal_offset(
        const AttentionParams<bf16>& p, int, int) {
        return p.causal_offset;
    }
    // decode: exclusive bound of the single query's attend range
    DEVICE_FORCEINLINE int decode_attend_len(const AttentionParams<bf16>& p, int) {
        return (p.kv_len < p.causal_offset + 1) ? p.kv_len : (p.causal_offset + 1);
    }

    template <int HEAD_DIM>
    DEVICE_FORCEINLINE KVContext make_ctx(
        const AttentionParams<bf16>& p, int batch, int kv_head) {
        KVContext c = {};
        c.kv_base = batch * p.kv_b_stride + kv_head * p.kv_h_stride;
        return c;
    }
    DEVICE_FORCEINLINE int resolve_token(
        const AttentionParams<bf16>& p, const KVContext& c, int kc, bool valid) {
        return valid ? kc : -1;
    }
    DEVICE_FORCEINLINE KVAddr kv_addr_from_token(
        const AttentionParams<bf16>& p, const KVContext& c, int token, int d) {
        const bool valid = token >= 0;
        const int safe_token = valid ? token : 0;
        const int64_t gmem_off = (int64_t)c.kv_base
            + (int64_t)safe_token * p.kv_l_stride
            + (int64_t)d * p.kv_d_stride;
        return {&p.k_ptr[gmem_off], &p.v_ptr[gmem_off], valid};
    }

    template <int VEC>
    DEVICE_FORCEINLINE KVAddr decode_addr(
        const AttentionParams<bf16>& p, const KVContext& c,
        int, int, int kc, int d, bool valid, bool) {
        int token = resolve_token(p, c, kc, valid);
        return kv_addr_from_token(p, c, token, d);
    }
};

// ---- Paged (SGLang-style flat pool) K/V ----
struct PagedKV {
    static constexpr bool kPaged = true;

    HOST_FORCEINLINE int host_kv_len(const AttentionParams<bf16>& p) {
        return p.max_context_len;
    }

    // decode: Q is [batch, q_head, head_dim], so batch is the outer row
    DEVICE_FORCEINLINE int q_decode_base(
        const AttentionParams<bf16>& p, int batch, int q_head) {
        return batch * p.q_l_stride + q_head * p.q_h_stride;
    }

    DEVICE_FORCEINLINE int kv_len(const AttentionParams<bf16>& p, int batch) {
        return p.kv_indptr[batch + 1] - p.kv_indptr[batch];
    }
    DEVICE_FORCEINLINE int causal_offset(
        const AttentionParams<bf16>& p, int batch, int q_len) {
        return kv_len(p, batch) - q_len;
    }
    // decode: the query is the last token, so [0, seq_len) IS its causal range
    DEVICE_FORCEINLINE int decode_attend_len(const AttentionParams<bf16>& p, int batch) {
        return kv_len(p, batch);
    }

    template <int HEAD_DIM>
    DEVICE_FORCEINLINE KVContext make_ctx(
        const AttentionParams<bf16>& p, int batch, int kv_head) {
        KVContext c = {};
        c.req_idx = p.req_pool_indices[batch];
        c.rtt_stride = (int64_t)p.max_context_len;
        c.pool_stride = (int64_t)p.kv_head * HEAD_DIM;
        c.head_off = (int64_t)kv_head * HEAD_DIM;
        return c;
    }
    DEVICE_FORCEINLINE int resolve_token(
        const AttentionParams<bf16>& p, const KVContext& c, int kc, bool valid) {
        return valid ? p.req_to_token[c.req_idx * c.rtt_stride + kc] : -1;
    }
    DEVICE_FORCEINLINE KVAddr kv_addr_from_token(
        const AttentionParams<bf16>& p, const KVContext& c, int slot, int d) {
        const bool valid = slot >= 0;
        const int safe_slot = valid ? slot : 0;
        const int64_t gmem_off = (int64_t)safe_slot * c.pool_stride + c.head_off + d;
        return {&p.k_ptr[gmem_off], &p.v_ptr[gmem_off], valid};
    }

    DEVICE_FORCEINLINE KVAddr new_kv_addr(
        const AttentionParams<bf16>& p, int batch, int kv_head, int d) {
        const int64_t off = (int64_t)batch * p.new_kv_b_stride
            + (int64_t)kv_head * p.new_kv_h_stride + d;
        return {&p.new_k_ptr[off], &p.new_v_ptr[off], true};
    }

    DEVICE_FORCEINLINE void store_new_kv(
        const AttentionParams<bf16>& p, const KVContext& c,
        int seq_len, int d, const KVAddr& src) {
        int slot = resolve_token(p, c, seq_len - 1, true);
        const int64_t off = (int64_t)slot * c.pool_stride + c.head_off + d;
        const_cast<bf16*>(p.k_ptr)[off] = *reinterpret_cast<const bf16*>(src.k);
        const_cast<bf16*>(p.v_ptr)[off] = *reinterpret_cast<const bf16*>(src.v);
    }

    template <int VEC>
    DEVICE_FORCEINLINE KVAddr decode_addr(
        const AttentionParams<bf16>& p, const KVContext& c,
        int batch, int kv_head, int kc, int d, bool valid, bool persist) {
        if (p.new_k_ptr && valid && kc == kv_len(p, batch) - 1) {
            KVAddr src = new_kv_addr(p, batch, kv_head, d);
            if (persist) {
                #pragma unroll
                for (int j = 0; j < VEC; j++) {
                    KVAddr value = new_kv_addr(p, batch, kv_head, d + j);
                    store_new_kv(p, c, kc + 1, d + j, value);
                }
            }
            return src;
        }
        int token = resolve_token(p, c, kc, valid);
        return kv_addr_from_token(p, c, token, d);
    }
};

// Scalar K/V tile fill shared by the non-MMA kernels: copies one chunk into
// linear (unswizzled) shared memory, zero-filling invalid slots.  AddrFn
// maps (kc, d) -> KVAddr; tid/stride carry the caller's thread mapping.
template <typename AddrFn>
DEVICE_FORCEINLINE void fill_kv_smem(
    bf16* k_smem, bf16* v_smem, int elems, int head_dim,
    int kv_base, int tid, int stride, const AddrFn& addr)
{
    for (int i = tid; i < elems; i += stride) {
        int s = i / head_dim;
        int d = i % head_dim;
        KVAddr a = addr(kv_base + s, d);
        k_smem[i] = a.valid ? *reinterpret_cast<const bf16*>(a.k) : (bf16)0.f;
        v_smem[i] = a.valid ? *reinterpret_cast<const bf16*>(a.v) : (bf16)0.f;
    }
}

}  // namespace attention
}  // namespace astrai
