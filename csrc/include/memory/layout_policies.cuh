#pragma once
#include <cuda_bf16.h>
#include <utils/attention_common.h>
#include <utils/define.cuh>
#include <utils/dtype.cuh>

// ============================================================================
// Attention layout policies keep Q scheduling independent from K/V storage.
// DenseQSchedule / PackedQSchedule map blocks to Q tiles; ContigKV / PagedKV
// resolve logical K/V positions to physical addresses. This lets the shared
// kernels compose Q layout and K/V storage without coupling the two concerns.
//
//   ContigKV<T>:  K/V are dense [batch, kv_head, kv_len, head_dim] tensors.
//              Params fields used: k, v, kv_stride_*, kv_len, q_len,
//              q_b_stride, causal_offset.
//   PagedKV<T>:   K/V live in a flat pool [size, kv_head, head_dim] indexed via
//              req_to_token.  Params fields used: k_cache, v_cache,
//              req_to_token, req_pool_indices, kv_indptr, qo_indptr,
//              max_context_len, q_l_stride.
//
// The K/V policy is also where the element type is bound: AttentionParams
// itself is dtype-agnostic (void* pointers + a dtype tag) and the kernels read
// the element type back off the policy they were instantiated with
// (`using T = typename KV::Elem;`), so the dispatch's dtype axis is exactly
// "which KV instantiation".  Every method takes the params by reference; the
// typed views below (kptr/vptr/...) are the only place void* becomes T*.
//
// Addressing state that is constant across a whole kernel invocation for one
// (batch, kv_head) pair is captured once by make_ctx<HEAD_DIM>() and passed
// to kv_addr, so the load loops never redo the hoistable base computation
// (e.g. the req_pool_indices global read) element-by-element.
// ============================================================================


namespace astrai {
namespace attention {

// ============================================================================
// Q scheduling policies
//
// Map CUDA blocks to request-local Q tiles independently of K/V storage.
// Dense tensors encode the request in blockIdx.z; packed ragged tensors use
// a compact precomputed work map indexed by blockIdx.x.
// ============================================================================

struct DenseQSchedule {
         static HOST_FORCEINLINE int host_q_blocks(
        const AttentionParams& p, int rows) {
        return (p.q_len + rows - 1) / rows;
    }

         static HOST_FORCEINLINE int host_grid_batch(
        const AttentionParams& p) {
        return p.batch;
    }

     static DEVICE_FORCEINLINE void map_block(
        const AttentionParams&, int& batch, int& q_tile) {
        batch = blockIdx.z;
        q_tile = blockIdx.x;
    }

    // GQA-packed prefill mapping: HB q-heads of one kv-head group share a
    // block's K/V stream, each head owning `rows` = BR*WPH consecutive q rows
    // per block.  Dense tensors tile q_len directly, one block per range.
         static HOST_FORCEINLINE int packed_grid_x(
        const AttentionParams& p, int rows) {
        return (p.q_len + rows - 1) / rows;
    }

     static DEVICE_FORCEINLINE void map_packed_block(
        const AttentionParams&, int rows, int& batch, int& row_base) {
        batch = blockIdx.z;
        row_base = blockIdx.x * rows;
    }

     static DEVICE_FORCEINLINE int q_len(
        const AttentionParams& p, int) {
        return p.q_len;
    }

     static DEVICE_FORCEINLINE int q_base(
        const AttentionParams& p, int batch, int q_head) {
        return batch * p.q_b_stride + q_head * p.q_h_stride;
    }
};

struct PackedQSchedule {
         static HOST_FORCEINLINE int host_q_blocks(
        const AttentionParams& p, int) {
        return p.num_q_tiles;
    }

         static HOST_FORCEINLINE int host_grid_batch(
        const AttentionParams&) {
        return 1;
    }

     static DEVICE_FORCEINLINE void map_block(
        const AttentionParams& p, int& batch, int& q_tile) {
        batch = p.q_tile_to_batch[blockIdx.x];
        q_tile = p.q_tile_to_index[blockIdx.x];
    }

    // GQA-packed prefill mapping: the host tile maps are built in
    // HOST_Q_TILE_ROWS granularity, so each host tile splits into
    // HOST_Q_TILE_ROWS / rows packed blocks along blockIdx.x.
         static HOST_FORCEINLINE int packed_grid_x(
        const AttentionParams& p, int rows) {
        return p.num_q_tiles * (HOST_Q_TILE_ROWS / rows);
    }

     static DEVICE_FORCEINLINE void map_packed_block(
        const AttentionParams& p, int rows, int& batch, int& row_base) {
        const int hb = HOST_Q_TILE_ROWS / rows;
        const int host_tile = blockIdx.x / hb;
        batch = p.q_tile_to_batch[host_tile];
        row_base = p.q_tile_to_index[host_tile] * HOST_Q_TILE_ROWS
            + (blockIdx.x - host_tile * hb) * rows;
    }

     static DEVICE_FORCEINLINE int q_len(
        const AttentionParams& p, int batch) {
        return p.qo_indptr[batch + 1] - p.qo_indptr[batch];
    }

     static DEVICE_FORCEINLINE int q_base(
        const AttentionParams& p, int batch, int q_head) {
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
template <typename T>
struct ContigKV {
    using Elem = T;
    static constexpr bool kPaged = false;

    // Typed views of the params' dtype-agnostic pointers. The restrict
    // locals at the deref sites re-state the alias promise the void* -> T*
    // cast drops.
     static DEVICE_FORCEINLINE const T* kptr(const AttentionParams& p) {
        return static_cast<const T*>(p.k_ptr);
    }
     static DEVICE_FORCEINLINE const T* vptr(const AttentionParams& p) {
        return static_cast<const T*>(p.v_ptr);
    }

         static HOST_FORCEINLINE int host_kv_len(const AttentionParams& p) {
        return p.kv_len;
    }

    // decode: same offset (q_len == 1, so there is no row stride component)
     static DEVICE_FORCEINLINE int q_decode_base(
        const AttentionParams& p, int batch, int q_head) {
        return batch * p.q_b_stride + q_head * p.q_h_stride;
    }

     static DEVICE_FORCEINLINE int kv_len(const AttentionParams& p, int) {
        return p.kv_len;
    }
     static DEVICE_FORCEINLINE int causal_offset(
        const AttentionParams& p, int, int) {
        return p.causal_offset;
    }
    // decode: exclusive bound of the single query's attend range
     static DEVICE_FORCEINLINE int decode_attend_len(const AttentionParams& p, int) {
        return (p.kv_len < p.causal_offset + 1) ? p.kv_len : (p.causal_offset + 1);
    }

    template <int HEAD_DIM>
     static DEVICE_FORCEINLINE KVContext make_ctx(
        const AttentionParams& p, int batch, int kv_head) {
        KVContext c = {};
        c.kv_base = batch * p.kv_b_stride + kv_head * p.kv_h_stride;
        return c;
    }
     static DEVICE_FORCEINLINE int resolve_token(
        const AttentionParams& p, const KVContext& c, int kc, bool valid) {
        return valid ? kc : -1;
    }
     static DEVICE_FORCEINLINE KVAddr kv_addr_from_token(
        const AttentionParams& p, const KVContext& c, int token, int d) {
        const bool valid = token >= 0;
        const int safe_token = valid ? token : 0;
        const int64_t gmem_off = (int64_t)c.kv_base
            + (int64_t)safe_token * p.kv_l_stride
            + (int64_t)d * p.kv_d_stride;
        const T* __restrict__ k = kptr(p);
        const T* __restrict__ v = vptr(p);
        return {&k[gmem_off], &v[gmem_off], valid};
    }

    template <int VEC>
     static DEVICE_FORCEINLINE KVAddr decode_addr(
        const AttentionParams& p, const KVContext& c,
        int, int, int kc, int d, bool valid, bool) {
        int token = resolve_token(p, c, kc, valid);
        return kv_addr_from_token(p, c, token, d);
    }
};

// ---- Paged (SGLang-style flat pool) K/V ----
template <typename T>
struct PagedKV {
    using Elem = T;
    static constexpr bool kPaged = true;

     static DEVICE_FORCEINLINE const T* kptr(const AttentionParams& p) {
        return static_cast<const T*>(p.k_ptr);
    }
     static DEVICE_FORCEINLINE const T* vptr(const AttentionParams& p) {
        return static_cast<const T*>(p.v_ptr);
    }
     static DEVICE_FORCEINLINE const T* new_kptr(const AttentionParams& p) {
        return static_cast<const T*>(p.new_k_ptr);
    }
     static DEVICE_FORCEINLINE const T* new_vptr(const AttentionParams& p) {
        return static_cast<const T*>(p.new_v_ptr);
    }

         static HOST_FORCEINLINE int host_kv_len(const AttentionParams& p) {
        return p.max_context_len;
    }

    // decode: Q is [batch, q_head, head_dim], so batch is the outer row
     static DEVICE_FORCEINLINE int q_decode_base(
        const AttentionParams& p, int batch, int q_head) {
        return batch * p.q_l_stride + q_head * p.q_h_stride;
    }

     static DEVICE_FORCEINLINE int kv_len(const AttentionParams& p, int batch) {
        return p.kv_indptr[batch + 1] - p.kv_indptr[batch];
    }
     static DEVICE_FORCEINLINE int causal_offset(
        const AttentionParams& p, int batch, int q_len) {
        return kv_len(p, batch) - q_len;
    }
    // decode: the query is the last token, so [0, seq_len) IS its causal range
     static DEVICE_FORCEINLINE int decode_attend_len(const AttentionParams& p, int batch) {
        return kv_len(p, batch);
    }

    template <int HEAD_DIM>
     static DEVICE_FORCEINLINE KVContext make_ctx(
        const AttentionParams& p, int batch, int kv_head) {
        KVContext c = {};
        c.req_idx = p.req_pool_indices[batch];
        c.rtt_stride = (int64_t)p.max_context_len;
        c.pool_stride = (int64_t)p.kv_head * HEAD_DIM;
        c.head_off = (int64_t)kv_head * HEAD_DIM;
        return c;
    }
     static DEVICE_FORCEINLINE int resolve_token(
        const AttentionParams& p, const KVContext& c, int kc, bool valid) {
        return valid ? p.req_to_token[c.req_idx * c.rtt_stride + kc] : -1;
    }
     static DEVICE_FORCEINLINE KVAddr kv_addr_from_token(
        const AttentionParams& p, const KVContext& c, int slot, int d) {
        const bool valid = slot >= 0;
        const int safe_slot = valid ? slot : 0;
        const int64_t gmem_off = (int64_t)safe_slot * c.pool_stride + c.head_off + d;
        const T* __restrict__ k = kptr(p);
        const T* __restrict__ v = vptr(p);
        return {&k[gmem_off], &v[gmem_off], valid};
    }

     static DEVICE_FORCEINLINE KVAddr new_kv_addr(
        const AttentionParams& p, int batch, int kv_head, int d) {
        const int64_t off = (int64_t)batch * p.new_kv_b_stride
            + (int64_t)kv_head * p.new_kv_h_stride + d;
        const T* __restrict__ nk = new_kptr(p);
        const T* __restrict__ nv = new_vptr(p);
        return {&nk[off], &nv[off], true};
    }

     static DEVICE_FORCEINLINE void store_new_kv(
        const AttentionParams& p, const KVContext& c,
        int seq_len, int d, const KVAddr& src) {
        int slot = resolve_token(p, c, seq_len - 1, true);
        const int64_t off = (int64_t)slot * c.pool_stride + c.head_off + d;
        T* __restrict__ k = const_cast<T*>(kptr(p));
        T* __restrict__ v = const_cast<T*>(vptr(p));
        k[off] = *reinterpret_cast<const T*>(src.k);
        v[off] = *reinterpret_cast<const T*>(src.v);
    }

    template <int VEC>
     static DEVICE_FORCEINLINE KVAddr decode_addr(
        const AttentionParams& p, const KVContext& c,
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

}  // namespace attention
}  // namespace astrai
