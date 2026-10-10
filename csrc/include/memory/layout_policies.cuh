#pragma once
#include <api/attention_common.h>
#include <cuda_bf16.h>
#include <datatype/element.cuh>
#include <memory/pipeline.cuh>
#include <utils/define.cuh>

/*
 * Q scheduling is independent of K/V storage. DenseQSchedule/PackedQSchedule
 * map Q tiles; ContigKV/PagedKV resolve logical K/V positions and bind Elem.
 * make_ctx hoists per-(batch, kv_head) address state outside the load loops.
 */

namespace astrai {
namespace attention {

/* One shared-memory layout is used by both the tile producer and MMA consumer. */
template <int LeadingDim> struct SharedTileLayout {
    static_assert(LeadingDim >= 16 && (LeadingDim & (LeadingDim - 1)) == 0,
                  "shared attention tiles require a power-of-two leading dimension");
    static constexpr int kMask = LeadingDim >= 64 ? 7 : LeadingDim / 8 - 1;
    static DEVICE_FORCEINLINE int column(int d, int row) {
        return (((d >> 3) ^ (row & kMask)) << 3) | (d & 7);
    }
};

template <typename Traits> struct KVTileLoader {
    // Keep the uncommon scalar path out of the unrolled asynchronous loader.
    static __device__ __noinline__ void load_unaligned(typename Traits::Elem* dst_k,
                                                       typename Traits::Elem* dst_v,
                                                       const typename Traits::Elem* src_k,
                                                       const typename Traits::Elem* src_v,
                                                       bool valid) {
#pragma unroll
        for (int j = 0; j < Traits::VEC; ++j) {
            dst_k[j] = valid ? src_k[j] : ElemTrait<typename Traits::Elem>::from_float(0.0f);
            dst_v[j] = valid ? src_v[j] : ElemTrait<typename Traits::Elem>::from_float(0.0f);
        }
    }

    template <typename AddrFn>
    static __device__ inline void
    load(typename Traits::Elem* sK, // ring bases (STAGES * BC * LD each)
         typename Traits::Elem* sV,
         int ti,
         int buf, // tile index, ring slot
         int seq_len,
         const AddrFn& addr) {
        int kv0 = ti * Traits::BC;
        typename Traits::Elem* dK = sK + buf * Traits::BC * Traits::LD;
        typename Traits::Elem* dV = sV + buf * Traits::BC * Traits::LD;
#pragma unroll
        for (int i = threadIdx.x * Traits::VEC; i < Traits::TOTAL;
             i += Traits::NUM_THREADS * Traits::VEC) {
            int r = i / Traits::HEAD_DIM, d = i % Traits::HEAD_DIM;
            int kc = kv0 + r;
            bool valid = kc < seq_len;
            auto a = addr(kc, d, valid);
            int off = r * Traits::LD + Traits::SharedLayout::column(d, r);
            if (a.vector_aligned) {
                astrai::cp_async_16(&dK[off], a.k, a.valid);
                astrai::cp_async_16(&dV[off], a.v, a.valid);
            } else {
                load_unaligned(&dK[off], &dV[off], static_cast<const typename Traits::Elem*>(a.k),
                               static_cast<const typename Traits::Elem*>(a.v), a.valid);
            }
        }
        astrai::cp_async_commit_group();
    }
};

struct MaskView {
    const bool* __restrict__ mask;
    int b_stride, h_stride, l_stride;
    int k_len, q_len;
    int batch, head0, head1;
    int qrow0, qrow1;
};

template <bool HasMask, bool MaskCoversShape = false> struct AttentionMask {
    const bool* __restrict__ mask;
    int base0, base1, max_key0, max_key1;
    bool valid0, valid1;
    DEVICE_FORCEINLINE AttentionMask(MaskView view, int max0, int max1, bool row0, bool row1)
        : mask(view.mask), base0(view.batch * view.b_stride + view.head0 * view.h_stride +
                                 view.qrow0 * view.l_stride),
          base1(view.batch * view.b_stride + view.head1 * view.h_stride +
                view.qrow1 * view.l_stride),
          max_key0(HasMask && !MaskCoversShape ? min(max0, view.k_len) : max0),
          max_key1(HasMask && !MaskCoversShape ? min(max1, view.k_len) : max1),
          valid0(row0 &&
                 (!HasMask || MaskCoversShape || view.q_len == 1 || view.qrow0 < view.q_len)),
          valid1(row1 &&
                 (!HasMask || MaskCoversShape || view.q_len == 1 || view.qrow1 < view.q_len)) {}

    DEVICE_FORCEINLINE bool is_masked(int row, int key) const {
        return !(row ? valid1 : valid0) || key >= (row ? max_key1 : max_key0) ||
               (HasMask && !mask[(row ? base1 : base0) + key]);
    }
};

/*
 * Q scheduling policies
 *
 * Map CUDA blocks to request-local Q tiles independently of K/V storage.
 * Dense tensors encode the request in blockIdx.z; packed ragged tensors use
 * a compact precomputed work map indexed by blockIdx.x.
 */

struct DenseQSchedule {
    static HOST_FORCEINLINE int host_grid_batch(const AttentionParams& p) { return p.batch; }

    static DEVICE_FORCEINLINE void map_block(const AttentionParams&, int& batch, int& q_tile) {
        batch = blockIdx.z;
        q_tile = blockIdx.x;
    }

    /*
     * PackGQA-folded prefill mapping: the block's row space is the packed
     * (head, row) space of the request — G heads folded, h = idx % G,
     * m = idx / G over the GLOBAL packed index (a per-block offset would
     * shift the head phase when block_m % G != 0). Dense tensors tile the
     * packed space G*q_len directly.
     */
    static HOST_FORCEINLINE int packed_grid_x(const AttentionParams& p, int, int block_m) {
        const int G = p.q_head / p.kv_head;
        return (p.q_len * G + block_m - 1) / block_m;
    }

    static DEVICE_FORCEINLINE void
    map_packed_block(const AttentionParams&, int block_m, int& batch, int& packed0) {
        batch = blockIdx.z;
        packed0 = blockIdx.x * block_m;
    }

    static DEVICE_FORCEINLINE int q_len(const AttentionParams& p, int) { return p.q_len; }

    static DEVICE_FORCEINLINE int q_base(const AttentionParams& p, int batch, int q_head) {
        return batch * p.q_b_stride + q_head * p.q_h_stride;
    }
};

struct PackedQSchedule {
    static HOST_FORCEINLINE int host_q_blocks(const AttentionParams& p, int) {
        return p.num_q_tiles;
    }

    static HOST_FORCEINLINE int host_grid_batch(const AttentionParams&) { return 1; }

    static DEVICE_FORCEINLINE void map_block(const AttentionParams& p, int& batch, int& q_tile) {
        batch = p.q_tile_to_batch[blockIdx.x];
        q_tile = p.q_tile_to_index[blockIdx.x];
    }

    /*
     * PackGQA-folded prefill mapping: the host tile maps are built in
     * HOST_Q_TILE_ROWS granularity and stay head-agnostic; host tile t covers
     * rows [t*HQR, (t+1)*HQR) of every head, i.e. packed idx
     * [t*HQR*G, ...+HQR*G). Blocks carve that space in block_m steps; the
     * kernel decodes h = idx % G over the request-local packed index.
     * qo_indptr is a device pointer — the host grid derives from the tile
     * count alone (a request's last tile is padded up by the host builder).
     */
    static HOST_FORCEINLINE int packed_grid_x(const AttentionParams& p, int, int block_m) {
        const int blocks_per_host_tile = (p.q_head / p.kv_head) * HOST_Q_TILE_ROWS / block_m;
        return p.num_q_tiles * blocks_per_host_tile;
    }

    static DEVICE_FORCEINLINE void
    map_packed_block(const AttentionParams& p, int block_m, int& batch, int& packed0) {
        const int G = p.q_head / p.kv_head;
        const int blocks_per_host_tile = G * HOST_Q_TILE_ROWS / block_m;
        const int host_tile = blockIdx.x / blocks_per_host_tile;
        batch = p.q_tile_to_batch[host_tile];
        const int in_tile = blockIdx.x - host_tile * blocks_per_host_tile;
        packed0 = p.q_tile_to_index[host_tile] * HOST_Q_TILE_ROWS * G + in_tile * block_m;
    }

    static DEVICE_FORCEINLINE int q_len(const AttentionParams& p, int batch) {
        return p.qo_indptr[batch + 1] - p.qo_indptr[batch];
    }

    static DEVICE_FORCEINLINE int q_base(const AttentionParams& p, int batch, int q_head) {
        return p.qo_indptr[batch] * p.q_l_stride + q_head * p.q_h_stride;
    }
};

// Hoisted per-(batch, kv_head) addressing context.
struct KVContext {
    int kv_base;         // contig: batch*kv_b_stride + kv_head*kv_h_stride
    int req_idx;         // paged: req_pool_indices[batch]
    int64_t rtt_stride;  // paged: max_context_len
    int64_t pool_stride; // paged: kv_head * HEAD_DIM
    int64_t head_off;    // paged: kv_head * HEAD_DIM
};

/*
 * Per-element K/V global addresses for one (kc, d) position of a K/V tile.
 * The pointers are ALWAYS the computed addresses (never nullptr) — callers
 * gate on `valid` (cp.async src_size=0, or a guarded scalar deref).  `valid`
 * starts as "within the request's seq_len"; the paged policy further degrades
 * it when req_to_token maps the position to a negative slot (empty padding).
 * This matches the original hand-rolled load loops, where the address was
 * always formed and the predicate decided whether anything was read.
 */
struct KVAddr {
    const void* k;
    const void* v;
    bool valid;
    bool vector_aligned = true;
};

// Contiguous K/V
template <typename T> struct ContigKV {
    using Elem = T;
    static constexpr bool kPaged = false;

    /*
     * Typed views of the params' dtype-agnostic pointers. The restrict
     * locals at the deref sites re-state the alias promise the void* -> T*
     * cast drops.
     */
    static DEVICE_FORCEINLINE const T* kptr(const AttentionParams& p) {
        return static_cast<const T*>(p.k_ptr);
    }
    static DEVICE_FORCEINLINE const T* vptr(const AttentionParams& p) {
        return static_cast<const T*>(p.v_ptr);
    }

    static HOST_FORCEINLINE int host_kv_len(const AttentionParams& p) { return p.kv_len; }

    // decode: same offset (q_len == 1, so there is no row stride component)
    static DEVICE_FORCEINLINE int q_decode_base(const AttentionParams& p, int batch, int q_head) {
        return batch * p.q_b_stride + q_head * p.q_h_stride;
    }

    static DEVICE_FORCEINLINE int kv_len(const AttentionParams& p, int) { return p.kv_len; }

    template <int HEAD_DIM>
    static DEVICE_FORCEINLINE KVContext make_ctx(const AttentionParams& p, int batch, int kv_head) {
        KVContext c = {};
        c.kv_base = batch * p.kv_b_stride + kv_head * p.kv_h_stride;
        return c;
    }
    static DEVICE_FORCEINLINE int
    resolve_token(const AttentionParams& p, const KVContext& c, int kc, bool valid) {
        return valid ? kc : -1;
    }
    static DEVICE_FORCEINLINE KVAddr kv_addr_from_token(const AttentionParams& p,
                                                        const KVContext& c,
                                                        int token,
                                                        int d) {
        const bool valid = token >= 0;
        const int safe_token = valid ? token : 0;
        const int64_t gmem_off =
            (int64_t)c.kv_base + (int64_t)safe_token * p.kv_l_stride + (int64_t)d * p.kv_d_stride;
        const T* __restrict__ k = kptr(p);
        const T* __restrict__ v = vptr(p);
        return {&k[gmem_off], &v[gmem_off], valid};
    }

    template <int VEC>
    static DEVICE_FORCEINLINE KVAddr decode_addr(const AttentionParams& p,
                                                 const KVContext& c,
                                                 int,
                                                 int,
                                                 int,
                                                 int kc,
                                                 int d,
                                                 bool valid,
                                                 bool) {
        int token = resolve_token(p, c, kc, valid);
        return kv_addr_from_token(p, c, token, d);
    }
};

// Paged K/V backed by an SGLang-style flat pool
template <typename T, bool AlignedNewKV = true> struct PagedKV {
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

    static HOST_FORCEINLINE int host_kv_len(const AttentionParams& p) { return p.max_context_len; }

    // Check both ends of the new-token copy without reading device metadata.
    static HOST_FORCEINLINE bool new_kv_aligned(const AttentionParams& p) {
        if (!p.new_k_ptr)
            return true;
        const uintptr_t addresses =
            reinterpret_cast<uintptr_t>(p.k_ptr) | reinterpret_cast<uintptr_t>(p.v_ptr) |
            reinterpret_cast<uintptr_t>(p.new_k_ptr) | reinterpret_cast<uintptr_t>(p.new_v_ptr);
        const uintptr_t batch_stride =
            p.batch > 1 ? static_cast<uintptr_t>(p.new_kv_b_stride) * sizeof(T) : 0;
        const uintptr_t head_stride =
            p.kv_head > 1 ? static_cast<uintptr_t>(p.new_kv_h_stride) * sizeof(T) : 0;
        return ((addresses | batch_stride | head_stride) & 15) == 0;
    }

    // decode: Q is [batch, q_head, head_dim], so batch is the outer row
    static DEVICE_FORCEINLINE int q_decode_base(const AttentionParams& p, int batch, int q_head) {
        return batch * p.q_l_stride + q_head * p.q_h_stride;
    }

    static DEVICE_FORCEINLINE int kv_len(const AttentionParams& p, int batch) {
        return p.kv_indptr[batch + 1] - p.kv_indptr[batch];
    }

    template <int HEAD_DIM>
    static DEVICE_FORCEINLINE KVContext make_ctx(const AttentionParams& p, int batch, int kv_head) {
        KVContext c = {};
        c.req_idx = p.req_pool_indices[batch];
        c.rtt_stride = (int64_t)p.max_context_len;
        c.pool_stride = (int64_t)p.kv_head * HEAD_DIM;
        c.head_off = (int64_t)kv_head * HEAD_DIM;
        return c;
    }
    static DEVICE_FORCEINLINE int
    resolve_token(const AttentionParams& p, const KVContext& c, int kc, bool valid) {
        return valid ? p.req_to_token[c.req_idx * c.rtt_stride + kc] : -1;
    }
    static DEVICE_FORCEINLINE KVAddr kv_addr_from_token(const AttentionParams& p,
                                                        const KVContext& c,
                                                        int slot,
                                                        int d) {
        const bool valid = slot >= 0;
        const int safe_slot = valid ? slot : 0;
        const int64_t gmem_off = (int64_t)safe_slot * c.pool_stride + c.head_off + d;
        const T* __restrict__ k = kptr(p);
        const T* __restrict__ v = vptr(p);
        return {&k[gmem_off], &v[gmem_off], valid};
    }

    static DEVICE_FORCEINLINE KVAddr new_kv_addr(const AttentionParams& p,
                                                 int batch,
                                                 int kv_head,
                                                 int d) {
        const int64_t off =
            (int64_t)batch * p.new_kv_b_stride + (int64_t)kv_head * p.new_kv_h_stride + d;
        const T* __restrict__ nk = new_kptr(p);
        const T* __restrict__ nv = new_vptr(p);
        return {&nk[off], &nv[off], true, AlignedNewKV};
    }

    template <int VEC>
    static DEVICE_FORCEINLINE void
    store_new_kv(const AttentionParams& p, const KVContext& c, int kc, int d, const KVAddr& src) {
        const int slot = resolve_token(p, c, kc, true);
        if (slot < 0)
            return;
        const int64_t off = (int64_t)slot * c.pool_stride + c.head_off + d;
        T* __restrict__ k = const_cast<T*>(kptr(p)) + off;
        T* __restrict__ v = const_cast<T*>(vptr(p)) + off;
        const T* __restrict__ nk = static_cast<const T*>(src.k);
        const T* __restrict__ nv = static_cast<const T*>(src.v);
        if constexpr (AlignedNewKV && VEC * sizeof(T) == 16) {
            // The host checks source and pool alignment before selecting this policy.
            *reinterpret_cast<uint4*>(k) = *reinterpret_cast<const uint4*>(nk);
            *reinterpret_cast<uint4*>(v) = *reinterpret_cast<const uint4*>(nv);
            return;
        }
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            k[j] = nk[j];
            v[j] = nv[j];
        }
    }

    template <int VEC>
    static DEVICE_FORCEINLINE KVAddr decode_addr(const AttentionParams& p,
                                                 const KVContext& c,
                                                 int batch,
                                                 int kv_head,
                                                 int seq_len,
                                                 int kc,
                                                 int d,
                                                 bool valid,
                                                 bool persist) {
        if (p.new_k_ptr && valid && kc == seq_len - 1) {
            KVAddr src = new_kv_addr(p, batch, kv_head, d);
            if (persist)
                store_new_kv<VEC>(p, c, kc, d, src);
            return src;
        }
        int token = resolve_token(p, c, kc, valid);
        return kv_addr_from_token(p, c, token, d);
    }
};

} // namespace attention
} // namespace astrai
