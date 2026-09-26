#pragma once
// Shared attention dispatchers — used by both production .cu and test .cu.
// No torch dependency; pure CUDA.
//
// The paged and contiguous kernels are unified by the KVSource policy
// (ContigKV<T> / PagedKV<T> from layout_policies.cuh), so each launcher struct
// below is templated on KV and the paged dispatch is just the same launcher
// instantiated with PagedKV.  Only the grid/split math differs, and that is
// covered by KV::host_kv_len. For the same reason the dispatch_* entries
// funnel through one impl per family (dispatch_prefill_impl /
// dispatch_decode_impl), so the MMA/scalar selection lives in one place
// instead of once per entry.
//
// The element type is a template parameter of every entry, and it rides on the
// KV instantiation (KV::Elem) from there down: the torch-facing .cu entries
// pick it from ASTRAI_ATTN_DTYPE_LIST (attention/dtype_list.cuh) and the C++
// harnesses pass it directly, so nothing here reads a runtime dtype tag and the
// params POD carries none.

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <algorithm>

#include <utils/dtype.cuh>
#include <utils/launch.cuh>
#include <memory/layout_policies.cuh>
#include <kernel/attention_prefill_split_q.cuh>
#include <kernel/attention_decode_split_kv.cuh>
#ifndef ASTRAI_NO_MMA
#include <kernel/attention_prefill_split_q_mma.cuh>
#include <kernel/attention_decode_split_kv_mma.cuh>
#endif

namespace astrai {
namespace attention {

// A head dim no kernel was instantiated for: the torch entry validates the
// granularity and the Python backend gates the set (drift-asserted against
// HEAD_DIMS), so this is the loud belt-and-braces arm of the switches below —
// the launch discipline: never run a kernel that was not built for the shape.
[[noreturn]] inline void head_dim_fatal(int head_dim) {
    std::fprintf(stderr,
                 "ASTRAI: attention: head_dim %d has no kernel instantiation "
                 "(instantiated: 32, 64, 128, 256)\n",
                 head_dim);
    std::exit(EXIT_FAILURE);
}

// Split-KV: compute number of splits to fill all SMs for small-batch decode.
// Caps splits so each split processes at least `min_tiles_per_split` tiles,
// avoiding excessive loop/prologue overhead when tiles are small.
//
// Target total grid blocks (`TARGET_BLOCKS`) rather than scaling splits by SM
// count.  Decode blocks are single-warp (32 threads) and a SM hosts ~11 of
// them, so the old `2*sm/base` cap badly undersplit at large batch (B=16 got
// 3 splits, optimal ~8).  Measured (L20, grid search): bandwidth saturates
// near 256-512 total blocks; 512 minimizes worst-case latency across the
// B x kv grid; more is pure oversplit overhead.
constexpr int DECODE_TARGET_BLOCKS = 512;
inline int compute_num_splits(int base_blocks, int tiles_total,
                               int min_tiles_per_split = 1) {
    int n = (DECODE_TARGET_BLOCKS + base_blocks - 1) / base_blocks;
    int max_by_work = tiles_total / min_tiles_per_split;
    return std::max(1, std::min(n, std::min(max_by_work, MAX_SPLITS)));
}

// Dispatch IsCausal × HasMask — eliminates the duplicated 4-way if/else
// ladder that appeared in each dispatch_* function.  FN must be a function
// template <int HEAD_DIM, bool IsCausal, bool HasMask>; HEAD_DIM is forwarded
// as the first template argument so callers only spell it once.
//
// Usage:  DISPATCH_CAUSAL_MASK(is_causal, has_mask, launcher<KV>::template launch, HEAD_DIM, p, stream);
#define DISPATCH_CAUSAL_MASK(is_causal, has_mask, FN, HEAD_DIM, ...) \
    do { \
        if (is_causal) { \
            if (has_mask) FN<HEAD_DIM, true,  true>(__VA_ARGS__); \
            else          FN<HEAD_DIM, true,  false>(__VA_ARGS__); \
        } else { \
            if (has_mask) FN<HEAD_DIM, false, true>(__VA_ARGS__); \
            else          FN<HEAD_DIM, false, false>(__VA_ARGS__); \
        } \
    } while (0)

// ======================================================================
// Kernel family: ONE place picks the MMA or the scalar implementation.
//
// Both families expose the same launcher interface — a static
// launch<HEAD_DIM, IsCausal, HasMask> — so the dispatchers below name only
// these aliases. Declared here and defined in the sections below;
// ASTRAI_NO_MMA (the manual escape hatch the build never sets) swaps both
// families at once, instead of the choice being repeated at each dispatch
// site.
// ======================================================================

#ifndef ASTRAI_NO_MMA
template <typename QSchedule, typename KV>
struct PrefillLauncherMMA;
template <typename KV>
struct DecodeLauncherMMA;
#endif

template <typename QSchedule, typename KV>
struct PrefillLauncherScalar;
template <typename KV>
struct DecodeLauncherScalar;

#ifndef ASTRAI_NO_MMA
template <typename QSchedule, typename KV>
using PrefillLauncher = PrefillLauncherMMA<QSchedule, KV>;
template <typename KV>
using DecodeLauncher = DecodeLauncherMMA<KV>;
#else
template <typename QSchedule, typename KV>
using PrefillLauncher = PrefillLauncherScalar<QSchedule, KV>;
template <typename KV>
using DecodeLauncher = DecodeLauncherScalar<KV>;
#endif

#ifndef ASTRAI_NO_MMA
template <int BC_>
struct PrefillKernelConfig {
    static constexpr int BC = BC_;
    static constexpr int WARPS = 4;
    static constexpr int STAGES = 2;
};

// Compile-time configuration map shared by contiguous and paged prefill.
// Unsupported head dimensions intentionally have no mapping.
template <int HEAD_DIM, bool IsCausal>
struct PrefillConfigMap;

template <> struct PrefillConfigMap<32, false>  : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<32, true>   : PrefillKernelConfig<64> {};
template <> struct PrefillConfigMap<64, false>  : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<64, true>   : PrefillKernelConfig<64> {};
template <> struct PrefillConfigMap<128, false> : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<128, true>  : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<256, false> : PrefillKernelConfig<16> {};
template <> struct PrefillConfigMap<256, true>  : PrefillKernelConfig<16> {};

template <typename QSchedule, typename KV>
struct PrefillLauncherMMA {
    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void launch(AttentionParams& p, cudaStream_t stream) {
        using Config = PrefillConfigMap<HEAD_DIM, IsCausal>;
        using Traits = KernelTraits<HEAD_DIM, Config::BC, Config::WARPS,
                                    Config::STAGES, typename KV::Elem>;
        // GQA head packing: HB = min(G, WARPS) q-heads of one kv-head group
        // share each block's K/V stream (~HB× less global K/V traffic).
        // Each head gets WPH = WARPS/HB 16-row chunks per block, so per-head
        // rows drop from 64 to BR*WPH while total mma work per K/V byte is
        // unchanged.  G=1 (MHA) reproduces the historical grid exactly.
        const int G = p.q_head / p.kv_head;
        const int HB = std::min(G, Config::WARPS);
        const int WPH = Config::WARPS / HB;
        constexpr int BR = Traits::BR;
        dim3 grid(QSchedule::packed_grid_x(p, BR * WPH),
                  p.kv_head * ((G + HB - 1) / HB),
                  QSchedule::host_grid_batch(p));
        dim3 block(Traits::NUM_THREADS);
        attn_prefill_split_q_mma_kernel<Traits, QSchedule, KV, IsCausal, HasMask>
            <<<grid, block, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
};
#endif

template <typename QSchedule, typename KV>
struct PrefillLauncherScalar {
    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void launch(AttentionParams& p, cudaStream_t stream) {
        constexpr int G = (HEAD_DIM == 32) ? 4 : 8, ROWS = 64, P_BC = 32;
        dim3 grid(QSchedule::host_q_blocks(p, ROWS), p.q_head,
                  QSchedule::host_grid_batch(p));
        dim3 block(G, ROWS);
        attn_prefill_split_q_kernel_t<HEAD_DIM, QSchedule, KV, G, ROWS, P_BC,
                                      IsCausal, HasMask>
            <<<grid, block, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
};

// The contiguous and paged entries differ only in the policy pair (Q schedule
// + KV source), so one implementation carries the MMA/scalar selection and the
// causal/mask ladder for both.
template <typename QSchedule, typename KV, int HEAD_DIM>
static inline void dispatch_prefill_impl(AttentionParams& p,
                                         cudaStream_t stream) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);

    using Launcher = PrefillLauncher<QSchedule, KV>;
    DISPATCH_CAUSAL_MASK(is_causal, has_mask,
                         Launcher::template launch,
                         HEAD_DIM, p, stream);
}

// One entry per family: T is the element type (the whole dtype axis), the head
// dim is switched here because it selects tile configs, not storage.
template <typename T>
static inline void dispatch_prefill(AttentionParams& p, cudaStream_t stream) {
    switch (p.head_dim) {
        case 32:  dispatch_prefill_impl<DenseQSchedule, ContigKV<T>, 32>(p, stream); break;
        case 64:  dispatch_prefill_impl<DenseQSchedule, ContigKV<T>, 64>(p, stream); break;
        case 128: dispatch_prefill_impl<DenseQSchedule, ContigKV<T>, 128>(p, stream); break;
        case 256: dispatch_prefill_impl<DenseQSchedule, ContigKV<T>, 256>(p, stream); break;
        default:  head_dim_fatal(p.head_dim);
    }
}

template <typename T>
static inline void dispatch_paged_prefill(AttentionParams& p, cudaStream_t stream) {
    switch (p.head_dim) {
        case 32:  dispatch_prefill_impl<PackedQSchedule, PagedKV<T>, 32>(p, stream); break;
        case 64:  dispatch_prefill_impl<PackedQSchedule, PagedKV<T>, 64>(p, stream); break;
        case 128: dispatch_prefill_impl<PackedQSchedule, PagedKV<T>, 128>(p, stream); break;
        case 256: dispatch_prefill_impl<PackedQSchedule, PagedKV<T>, 256>(p, stream); break;
        default:  head_dim_fatal(p.head_dim);
    }
}

// ======================================================================
// Decode launchers (KV selects ContigKV or PagedKV addressing)
// ======================================================================

#ifndef ASTRAI_NO_MMA
// BC=16: halves smem (16KB vs 32KB) → doubles occupancy (6 vs 3 blocks/SM).
// For D=256, BC=16 also reduces register pressure (fewer Sacc/PV frags),
// enabling STAGES=2 (double-buffer) within the 32KB smem budget — eliminates
// the 176-byte spill that STAGES=1+BC=32 suffered.
template <typename KV>
struct DecodeLauncherMMA {
    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void launch(AttentionParams& p, cudaStream_t stream) {
        int G = p.q_head / p.kv_head;
        constexpr int MAX_G = 16;
        int num_passes = (G + MAX_G - 1) / MAX_G;
        constexpr int BC = 16;
        int kv_len = KV::host_kv_len(p);
        int tiles_total = (kv_len + BC - 1) / BC;
        p.num_splits = compute_num_splits(p.batch * p.kv_head * num_passes, tiles_total, 2);
        constexpr int STAGES = 2;
        using Traits = KernelTraits<HEAD_DIM, BC, 1, STAGES, typename KV::Elem>;
        dim3 grid(p.kv_head * num_passes, p.batch, p.num_splits);
        attn_decode_split_kv_mma_kernel<Traits, KV, IsCausal, HasMask>
            <<<grid, 32, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
};
#endif

template <typename KV>
struct DecodeLauncherScalar {
    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void launch(AttentionParams& p, cudaStream_t stream) {
        int kv_len = KV::host_kv_len(p);
        int chunks_total = (kv_len + DC_CHUNK - 1) / DC_CHUNK;
        p.num_splits = compute_num_splits(p.batch * p.kv_head, chunks_total);
        size_t smem = 2 * DC_CHUNK * p.head_dim * sizeof(typename KV::Elem);
        int group_size = p.q_head / p.kv_head;
        int g = min(group_size, 32);  // cap at 32 to respect 1024-thread limit
        dim3 grid(p.batch * p.kv_head, 1, p.num_splits);
        dim3 block(32, g);
        ASTRAI_CUDA_CHECK(cudaFuncSetAttribute(
            attn_decode_split_kv_kernel<HEAD_DIM, KV, IsCausal, HasMask>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem));
        attn_decode_split_kv_kernel<HEAD_DIM, KV, IsCausal, HasMask>
            <<<grid, block, smem, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
};

// Same funnel as prefill: the entry's only difference is the KV policy, which
// also parameterizes the combine pass reducing the split partials.
template <typename KV, int HEAD_DIM>
static inline void dispatch_decode_impl(AttentionParams& p,
                                        cudaStream_t stream) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);

    using Launcher = DecodeLauncher<KV>;
    DISPATCH_CAUSAL_MASK(is_causal, has_mask,
                         Launcher::template launch,
                         HEAD_DIM, p, stream);

    attn_decode_combine_kernel<KV><<<p.batch * p.q_head, p.head_dim, 0, stream>>>(p);
    ASTRAI_LAUNCH_CHECK();
}

template <typename T>
static inline void dispatch_decode(AttentionParams& p, cudaStream_t stream) {
    switch (p.head_dim) {
        case 32:  dispatch_decode_impl<ContigKV<T>, 32>(p, stream); break;
        case 64:  dispatch_decode_impl<ContigKV<T>, 64>(p, stream); break;
        case 128: dispatch_decode_impl<ContigKV<T>, 128>(p, stream); break;
        case 256: dispatch_decode_impl<ContigKV<T>, 256>(p, stream); break;
        default:  head_dim_fatal(p.head_dim);
    }
}

template <typename T>
static inline void dispatch_paged_decode(AttentionParams& p, cudaStream_t stream) {
    switch (p.head_dim) {
        case 32:  dispatch_decode_impl<PagedKV<T>, 32>(p, stream); break;
        case 64:  dispatch_decode_impl<PagedKV<T>, 64>(p, stream); break;
        case 128: dispatch_decode_impl<PagedKV<T>, 128>(p, stream); break;
        case 256: dispatch_decode_impl<PagedKV<T>, 256>(p, stream); break;
        default:  head_dim_fatal(p.head_dim);
    }
}

}  // namespace attention
}  // namespace astrai
