// Shared attention launch vocabulary — split-KV heuristic, head_dim guard,
// causal×mask dispatch ladder, kernel-family launchers with their tile-config
// maps, and the four family dispatchers. Pure CUDA, no torch: the standalone
// harnesses compile the exact code the production dispatch runs, and each
// production .cu is then exactly one torch-facing function. Kernel bodies
// live in kernel/attention_{prefill_split_q,decode_split_kv}_mma.cuh.

#pragma once

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <cuda_runtime.h>

#include <kernel/attention_decode_split_kv_mma.cuh>
#include <kernel/attention_prefill_split_q_mma.cuh>
#include <memory/layout_policies.cuh>
#include <utils/attention_common.h>
#include <utils/launch.cuh>

namespace astrai {
namespace attention {

// A head dim no kernel was instantiated for — the launch discipline: never
// run a kernel that was not built for the shape (the Python backend gates
// the same set, drift-asserted against HEAD_DIMS).
[[noreturn]] inline void head_dim_fatal(int head_dim) {
    std::fprintf(stderr,
                 "ASTRAI: attention: head_dim %d has no kernel instantiation "
                 "(instantiated: 32, 64, 128, 256)\n",
                 head_dim);
    std::exit(EXIT_FAILURE);
}

// Split-KV: fill all SMs for small-batch decode, targeting total grid
// blocks rather than scaling by SM count (measured: bandwidth saturates
// near 256-512 blocks; 512 minimizes worst-case latency over the B×kv
// grid). Caps splits so each processes at least `min_tiles_per_split`
// tiles — more is pure oversplit overhead.
constexpr int DECODE_TARGET_BLOCKS = 512;
inline int compute_num_splits(int base_blocks, int tiles_total,
                              int min_tiles_per_split = 1) {
    int n = (DECODE_TARGET_BLOCKS + base_blocks - 1) / base_blocks;
    int max_by_work = tiles_total / min_tiles_per_split;
    return std::max(1, std::min(n, std::min(max_by_work, MAX_SPLITS)));
}

// Dispatch IsCausal × HasMask. FN is a function template
// <int HEAD_DIM, bool IsCausal, bool HasMask>; HEAD_DIM forwards first so
// callers spell it once:
//   DISPATCH_CAUSAL_MASK(is_causal, has_mask,
//                        launcher<KV>::template launch, HEAD_DIM, p, stream);
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

// Kernel-family launchers. Every supported target is sm_80+, so the
// tensor-core kernels are the only implementations; both families expose
// the same static launch<HEAD_DIM, IsCausal, HasMask> interface.

template <int BC_>
struct PrefillKernelConfig {
    static constexpr int BC = BC_;
    static constexpr int WARPS = 4;
    static constexpr int STAGES = 2;
};

// Prefill tile-config map (BC by head_dim × causal), shared by the
// contiguous and paged entries. Unsupported head dims have no mapping.
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
struct PrefillLauncher {
    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void launch(AttentionParams& p, cudaStream_t stream) {
        using Config = PrefillConfigMap<HEAD_DIM, IsCausal>;
        using Traits = KernelTraits<HEAD_DIM, Config::BC, Config::WARPS,
                                    Config::STAGES, typename KV::Elem>;
        // GQA head packing: HB = min(G, WARPS) q-heads share each block's
        // K/V stream (~HB× less traffic); G=1 (MHA) reproduces the
        // historical grid exactly. See the prefill kernel header for the
        // full warp-to-head/chunk mapping.
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

// BC=16: halves smem (16KB vs 32KB) → doubles occupancy; for D=256 it also
// cuts register pressure enough for STAGES=2 within the 32KB budget,
// eliminating the 176-byte spill of STAGES=1+BC=32.
template <typename KV>
struct DecodeLauncher {
    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void launch(AttentionParams& p, cudaStream_t stream) {
        int G = p.q_head / p.kv_head;
        constexpr int MAX_G = 16;
        int num_passes = (G + MAX_G - 1) / MAX_G;
        constexpr int BC = 16;
        int kv_len = KV::host_kv_len(p);
        int tiles_total = (kv_len + BC - 1) / BC;
        p.num_splits = compute_num_splits(p.batch * p.kv_head * num_passes,
                                          tiles_total, 2);
        constexpr int STAGES = 2;
        using Traits = KernelTraits<HEAD_DIM, BC, 1, STAGES, typename KV::Elem>;
        dim3 grid(p.kv_head * num_passes, p.batch, p.num_splits);
        attn_decode_split_kv_mma_kernel<Traits, KV, IsCausal, HasMask>
            <<<grid, 32, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
};

// Family dispatchers — shared between the production .cu entries and the
// standalone torch-free harnesses, which compile them directly (a .o link
// would drag the pybind module and torch in). T is the element type; the
// head dim switches here because it selects tile configs, not storage. The
// four entries differ only in the policy pair, so they funnel into one impl
// per family.

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

template <typename T>
static inline void dispatch_prefill(AttentionParams& p, cudaStream_t stream) {
    switch (p.head_dim) {
        case 32:
            dispatch_prefill_impl<DenseQSchedule, ContigKV<T>, 32>(p, stream);
            break;
        case 64:
            dispatch_prefill_impl<DenseQSchedule, ContigKV<T>, 64>(p, stream);
            break;
        case 128:
            dispatch_prefill_impl<DenseQSchedule, ContigKV<T>, 128>(p, stream);
            break;
        case 256:
            dispatch_prefill_impl<DenseQSchedule, ContigKV<T>, 256>(p, stream);
            break;
        default:
            head_dim_fatal(p.head_dim);
    }
}

template <typename QSchedule, typename KV, int HEAD_DIM>
static inline void dispatch_paged_prefill_impl(AttentionParams& p,
                                               cudaStream_t stream) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);

    using Launcher = PrefillLauncher<QSchedule, KV>;
    DISPATCH_CAUSAL_MASK(is_causal, has_mask,
                         Launcher::template launch,
                         HEAD_DIM, p, stream);
}

template <typename T>
static inline void dispatch_paged_prefill(AttentionParams& p,
                                          cudaStream_t stream) {
    switch (p.head_dim) {
        case 32:
            dispatch_paged_prefill_impl<PackedQSchedule, PagedKV<T>, 32>(p, stream);
            break;
        case 64:
            dispatch_paged_prefill_impl<PackedQSchedule, PagedKV<T>, 64>(p, stream);
            break;
        case 128:
            dispatch_paged_prefill_impl<PackedQSchedule, PagedKV<T>, 128>(p, stream);
            break;
        case 256:
            dispatch_paged_prefill_impl<PackedQSchedule, PagedKV<T>, 256>(p, stream);
            break;
        default:
            head_dim_fatal(p.head_dim);
    }
}

// Decode funnel: the causal/mask ladder plus the combine pass reducing the
// split partials.
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
static inline void dispatch_paged_decode(AttentionParams& p,
                                         cudaStream_t stream) {
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
