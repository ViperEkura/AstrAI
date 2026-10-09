/*
 * Shared attention launch vocabulary — split-KV heuristic, head_dim guard,
 * causal×mask dispatch ladder, kernel-family launchers with their tile-config
 * maps, and the four family dispatchers. Pure CUDA, no torch: the standalone
 * harnesses compile the exact code the production dispatch runs, and each
 * production .cu is then exactly one torch-facing function. Kernel bodies
 * live in kernel/attention/split_{q,kv}.cuh.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include <api/attention_common.h>
#include <kernel/attention/split_kv.cuh>
#include <kernel/attention/split_q.cuh>
#include <memory/layout_policies.cuh>
#include <utils/launch.cuh>

namespace astrai {
namespace attention {

/*
 * The one list of instantiated head dims — the dispatch switch, the error
 * message and (drift-asserted) the Python backend's HEAD_DIMS all read it.
 */
#define ASTRAI_ATTN_HEAD_DIMS(X) X(32) X(64) X(128) X(256)

/*
 * A head dim no kernel was instantiated for — the launch discipline: never
 * run a kernel that was not built for the shape.
 */
inline std::string head_dim_error(int head_dim) {
    std::string message = "ASTRAI: attention: head_dim " + std::to_string(head_dim) +
                          " has no kernel instantiation (instantiated:";
#define ASTRAI_HEAD_DIM_ROW(D) message += " " + std::to_string(D);
    ASTRAI_ATTN_HEAD_DIMS(ASTRAI_HEAD_DIM_ROW)
#undef ASTRAI_HEAD_DIM_ROW
    return message + ")";
}

/*
 * Split-KV count: fill exactly one wave of blocks.
 *
 * The GPU runs blocks in waves of (SM count x resident blocks per SM). A
 * grid smaller than a wave leaves SMs idle; a grid that crosses into a
 * second wave pays a full extra wave of latency for the few straggler
 * blocks. So the split count is chosen to bring the grid as close to one
 * full wave as possible without crossing it:
 *   grid = base_blocks * splits <= wave_capacity
 *   =>   splits = floor(wave_capacity / base_blocks)
 * The work caps still apply: never more splits than the tile count allows
 * (each split needs at least min_tiles_per_split tiles to not be pure
 * combine overhead) and never more than MAX_SPLITS.
 */
inline int compute_num_splits(int base_blocks,
                              int tiles_total,
                              int wave_capacity,
                              int min_tiles_per_split = 1) {
    int cap = std::min(tiles_total / std::max(min_tiles_per_split, 1), MAX_SPLITS);
    if (cap <= 1)
        return 1;
    return std::max(1, std::min(wave_capacity / std::max(base_blocks, 1), cap));
}

/*
 * Cache wave capacity per kernel instantiation and device. Head dims and mask
 * variants can have different register footprints.
 */
template <auto Kernel, int Threads> inline int decode_wave_capacity() {
    static std::atomic<int> cached[64] = {};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return 1;
    const bool cacheable = device >= 0 && device < 64;
    if (cacheable) {
        const int value = cached[device].load(std::memory_order_relaxed);
        if (value)
            return value;
    }
    int sms = 0, per_sm = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
        return 1;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, Kernel, Threads, 0) != cudaSuccess ||
        per_sm < 1)
        per_sm = 1;
    const int capacity = std::max(1, sms * per_sm);
    if (cacheable)
        cached[device].store(capacity, std::memory_order_relaxed);
    return capacity;
}

/*
 * Kernel-family launchers. Every supported target is sm_80+, so the
 * tensor-core kernels are the only implementations; both families expose
 * the same static launch<HEAD_DIM, IsCausal, HasMask> interface.
 */

template <int BC_> struct PrefillKernelConfig {
    static constexpr int BC = BC_;
    static constexpr int WARPS = 4;
    static constexpr int STAGES = 2;
};

/*
 * Prefill tile-config map (BC by head_dim × causal), shared by the
 * contiguous and paged entries. Unsupported head dims have no mapping.
 */
template <int HEAD_DIM, bool IsCausal> struct PrefillConfigMap;

template <> struct PrefillConfigMap<32, false> : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<32, true> : PrefillKernelConfig<64> {};
template <> struct PrefillConfigMap<64, false> : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<64, true> : PrefillKernelConfig<64> {};
template <> struct PrefillConfigMap<128, false> : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<128, true> : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<256, false> : PrefillKernelConfig<16> {};
template <> struct PrefillConfigMap<256, true> : PrefillKernelConfig<16> {};

template <typename QSchedule, typename KV> struct PrefillLauncher {
    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void launch(AttentionParams& p, cudaStream_t stream) {
        using Config = PrefillConfigMap<HEAD_DIM, IsCausal>;
        using Traits =
            KernelTraits<HEAD_DIM, Config::BC, Config::WARPS, Config::STAGES, typename KV::Elem>;
        /*
         * PackGQA-folded grid: each block owns BLOCK_M packed (head,row) rows
         * of the request's packed space; grid.y is the kv head (see the
         * prefill kernel header for the fold math).
         */
        constexpr int BLOCK_M = Traits::BR * Config::WARPS;
        dim3 grid(QSchedule::packed_grid_x(p, BLOCK_M, BLOCK_M), p.kv_head,
                  QSchedule::host_grid_batch(p));
        dim3 block(Traits::NUM_THREADS);
        attn_prefill_split_q_mma_kernel<Traits, QSchedule, KV, IsCausal, HasMask>
            <<<grid, block, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
};

/*
 * BC=16: halves smem (16KB vs 32KB) → doubles occupancy; for D=256 it also
 * cuts register pressure enough for STAGES=2 within the 32KB budget,
 * eliminating the 176-byte spill of STAGES=1+BC=32.
 */
template <typename KV> struct DecodeLauncher {
    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void plan(AttentionParams& p, cudaStream_t) {
        int G = p.q_head / p.kv_head;
        constexpr int MAX_G = 16;
        int num_passes = (G + MAX_G - 1) / MAX_G;
        constexpr int BC = 16;
        int kv_len = KV::host_kv_len(p);
        int tiles_total = (kv_len + BC - 1) / BC;
        using Traits = KernelTraits<HEAD_DIM, BC, 1, 2, typename KV::Elem>;
        p.num_splits = compute_num_splits(
            p.batch * p.kv_head * num_passes, tiles_total,
            decode_wave_capacity<attn_decode_split_kv_mma_kernel<Traits, KV, HasMask>,
                                 32>(),
            2);
        // Wider direct-output kernels lose graph latency; retain the split path.
        p.direct_output = p.num_splits == 1 && HEAD_DIM <= 128;
    }

    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void launch(AttentionParams& p, cudaStream_t stream) {
        using Traits = KernelTraits<HEAD_DIM, 16, 1, 2, typename KV::Elem>;
        const int passes = (p.q_head / p.kv_head + 15) / 16;
        const dim3 grid(p.kv_head * passes, p.batch, p.num_splits);
        if constexpr (HEAD_DIM <= 128) {
            if (p.direct_output) {
                attn_decode_split_kv_mma_kernel<Traits, KV, HasMask, true>
                    <<<grid, 32, 0, stream>>>(p);
                ASTRAI_LAUNCH_CHECK();
                return;
            }
        }
        attn_decode_split_kv_mma_kernel<Traits, KV, HasMask>
            <<<grid, 32, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
};

template <typename KV> struct DecodePlanner {
    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void launch(AttentionParams& p, cudaStream_t stream) {
        DecodeLauncher<KV>::template plan<HEAD_DIM, IsCausal, HasMask>(p, stream);
    }
};

template <int HEAD_DIM, typename Launcher>
inline void dispatch_causal_mask(bool causal, bool mask, AttentionParams& p, cudaStream_t stream) {
    if (causal && mask)
        return Launcher::template launch<HEAD_DIM, true, true>(p, stream);
    if (causal)
        return Launcher::template launch<HEAD_DIM, true, false>(p, stream);
    if (mask)
        return Launcher::template launch<HEAD_DIM, false, true>(p, stream);
    return Launcher::template launch<HEAD_DIM, false, false>(p, stream);
}

/*
 * Family dispatchers — shared between the production .cu entries and the
 * standalone torch-free harnesses, which compile them directly (a .o link
 * would drag the pybind module and torch in). T is the element type; the
 * head dim switches here because it selects tile configs, not storage. The
 * four entries differ only in the policy pair, so they funnel into one impl
 * per family.
 */

template <typename QSchedule, typename KV, int HEAD_DIM>
static inline void dispatch_prefill_impl(AttentionParams& p, cudaStream_t stream) {
    bool is_causal = p.is_causal;
    bool has_mask = (p.mask != nullptr);

    using Launcher = PrefillLauncher<QSchedule, KV>;
    dispatch_causal_mask<HEAD_DIM, Launcher>(is_causal, has_mask, p, stream);
}

/*
 * Decode funnel: the causal/mask ladder plus the combine pass reducing the
 * split partials.
 */
template <typename KV, int HEAD_DIM>
static inline void dispatch_decode_impl(AttentionParams& p, cudaStream_t stream) {
    bool has_mask = (p.mask != nullptr);

    if (p.num_splits == 0)
        dispatch_causal_mask<HEAD_DIM, DecodePlanner<KV>>(false, has_mask, p, stream);
    using Launcher = DecodeLauncher<KV>;
    dispatch_causal_mask<HEAD_DIM, Launcher>(false, has_mask, p, stream);

    if (!p.direct_output) {
        attn_decode_combine_kernel<KV><<<p.batch * p.q_head, p.head_dim, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
}

/*
 * One table-driven head-dim dispatch per family: the caller passes a
 * Fn type exposing run_dim<HEAD_DIM>(p, stream).
 */
template <typename Fn>
static inline void dispatch_head_dim(AttentionParams& p, cudaStream_t stream) {
    switch (p.head_dim) {
#define ASTRAI_HEAD_DIM_CASE(D)                                                                    \
    case D:                                                                                        \
        return Fn::template run_dim<D>(p, stream);
        ASTRAI_ATTN_HEAD_DIMS(ASTRAI_HEAD_DIM_CASE)
#undef ASTRAI_HEAD_DIM_CASE
    }
    throw std::runtime_error(head_dim_error(p.head_dim));
}

template <typename QSchedule, typename KV> struct PrefillDispatch {
    template <int HEAD_DIM> static void run_dim(AttentionParams& p, cudaStream_t stream) {
        dispatch_prefill_impl<QSchedule, KV, HEAD_DIM>(p, stream);
    }
    static void run(AttentionParams& p, cudaStream_t stream) {
        dispatch_head_dim<PrefillDispatch>(p, stream);
    }
};

template <typename KV> struct DecodeDispatch {
    template <int HEAD_DIM> static void run_dim(AttentionParams& p, cudaStream_t stream) {
        dispatch_decode_impl<KV, HEAD_DIM>(p, stream);
    }
    static void run(AttentionParams& p, cudaStream_t stream) {
        dispatch_head_dim<DecodeDispatch>(p, stream);
    }
};

template <typename T> using AttnDispatchPrefill = PrefillDispatch<DenseQSchedule, ContigKV<T>>;
template <typename T> using AttnDispatchPagedPrefill = PrefillDispatch<PackedQSchedule, PagedKV<T>>;
template <typename T> using AttnDispatchDecode = DecodeDispatch<ContigKV<T>>;
template <typename T> using AttnDispatchPagedDecode = DecodeDispatch<PagedKV<T>>;

} // namespace attention
} // namespace astrai
