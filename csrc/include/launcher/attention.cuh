/*
 * Attention dispatch: select a kernel type, build a launch plan, then execute.
 * The typed visitor binds planning and launch to the same specialization.
 * Pure CUDA, no torch: native harnesses use the production dispatch path.
 * Kernel bodies live in kernel/attention/split_{q,kv}.cuh.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

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

// Plans contain host launch metadata only; tensor addresses stay in AttentionParams.
struct PrefillLaunchPlan {
    dim3 grid;
};

struct DecodePlanQuery {
    int batch;
    int q_heads;
    int kv_heads;
    int head_dim;
    int kv_tiles;
    int wave_capacity;
};

struct DecodeLaunchPlan {
    dim3 grid;
    bool direct_output;
};

// Pure split policy, shared by all layouts and independently testable.
inline DecodeLaunchPlan make_decode_plan(const DecodePlanQuery& query) {
    const int passes = (query.q_heads / query.kv_heads + 15) / 16;
    const int blocks_per_batch = query.kv_heads * passes;
    const int splits = compute_num_splits(
        query.batch * blocks_per_batch, query.kv_tiles, query.wave_capacity, 2);
    // Wider direct-output kernels lose graph latency; retain the combine path.
    return {dim3(blocks_per_batch, query.batch, splits),
            splits == 1 && query.head_dim <= 128};
}

template <int HEAD_DIM, typename QSchedule, typename KV, bool IsCausal, bool HasMask,
          bool MaskCoversShape = false>
struct PrefillKernel {
    using Config = PrefillConfigMap<HEAD_DIM, IsCausal>;
    using Traits =
        KernelTraits<HEAD_DIM, Config::BC, Config::WARPS, Config::STAGES, typename KV::Elem>;

    static PrefillLaunchPlan plan(const AttentionParams& p) {
        // Each block owns BLOCK_M packed (head, row) rows; grid.y selects the KV head.
        constexpr int BLOCK_M = Traits::BR * Config::WARPS;
        return {dim3(QSchedule::packed_grid_x(p, BLOCK_M, BLOCK_M), p.kv_head,
                     QSchedule::host_grid_batch(p))};
    }

    static void launch(const AttentionParams& p, const PrefillLaunchPlan& plan,
                       cudaStream_t stream) {
        attn_prefill_split_q_mma_kernel<Traits, QSchedule, KV, IsCausal, HasMask, MaskCoversShape>
            <<<plan.grid, Traits::NUM_THREADS, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
};

template <int HEAD_DIM, typename KV, bool HasMask> struct DecodeKernel {
    // BC=16 retains the existing shared-memory/register footprint and occupancy.
    using Traits = KernelTraits<HEAD_DIM, 16, 1, 2, typename KV::Elem>;

    static DecodePlanQuery query(const AttentionParams& p) {
        const int kv_len = KV::host_kv_len(p);
        // Keep the partial-output kernel as the occupancy reference for both modes.
        return {p.batch, p.q_head, p.kv_head, HEAD_DIM, (kv_len + Traits::BC - 1) / Traits::BC,
                decode_wave_capacity<attn_decode_split_kv_mma_kernel<Traits, KV, HasMask>,
                                     Traits::NUM_THREADS>()};
    }

    static DecodeLaunchPlan plan(const AttentionParams& p) {
        return make_decode_plan(query(p));
    }

    static void launch(AttentionParams& p, const DecodeLaunchPlan& plan, cudaStream_t stream) {
        p.num_splits = static_cast<int>(plan.grid.z);
        if constexpr (HEAD_DIM <= 128) {
            if (plan.direct_output) {
                attn_decode_split_kv_mma_kernel<Traits, KV, HasMask, true>
                    <<<plan.grid, Traits::NUM_THREADS, 0, stream>>>(p);
                ASTRAI_LAUNCH_CHECK();
                return;
            }
        }
        attn_decode_split_kv_mma_kernel<Traits, KV, HasMask>
            <<<plan.grid, Traits::NUM_THREADS, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
        attn_decode_combine_kernel<KV><<<p.batch * p.q_head, HEAD_DIM, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
};

// One head-dimension switch serves torch entries, native launches and plan inspection.
template <typename Fn> inline auto with_head_dim(int head_dim, Fn&& fn) {
    switch (head_dim) {
#define ASTRAI_HEAD_DIM_CASE(D) case D: return fn(std::integral_constant<int, D>{});
        ASTRAI_ATTN_HEAD_DIMS(ASTRAI_HEAD_DIM_CASE)
#undef ASTRAI_HEAD_DIM_CASE
    }
    throw std::runtime_error(head_dim_error(head_dim));
}

template <typename QSchedule, typename KV, typename Fn>
inline auto with_prefill_kernel(const AttentionParams& p, Fn&& fn) {
    // Paged partial masks need extent checks; full coverage can omit them.
    // Dense keeps its checked kernel to avoid redundant compiler-generated comparisons.
    const bool mask_covers_shape =
        KV::kPaged && p.mask && p.mask_k_len >= KV::host_kv_len(p) &&
        (p.mask_q_len == 1 || p.mask_q_len >= p.q_len);
    return with_head_dim(p.head_dim, [&](auto dim) {
        constexpr int D = decltype(dim)::value;
        if (p.is_causal) {
            if (p.mask) {
                if constexpr (KV::kPaged) {
                    if (mask_covers_shape)
                        return fn(PrefillKernel<D, QSchedule, KV, true, true, true>{});
                }
                return fn(PrefillKernel<D, QSchedule, KV, true, true>{});
            }
            return fn(PrefillKernel<D, QSchedule, KV, true, false>{});
        }
        if (p.mask) {
            if constexpr (KV::kPaged) {
                if (mask_covers_shape)
                    return fn(PrefillKernel<D, QSchedule, KV, false, true, true>{});
            }
            return fn(PrefillKernel<D, QSchedule, KV, false, true>{});
        }
        return fn(PrefillKernel<D, QSchedule, KV, false, false>{});
    });
}

template <typename KV, typename Fn>
inline auto with_decode_kernel(const AttentionParams& p, Fn&& fn) {
    return with_head_dim(p.head_dim, [&](auto dim) {
        constexpr int D = decltype(dim)::value;
        // Keep scalar new-token loads out of the aligned kernel's unrolled loader.
        if constexpr (KV::kPaged) {
            if (!KV::new_kv_aligned(p)) {
                using UnalignedKV = PagedKV<typename KV::Elem, false>;
                if (p.mask)
                    return fn(DecodeKernel<D, UnalignedKV, true>{});
                return fn(DecodeKernel<D, UnalignedKV, false>{});
            }
        }
        // A single right-aligned query needs no independent causal specialization.
        if (p.mask)
            return fn(DecodeKernel<D, KV, true>{});
        return fn(DecodeKernel<D, KV, false>{});
    });
}

template <typename QSchedule, typename KV> struct PrefillDispatch {
    static void run(AttentionParams& p, cudaStream_t stream) {
        with_prefill_kernel<QSchedule, KV>(p, [&](auto kernel) {
            const auto plan = kernel.plan(p);
            kernel.launch(p, plan, stream);
        });
    }
};

template <typename KV> struct DecodeDispatch {
    static void run(AttentionParams& p, cudaStream_t stream) {
        with_decode_kernel<KV>(p, [&](auto kernel) {
            const auto plan = kernel.plan(p);
            kernel.launch(p, plan, stream);
        });
    }
};

template <typename T> using AttnDispatchPrefill = PrefillDispatch<DenseQSchedule, ContigKV<T>>;
template <typename T> using AttnDispatchPagedPrefill = PrefillDispatch<PackedQSchedule, PagedKV<T>>;
template <typename T> using AttnDispatchDecode = DecodeDispatch<ContigKV<T>>;
template <typename T> using AttnDispatchPagedDecode = DecodeDispatch<PagedKV<T>>;

} // namespace attention
} // namespace astrai
