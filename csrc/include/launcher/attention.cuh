#pragma once

#include <algorithm>
#include <atomic>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <api/native/attention.h>
#include <kernel/attention/epilogue.cuh>
#include <kernel/attention/kernel.cuh>
#include <kernel/attention/mainloop.cuh>
#include <memory/layout_policies.cuh>
#include <utils/launch.cuh>

namespace astrai::attention {

// Convert the runtime head dimension into a compiled kernel specialization.
template <typename Fn>
inline auto with_head_dim(int head_dim, Fn&& fn) {
    switch (head_dim) {
    case 32:
        return fn(std::integral_constant<int, 32>{});
    case 64:
        return fn(std::integral_constant<int, 64>{});
    case 128:
        return fn(std::integral_constant<int, 128>{});
    case 256:
        return fn(std::integral_constant<int, 256>{});
    }
    throw std::runtime_error("ASTRAI: attention: head_dim " + std::to_string(head_dim) +
                             " has no kernel instantiation (instantiated: 32 64 128 256)");
}

// Fill at most one occupancy wave, with at least two KV tiles per split.
inline int compute_num_splits(int blocks, int tiles, int wave_capacity) {
    return std::max(1, std::min({wave_capacity / std::max(blocks, 1), tiles / 2, MAX_SPLITS}));
}

// Cache occupancy per device and kernel specialization.
template <auto Kernel, int Threads>
inline int decode_wave_capacity() {
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

// Resolve runtime flags to compile-time kernel arguments at the launch site.
#define ASTRAI_ATTN_BOOL_SWITCH(condition, name, ...)                                              \
    do {                                                                                           \
        if (condition) {                                                                           \
            constexpr bool name = true;                                                            \
            __VA_ARGS__;                                                                           \
        } else {                                                                                   \
            constexpr bool name = false;                                                           \
            __VA_ARGS__;                                                                           \
        }                                                                                          \
    } while (false)

template <typename QSchedule, typename KV>
inline void launch_attention_split_q(AttentionParams& p, cudaStream_t stream) {
    with_head_dim(p.head_dim, [&](auto dim) {
        constexpr int D = decltype(dim)::value;
        ASTRAI_ATTN_BOOL_SWITCH(p.is_causal, Causal, {
            ASTRAI_ATTN_BOOL_SWITCH(p.kv_split_size != 0, FixedSplits, {
                constexpr int BC =
                    FixedSplits ? 16 : (D == 256 ? 16 : (Causal && D <= 64 ? 64 : 32));
                using Traits = KernelTraits<D, BC, 4, 2, typename KV::Elem>;
                using Output = OutputEpilogue<Traits, true>;
                auto kernel = &attention_kernel<
                    PrefillMainloop<Traits, QSchedule, KV, Causal, false, false, FixedSplits>,
                    Output>;
                if (p.mask) {
                    kernel = &attention_kernel<
                        PrefillMainloop<Traits, QSchedule, KV, Causal, true, false, FixedSplits>,
                        Output>;
                    if constexpr (KV::kPaged) {
                        if (p.mask_k_size >= KV::host_kv_len(p) &&
                            (p.mask_q_size == 1 || p.mask_q_size >= p.q_len))
                            kernel =
                                &attention_kernel<PrefillMainloop<Traits, QSchedule, KV, Causal,
                                                                  true, true, FixedSplits>,
                                                  Output>;
                    }
                }
                constexpr int rows = Traits::BR * 4;
                const dim3 grid(QSchedule::packed_grid_x(p, rows, rows), p.kv_head,
                                QSchedule::host_grid_batch(p));
                kernel<<<grid, Traits::NUM_THREADS, 0, stream>>>(p);
                ASTRAI_LAUNCH_CHECK();
            });
        });
    });
}

template <typename KVSource, typename Allocate>
inline void launch_attention_split_kv(AttentionParams& p, cudaStream_t stream, Allocate& allocate) {
    bool aligned = true;
    if constexpr (KVSource::kPaged)
        aligned = KVSource::new_kv_aligned(p);
    with_head_dim(p.head_dim, [&](auto dim) {
        constexpr int D = decltype(dim)::value;
        ASTRAI_ATTN_BOOL_SWITCH(aligned, Aligned, {
            using KV = std::conditional_t<KVSource::kPaged,
                                          PagedKV<typename KVSource::Elem, Aligned>, KVSource>;
            ASTRAI_ATTN_BOOL_SWITCH(p.mask != nullptr, Mask, {
                using Traits = KernelTraits<D, 16, 1, 2, typename KV::Elem>;
                using Output = SplitEpilogue<Traits>;
                auto kernel = &attention_kernel<DecodeMainloop<Traits, KV, Mask>, Output>;
                auto combine = &split_kv_combine<KV, Traits::BC, false, false>;
                const int blocks = p.kv_head * ((p.q_head / p.kv_head + 15) / 16);
                const int length = KV::host_kv_len(p);
                const bool fixed = p.kv_split_size != 0;
                if (fixed) {
                    kernel =
                        &attention_kernel<DecodeMainloop<Traits, KV, Mask, false, true>, Output>;
                    combine = &split_kv_combine<KV, Traits::BC, false, true>;
                    p.num_splits = length > 0 ? (length - 1) / p.kv_split_size + 1 : 1;
                } else {
                    int capacity;
                    if (KV::kPaged && p.kv_len >= SKIP_EMPTY_SPLITS_MIN_CAPACITY) {
                        kernel = &attention_kernel<DecodeMainloop<Traits, KV, Mask, true>, Output>;
                        combine = &split_kv_combine<KV, Traits::BC, true, false>;
                        capacity = decode_wave_capacity<
                            attention_kernel<DecodeMainloop<Traits, KV, Mask, true>, Output>,
                            Traits::NUM_THREADS>();
                    } else {
                        capacity = decode_wave_capacity<
                            attention_kernel<DecodeMainloop<Traits, KV, Mask>, Output>,
                            Traits::NUM_THREADS>();
                    }
                    p.num_splits =
                        compute_num_splits(p.batch * blocks, (length + 15) / 16, capacity);
                }
                const bool direct = p.num_splits == 1 && (fixed || D <= 128);
                auto workspace = allocate(direct);
                const dim3 grid(blocks, p.batch, p.num_splits);
                if (direct) {
                    attention_kernel<DecodeMainloop<Traits, KV, Mask>, OutputEpilogue<Traits>>
                        <<<grid, Traits::NUM_THREADS, 0, stream>>>(p);
                } else {
                    kernel<<<grid, Traits::NUM_THREADS, 0, stream>>>(p);
                    ASTRAI_LAUNCH_CHECK();
                    combine<<<p.batch * p.q_head, D, 0, stream>>>(p);
                }
                ASTRAI_LAUNCH_CHECK();
            });
        });
    });
}

#undef ASTRAI_ATTN_BOOL_SWITCH

template <typename T>
inline void launch_attention(AttentionParams& p, cudaStream_t stream) {
    launch_attention_split_q<DenseQSchedule, ContigKV<T>>(p, stream);
}

template <typename T>
inline void launch_attention_varlen(AttentionParams& p, cudaStream_t stream) {
    launch_attention_split_q<VarlenQSchedule, VarlenKV<T>>(p, stream);
}

template <typename T, typename Allocate>
inline void
launch_attention_with_kvcache(AttentionParams& p, cudaStream_t stream, Allocate allocate) {
    if (p.cu_seqlens_q) {
        append_kv<T><<<p.q_len, 128, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
        return launch_attention_split_q<VarlenQSchedule, PagedKV<T>>(p, stream);
    }
    launch_attention_split_kv<PagedKV<T>>(p, stream, allocate);
}

} // namespace astrai::attention
