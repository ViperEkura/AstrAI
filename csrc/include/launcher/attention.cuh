#pragma once

#include <kernel/attention/append.cuh>
#include <kernel/attention/dispatch.cuh>

namespace astrai::attention {

template <typename T>
inline void launch_attention(AttentionParams& p, cudaStream_t stream) {
    launch_prefill<DenseQSchedule, ContigKV<T>>(p, stream);
}

template <typename T>
inline void launch_attention_varlen(AttentionParams& p, cudaStream_t stream) {
    launch_prefill<VarlenQSchedule, VarlenKV<T>>(p, stream);
}

// The caller owns split storage, including workspace supplied for graph replay.
template <typename T, typename Decode>
inline void launch_attention_with_kvcache(AttentionParams& p, cudaStream_t stream, Decode decode) {
    if (p.qo_indptr) {
        append_kv<T><<<p.q_len, 128, 0, stream>>>(p);
        launch_prefill<CacheQSchedule, PagedKV<T>>(p, stream);
    } else {
        with_decode_kernel<PagedKV<T>>(p, [&](auto kernel) { decode(kernel, kernel.plan(p)); });
    }
}

} // namespace astrai::attention
