#pragma once

#include <arith/softmax.cuh>
#include <kernel/attention/mma.cuh>

namespace astrai::attention {

// Keep the extra split bookkeeping on large graph capacities only. The device
// skips partials when the replayed request uses under one eighth of the table.
constexpr int SKIP_EMPTY_SPLITS_MIN_CAPACITY = 8192;
constexpr int SKIP_EMPTY_SPLITS_CAPACITY_RATIO = 8;

DEVICE_FORCEINLINE bool skip_empty_decode_splits(const AttentionParams& p, int seq_len) {
    return seq_len < p.max_context_len / SKIP_EMPTY_SPLITS_CAPACITY_RATIO;
}

template <typename Traits> struct AttentionStorage {
    alignas(16) typename Traits::Elem k[Traits::STAGES * Traits::BC * Traits::LD];
    alignas(16) typename Traits::Elem v[Traits::STAGES * Traits::BC * Traits::LD];
};

template <typename Traits> struct AttentionTile {
    typename Traits::OutputFragment output;
    WarpSoftmax<Traits> softmax;
    AttentionRows rows;
    bool valid0, valid1;
};

// Compile-time composition: the mainloop computes a tile; the epilogue stores it.
template <typename Mainloop, typename Epilogue>
__global__ void attention_kernel(const AttentionParams p) {
    using Traits = typename Mainloop::Traits;
    __shared__ AttentionStorage<Traits> storage;
    AttentionTile<Traits> tile;
    if (Mainloop::run(p, storage, tile))
        Epilogue::store(p, tile);
}

} // namespace astrai::attention
