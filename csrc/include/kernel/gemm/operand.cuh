#pragma once

#include <type_traits>

#include <memory/load_async.cuh>
#include <memory/load_crosswise.cuh>
#include <memory/load_crosswise_packed.cuh>
#include <policy/gemm.cuh>

namespace astrai {
namespace gemm {

// A and B differ in their row extent and which storage layout is crosswise.
// For either operand the same tile description drives the copy and the warp reader.
template <typename Policy, typename Element, int Rows, bool Crosswise, bool Dequant>
struct GemmOperand {
    using Traits = typename Policy::Traits;
    using Elem = Element;
    static constexpr int kRows = Rows;
    static constexpr int kTile = Traits::kTile;
    static constexpr int kDepth = Policy::Smem::kRingDepth;
    static constexpr int kThreads = Traits::kCtaThreads;
    static constexpr bool kRegisterLoad = Crosswise && sizeof(Elem) == 1;
    static constexpr bool kTranspose = Crosswise && sizeof(Elem) == 2;
    static_assert(!Dequant || sizeof(Elem) == 1);

    static constexpr int kChunks = kTile / (16 / sizeof(Elem));
    static constexpr int kTransChunks = Rows / (16 / sizeof(Elem));
    static constexpr int kPackChunks = Rows / 8;
    static constexpr bool kPacked = kRegisterLoad && !Dequant && kTile % 2 == 0 &&
                                    kPackChunks >= 8 && (kPackChunks & (kPackChunks - 1)) == 0;
    template <int Height, int Chunks>
    using TransposedLayout = decltype(composition(
        Swizzle<log2_const<(Chunks < 8 ? Chunks : 8)>::value, log2_const<Chunks>::value>{},
        Layout<Shape<Height, Chunks>, Stride<Chunks, 1>>{}));
    using Canonical = decltype(composition(Swizzle<log2_const<kChunks>::value, 3>{},
                                           Layout<Shape<Rows, kChunks>, Stride<kChunks, 1>>{}));
    using Transposed = TransposedLayout<kTile, kTransChunks>;
    using Packed = TransposedLayout<kTile / 2, kPackChunks>;
    using SharedLayout =
        std::conditional_t<kPacked, Packed, std::conditional_t<kTranspose, Transposed, Canonical>>;
    using Ring = Tensor<PtrEngine<Elem>, RingLayout<SharedLayout, kDepth>>;
    using Tile = Tensor<PtrEngine<Elem>, SharedLayout>;
    using RegisterCarry =
        std::conditional_t<kPacked,
                           PairPackCarry<SharedLayout, Elem, kThreads, true>,
                           CrosswiseCarry<SharedLayout, Elem, kThreads, kRegisterLoad>>;

    static constexpr int kBytes = Ring::Layout::kTotalBytes;
    static constexpr int kStageBytes = Ring::Layout::kStageBytes;

    static DEVICE_FORCEINLINE bool aligned(const Elem* src, int64_t ld) {
        // The leading dimension is in elements. Each BF16 row needs eight elements of alignment.
        return ((reinterpret_cast<uintptr_t>(src) | (uint64_t(ld) * sizeof(Elem))) & 15) == 0;
    }

    template <bool Interior>
    static DEVICE_FORCEINLINE void copy_async(Tile tile,
                                              const Elem* src,
                                              int64_t rows,
                                              int64_t k,
                                              int64_t ld,
                                              int tid,
                                              int64_t k_base,
                                              int64_t row0) {
        if constexpr (!kRegisterLoad)
            load_operand_tile<SharedLayout, Elem, kThreads, kTranspose, Interior>(
                tile, src, rows, k, ld, tid, k_base, row0);
    }

    // The prologue drains the same register carry that overlaps reads with MMA in the mainloop.
    static DEVICE_FORCEINLINE void prime_registers(Tile tile,
                                                   const Elem* src,
                                                   int64_t rows,
                                                   int64_t k,
                                                   int64_t ld,
                                                   int tid,
                                                   int64_t k_base,
                                                   int64_t row0) {
        if constexpr (kRegisterLoad) {
            RegisterCarry carry;
            carry.issue(src, rows, k, ld, tid, k_base, row0);
            carry.commit(tile, src, rows, k, ld, tid, k_base, row0);
        }
    }
    template <bool Async>
    using Prefetch = PrefetchCarry<Async && !kRegisterLoad, Ring, kThreads, kTranspose>;
};

template <typename Policy> struct GemmOperands {
    using T = typename Policy::Traits;
    using Smem = typename Policy::Smem;
    using A = GemmOperand<Policy, typename T::ElemA, T::kBlockM, Smem::kDirectA, T::kDequantA>;
    using B = GemmOperand<Policy, typename T::ElemB, T::kBlockN, Smem::kDirectB, T::kDequantB>;
};

} // namespace gemm
} // namespace astrai
