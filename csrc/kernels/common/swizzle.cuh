// Unified staging-swizzle vocabulary, CUTLASS-style: a swizzle is a TYPE
// composed with a layout into the staged tile's address map — "a bijection
// over the LINEAR 16B-chunk index, Swizzle<Bits,Shift>{}(L) = L ^
// ((L>>Shift)&(2^Bits-1))" applied to the row-major chunk layout. One
// declared instance per staged tile names the whole map (loaders, fragment
// readers and lane-offset mirrors consume the same type), so every staged
// tile is one (Bits, Shift) pair:
//
//   congruous tile 2B elems: <3,3> (TMA SWIZZLE_128B); 1B elems: <2,3> (64B);
//   crosswise trans tile:    <3, log2 chunks>; epilogue wide-row: <log2, log2>
//
// The Shift==3 members ARE the hardware TMA swizzle modes — kTmaMode marks
// them (the descriptor derives its enum from the same Bits; TMA applies the
// XOR to the ABSOLUTE smem address, so consumers align the tile to 1024B or
// phase by the tile's address bits). The vocabulary costs one IMAD + one XOR
// per use. Extent vocabulary (Shape/Stride) lives in shape.cuh.

#pragma once

#include <cstdint>

#include "shape.cuh"

namespace astrai {

template <int Bits, int Shift>
struct Swizzle {
    static_assert(Bits >= 0 && Shift >= 0 && Bits + Shift <= 16,
                  "swizzle field out of the 16B-chunk index range");
    static constexpr int kBits = Bits;
    static constexpr int kShift = Shift;
    static constexpr uint32_t kMask = (uint32_t(1) << Bits) - 1;
    static constexpr bool kTmaMode = Shift == 3 && Bits >= 1 && Bits <= 3;
    __device__ __forceinline__ uint32_t operator()(uint32_t linear) const {
        return linear ^ ((linear >> Shift) & kMask);
    }
};

// Layout carriers: Shape carries extents, Stride the affine map. All staged
// tiles are row-major packed 16B-chunk grids (row stride = chunk count).
template <int... Ns>
struct Stride;

template <typename ShapeT, typename StrideT>
struct Layout;

template <int Rows, int Chunks, int RowStride, int ColStride>
struct Layout<Shape<Rows, Chunks>, Stride<RowStride, ColStride>> {
    static_assert(RowStride == Chunks && ColStride == 1,
                  "staged chunk grids are row-major packed");
    static constexpr int kRows = Rows;
    static constexpr int kChunks = Chunks;
    __device__ __forceinline__ uint32_t operator()(uint32_t row,
                                                   uint32_t chunk) const {
        return row * Chunks + chunk;
    }
};

// composition(Swizzle, Layout): the swizzle bijection pre-folded to its
// closed two-coordinate form, chunk' = (chunk & ~kMask) | ((chunk ^
// (row>>kRowShift)) & kMask). The XOR derives from the row ALONE, so it
// computes in parallel with the chunk extraction instead of serializing
// behind the row*stride IMAD (CUTLASS 2.x iterators apply the swizzle the
// same way); row bits are narrower than the chunk field, so no carry.
template <typename SwzT, typename LayT>
struct ComposedLayout {
    using Swz = SwzT;
    using Lay = LayT;
    static constexpr bool kChunkUnit = true;  // tensor layer scales per dtype
    static constexpr int kRows = LayT::kRows;
    static constexpr int kChunks = LayT::kChunks;
    static_assert(kChunks >= 1 && (kChunks & (kChunks - 1)) == 0,
                  "the XOR swizzle needs a power-of-two chunk count");
    static_assert(SwzT::kBits <= log2_const<kChunks>::value,
                  "closed two-coordinate form needs the XOR inside the "
                  "chunk field");
    static constexpr int kRowShift = SwzT::kShift - log2_const<kChunks>::value;
    static_assert(kRowShift >= 0,
                  "swizzle source must start inside the row field");
    static constexpr uint32_t kMask = SwzT::kMask;
    // Swizzled chunk coordinate alone; the row term stays out so the tensor
    // scales the two terms separately in 32-bit (the address chain never
    // widens to 64-bit).
    __device__ __forceinline__ uint32_t chunk_of(uint32_t row,
                                                 uint32_t chunk) const {
        const uint32_t swz = (row >> kRowShift) & kMask;
        return (chunk & ~kMask) | ((chunk ^ swz) & kMask);
    }
    __device__ __forceinline__ uint32_t operator()(uint32_t row,
                                                   uint32_t chunk) const {
        return row * (uint32_t)kChunks + chunk_of(row, chunk);
    }
};

template <typename SwzT, typename LayT>
constexpr ComposedLayout<SwzT, LayT> composition(SwzT, LayT) {
    return {};
}

}  // namespace astrai
