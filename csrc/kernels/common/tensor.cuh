// Tensor vocabulary after CUTLASS/cute's Tensor<Engine, Layout>: ONE tensor
// type — storage (engine) and addressing (layout) are its two template
// parameters, every operation dispatches to a layout op. Use sites spell
// Tensor<...> directly; no second names.
//
//   engines:  PtrEngine<T> (shared/global pointer), ArrayEngine<T, N>
//             (registers — cute's Array role: mma fragments are arrays)
//   layouts:  the ComposedLayout instances of common/swizzle.cuh (16B chunk
//             grids, dtype-agnostic) + RingLayout / CellLayout here
//   ops:      make_ring (construct over a raw carve), stage_of (slice one
//             ring slot's tile — cute's tensor slicing)
//
// A staged tile is Tensor<PtrEngine<Elem>, ComposedLayout>; the ring adds
// the slot dimension via RingLayout; the accumulator is
// Tensor<ArrayEngine<CFrag>, CellLayout>. All carriers are standard-layout
// types and every method folds away at -O3 — the SASS is unchanged.

#pragma once

#include <cstdint>

#include "swizzle.cuh"

namespace astrai {

// --- engines -----------------------------------------------------------------

// Shared/global-memory storage: the engine knows the element, the layout
// knows the address map.
template <typename T>
struct PtrEngine {
    using Elem = T;
    T* ptr;
    __device__ __forceinline__ T* base() const { return ptr; }
};

// Register-array storage (cute's Array role — an mma fragment cell IS an
// array). Passed BY REFERENCE to the mma/ldmatrix emitters so the
// registers stay in place — no address arithmetic can appear at the seams.
template <typename T, int N>
struct ArrayEngine {
    using Elem = T;
    T storage[N];
    __device__ __forceinline__ T& operator[](int i) { return storage[i]; }
    __device__ __forceinline__ const T& operator[](int i) const {
        return storage[i];
    }
    __device__ __forceinline__ T* base() { return storage; }
    __device__ __forceinline__ const T* base() const { return storage; }
};

// --- layouts (the tensor's address maps; chunk-grid ops are dtype-blind) -----

// Ring layout: slot rotation over a per-stage chunk grid — the staged
// ring expressed as one (slot, row, chunk) map instead of a bespoke ring
// object. Chunk units are 16B, so the byte budget is dtype-independent.
template <typename StageLay_, int kSlots_>
struct RingLayout {
    using Stage = StageLay_;
    static constexpr bool kChunkUnit = true;
    static constexpr int kSlots = kSlots_;
    static constexpr int kStageChunks = StageLay_::kRows * StageLay_::kChunks;
    static constexpr int kStageBytes = kStageChunks * 16;
    static constexpr int kTotalBytes = kSlots * kStageBytes;
    __device__ __forceinline__ uint32_t operator()(uint32_t slot, uint32_t row,
                                                   uint32_t chunk) const {
        return slot * (uint32_t)kStageChunks +
               StageLay_{}(row, chunk);
    }
};

// Cell layout: an element-unit row-major (m, n) grid — the accumulator's
// (mt, nt) mma-cell coordinates.
template <int kCols>
struct CellLayout {
    static constexpr bool kChunkUnit = false;
    __device__ __forceinline__ uint32_t operator()(uint32_t m,
                                                   uint32_t n) const {
        return m * (uint32_t)kCols + n;
    }
};

// --- the tensor ---------------------------------------------------------------

// cute's Tensor<Engine, Layout>: operator() dispatches to the layout op
// and indexes the engine — the tensor itself holds no address math.
// Chunk-unit layouts (ComposedLayout / RingLayout) speak 16B chunks, so
// the tensor applies its dtype's element scaling and keeps the LAST
// coordinate element-granular; the chunk-grid swizzle stays dtype-blind.
// Addresses come back as pointers (the smem seams feed cp.async /
// ldmatrix byte math); element-unit layouts (CellLayout) address whole
// engine cells.
template <typename EngineT, typename LayoutT>
struct Tensor {
    using Elem = typename EngineT::Elem;
    using Layout = LayoutT;
    static constexpr int kChunkElems =
        LayoutT::kChunkUnit ? 16 / (int)sizeof(Elem) : 1;

    EngineT engine;
    LayoutT layout;

    // Chunk-unit 2-coordinate tile view: the row term and the layout's
    // swizzled chunk scale separately in 32-BIT (one IMAD + one shift) and
    // widen once at the pointer add — a 64-bit multiply on this chain
    // regressed the crosswise-direct readers' register budget. The XOR
    // derives from the row ALONE (ComposedLayout's closed form) — it must
    // not serialize behind the row*stride IMAD (a linearized form
    // regressed W8A8 up to +29%; see docs/developer/cuda_kernels.md).
    template <bool kChunk = LayoutT::kChunkUnit,
              std::enable_if_t<kChunk, int> = 0>
    __device__ __forceinline__ Elem* operator()(int row, int col) const {
        constexpr int kShift = log2_const<kChunkElems>::value;
        const uint32_t off =
            (uint32_t)row * (uint32_t)(LayoutT::kChunks * kChunkElems) +
            (layout.chunk_of((uint32_t)row, (uint32_t)(col >> kShift))
             << kShift) +
            (uint32_t)(col & (kChunkElems - 1));
        return engine.base() + (ptrdiff_t)off;
    }
    // Chunk-unit 3-coordinate ring view: layout(slot, row, chunk).
    template <bool kChunk = LayoutT::kChunkUnit,
              std::enable_if_t<kChunk, int> = 0>
    __device__ __forceinline__ Elem* operator()(int slot, int row,
                                                int col) const {
        constexpr int kShift = log2_const<kChunkElems>::value;
        const typename LayoutT::Stage stage{};
        const uint32_t off =
            (uint32_t)slot *
                (uint32_t)(LayoutT::kStageChunks * kChunkElems) +
            (uint32_t)row *
                (uint32_t)(LayoutT::Stage::kChunks * kChunkElems) +
            (stage.chunk_of((uint32_t)row, (uint32_t)(col >> kShift))
             << kShift) +
            (uint32_t)(col & (kChunkElems - 1));
        return engine.base() + (ptrdiff_t)off;
    }
    // Element-unit cell view: layout(m, n) -> &engine cell. Non-const: the
    // accumulator rides non-const references through the mainloop/epilogue.
    template <bool kChunk = LayoutT::kChunkUnit,
              std::enable_if_t<!kChunk, int> = 0>
    __device__ __forceinline__ Elem* operator()(int m, int n) {
        return engine.base() + (size_t)layout((uint32_t)m, (uint32_t)n);
    }
};

// --- the tensor ops (factories + slicing, cute's make_tensor / slice role) --

// Construct the staged ring tensor over a raw shared-memory carve.
template <typename ElemT, typename StageLay, int kSlots>
__device__ __forceinline__ Tensor<PtrEngine<ElemT>,
                                 RingLayout<StageLay, kSlots>>
make_ring(char* smem) {
    return {PtrEngine<ElemT>{reinterpret_cast<ElemT*>(smem)}, {}};
}

// Slice one slot's tile out of the ring (slot = tile % kSlots) — cute's
// tensor slicing; .engine.ptr also serves the layout-agnostic writers
// (cp.async / TMA boxes stage through the raw address). The smem carve
// points and the TMA barrier placement measure against RingLayout's byte
// facts.
template <typename ElemT, typename StageLay, int kSlots>
__device__ __forceinline__ Tensor<PtrEngine<ElemT>, StageLay>
stage_of(const Tensor<PtrEngine<ElemT>, RingLayout<StageLay, kSlots>>& ring,
         int64_t tile) {
    return {PtrEngine<ElemT>{
                ring.engine.ptr +
                (size_t)(tile % kSlots) *
                    RingLayout<StageLay, kSlots>::kStageChunks *
                    (16 / (int)sizeof(ElemT))},
            {}};
}

}  // namespace astrai
