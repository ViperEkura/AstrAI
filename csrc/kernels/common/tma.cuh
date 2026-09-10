// TMA staging vocabulary (sm_90+): the device-side cp.async.bulk.tensor
// emitters with mbarrier completion, plus the host-side tensor-map
// encoding and its exact-match cache. Pure CUDA + driver API — no torch.
//
// The congruous operand staging this feeds is layout-identical to the
// cp.async rings it replaces: the staging swizzles are already the TMA
// hardware modes (Swizzle<3,3> = SWIZZLE_128B for 2-byte elements,
// Swizzle<2,3> = SWIZZLE_64B for 1-byte — see common/swizzle.cuh), so
// fragment addressing, the ring slots and the epilogue reclaim are all
// unchanged. Two TMA-specific facts the rest of the family respects:
//
//   * TMA applies its swizzle to the ABSOLUTE shared-memory address, so a
//     TMA-fed tile must sit on a 1024B boundary (the 128B-mode pattern
//     period); the kernel aligns the ring base up and budgets the pad.
//   * OOB coordinates zero-fill the box (rows past M/N, contract past K),
//     so the predication the cp.async zfill iterators computed per chunk
//     collapses into the descriptor's bounds — the k tail included.
//
// Host side: cuTensorMapEncodeTiled per (pointer, geometry, box) — a few
// microseconds — cached exact-match so steady-state serving loops (same
// activation buffer, same weight) pay it once. The driver symbol is
// dlsym'd, so neither the CMake link line nor the standalone C tests
// need -lcuda.
//
// All includes live at file scope OUTSIDE namespace astrai: standard
// headers must never be included inside a namespace (their std::
// declarations would nest and poison later inclusion).

#pragma once

#include <cstdint>
#include <dlfcn.h>
#include <mutex>

#include <cuda.h>
#include <cuda_runtime.h>

#include "swizzle.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define ASTRAI_TMA_ENABLED 1
#else
#define ASTRAI_TMA_ENABLED 0
#endif

namespace astrai {

// ---------------------------------------------------------------------------
// Device emitters (sm_90+). Declared unconditionally so both compilation
// passes can name them (the pipeline.cuh mbarrier pattern); the asm
// bodies compile away on pre-sm_90 passes, where they must never execute.
// ---------------------------------------------------------------------------

// One 2D tiled bulk copy: box (x runs the contract dim, y the row dim)
// from the descriptor into shared memory, completion counted on `bar`'s
// transaction bytes. Coordinates are in the map's element units — the
// host encodes byte-granular dims for operand tiles, so x is a byte
// offset along K.
__device__ __forceinline__ void tma_load_2d(const void* map, uint64_t* bar,
                                            void* smem_dst, int x, int y) {
#if ASTRAI_TMA_ENABLED
    const unsigned dst = __cvta_generic_to_shared(smem_dst);
    const unsigned bar_addr = __cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(map), "r"(x), "r"(y), "r"(bar_addr)
        : "memory");
#else
    (void)map; (void)bar; (void)smem_dst; (void)x; (void)y;
#endif
}

// 3D form: the batch is the outer map dimension (stride-0 broadcast
// operands encode as 2D and keep the 2D emitter).
__device__ __forceinline__ void tma_load_3d(const void* map, uint64_t* bar,
                                            void* smem_dst, int x, int y,
                                            int z) {
#if ASTRAI_TMA_ENABLED
    const unsigned dst = __cvta_generic_to_shared(smem_dst);
    const unsigned bar_addr = __cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(map), "r"(x), "r"(y), "r"(z), "r"(bar_addr)
        : "memory");
#else
    (void)map; (void)bar; (void)smem_dst; (void)x; (void)y; (void)z;
#endif
}

// Typed issue: the rank rides the type, so the per-stage 2D/3D pick is a
// compile-time branch instead of a runtime flag. `z` is dead in the rank-2
// form (broadcast operands share one 2D map's coordinates across grid.z).
template <bool kRank3>
__device__ __forceinline__ void tma_load(const void* map, uint64_t* bar,
                                         void* smem_dst, int x, int y, int z) {
    if constexpr (kRank3)
        tma_load_3d(map, bar, smem_dst, x, y, z);
    else
        tma_load_2d(map, bar, smem_dst, x, y);
}

// ---------------------------------------------------------------------------
// Host: tensor-map encode (driver API via dlsym) + exact-match cache.
// Unconditional (both passes parse host functions; the dlsym'd symbol
// only ever runs on the host).
// ---------------------------------------------------------------------------

// Map geometry one operand launch needs. Dim/stride/box units are BYTES
// along the contract dim (dtype CU_TENSOR_MAP_DATA_TYPE_UINT8): the box's
// inner extent is then exactly the swizzle span (128B for 2-byte
// elements, 64B for 1-byte), the constraint SWIZZLE_* imposes.
struct TmaMapSpec {
    const void* ptr = nullptr;
    uint64_t dim0 = 0;         // contract extent, bytes (K * elem size)
    uint64_t dim1 = 0;         // rows (M for A, N for B)
    uint64_t stride1 = 0;      // row stride, bytes (ld * elem size)
    uint32_t box0 = 0;         // stage tile contract extent, bytes (kK * es)
    uint32_t box1 = 0;         // stage tile rows (kBlockM / kBlockN)
    uint32_t batch = 1;        // >1 with a nonzero batch stride encodes rank 3
    uint64_t batch_stride = 0; // bytes; 0 broadcasts (rank 2, shared coords)
    int swizzle_bits = 0;      // 3 = SWIZZLE_128B (2B elems), 2 = SWIZZLE_64B (1B)

    bool aligned16() const {
        return ((reinterpret_cast<uintptr_t>(ptr) | stride1 |
                 (batch > 1 ? batch_stride : 0)) &
                15) == 0;
    }
};

// Staging-layout -> map facts: the swizzled staging tiles ARE hardware TMA
// modes (Swizzle<Bits, 3> members — kTmaMode marks them), so one trait over
// the declared ComposedLayout carries everything compile-time-derivable:
// the swizzle enum, and the box's inner extent — which SWIZZLE_* pins to
// the mode's span (16B << Bits). The staging layout is the single source:
// deriving the swizzle from sizeof(Elem) again (as the hand-built specs
// once did) can drift from what the fragments actually read.
template <typename StagedT>
struct TmaSwizzleOf;  // undefined: only swizzled congruous staging feeds TMA

template <typename SwzT, typename LayT>
struct TmaSwizzleOf<ComposedLayout<SwzT, LayT>> {
    static_assert(SwzT::kTmaMode,
                  "TMA staging needs a hardware swizzle mode (Swizzle<1-3, 3>)");
    static constexpr int kBits = SwzT::kBits;
    static constexpr CUtensorMapSwizzle kSwizzle =
        kBits == 3 ? CU_TENSOR_MAP_SWIZZLE_128B
                   : kBits == 2 ? CU_TENSOR_MAP_SWIZZLE_64B
                                : CU_TENSOR_MAP_SWIZZLE_NONE;
    static constexpr uint32_t kBox0Bytes = 16u << kBits;  // the swizzle span
};

// Host spec for one congruous staged operand: the compile-time facts (map
// dtype facts through the swizzle trait above, box rows through the tile)
// ride the type; only the runtime geometry (extents / strides / batch)
// fills the struct. `rows` is M for A, N for B.
template <typename ElemT, typename StagedT, int kBoxRows>
TmaMapSpec tma_spec(const void* ptr, int64_t rows, int64_t k, int64_t ld,
                    int batch, int64_t batch_stride) {
    using Swz = TmaSwizzleOf<StagedT>;
    static_assert(kBoxRows % 8 == 0, "box rows must tile the MMA m/n extent");
    TmaMapSpec s;
    s.ptr = ptr;
    s.dim0 = (uint64_t)k * sizeof(ElemT);
    s.dim1 = (uint64_t)rows;
    s.stride1 = (uint64_t)ld * sizeof(ElemT);
    s.box0 = Swz::kBox0Bytes;
    s.box1 = (uint32_t)kBoxRows;
    s.batch = (uint32_t)batch;
    s.batch_stride = (uint64_t)batch_stride * sizeof(ElemT);
    s.swizzle_bits = Swz::kBits;
    return s;
}

using TmaEncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t,
                                 void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*,
                                 CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

inline TmaEncodeFn tma_encode_fn() {
    static TmaEncodeFn fn = [] {
        void* handle = dlopen("libcuda.so.1", RTLD_LAZY);
        if (handle == nullptr) handle = dlopen("libcuda.so", RTLD_LAZY);
        return handle ? reinterpret_cast<TmaEncodeFn>(
                            dlsym(handle, "cuTensorMapEncodeTiled"))
                      : nullptr;
    }();
    return fn;
}

inline bool tma_encode(const TmaMapSpec& s, CUtensorMap* map) {
    const TmaEncodeFn fn = tma_encode_fn();
    if (fn == nullptr || !s.aligned16() || s.box1 == 0 || s.dim1 == 0)
        return false;
    const bool rank3 = s.batch > 1 && s.batch_stride > 0;
    const cuuint64_t dims[3] = {s.dim0, s.dim1,
                                rank3 ? (cuuint64_t)s.batch : 1};
    const cuuint64_t strides[2] = {s.stride1,
                                   rank3 ? s.batch_stride : (cuuint64_t)16};
    const cuuint32_t box[3] = {s.box0, s.box1, 1};
    const cuuint32_t elem_strides[3] = {1, 1, 1};
    const CUtensorMapSwizzle swz =
        s.swizzle_bits == 3
            ? CU_TENSOR_MAP_SWIZZLE_128B
            : s.swizzle_bits == 2 ? CU_TENSOR_MAP_SWIZZLE_64B
                                  : CU_TENSOR_MAP_SWIZZLE_NONE;
    // Box inner extent must equal the swizzle span (128B/64B): the staging
    // layouts are full-line swizzled.
    if ((swz == CU_TENSOR_MAP_SWIZZLE_128B && s.box0 != 128) ||
        (swz == CU_TENSOR_MAP_SWIZZLE_64B && s.box0 != 64))
        return false;
    const CUresult r = fn(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank3 ? 3 : 2,
                          const_cast<void*>(s.ptr), dims, strides, box,
                          elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return r == CUDA_SUCCESS;
}

// Exact-match descriptor cache: steady-state calls (same tensors, same
// tile) hit; a rotating buffer (dynamic M) cycles the small ring. The
// CUtensorMap is a POD the launcher copies into the kernel parameter.
class TmaMapCache {
  public:
    const CUtensorMap* lookup(const TmaMapSpec& s) {
        const std::lock_guard<std::mutex> lock(mu_);
        for (Entry& e : entries_)
            if (e.used && matches(e, s)) return &e.map;
        Entry& e = entries_[next_];
        if (!tma_encode(s, &e.map)) return nullptr;
        e.used = true;
        e.spec = s;
        next_ = (next_ + 1) % kCap;
        return &e.map;
    }

  private:
    static constexpr int kCap = 16;
    struct Entry {
        TmaMapSpec spec;
        CUtensorMap map;
        bool used = false;
    };
    static bool matches(const Entry& e, const TmaMapSpec& s) {
        return e.spec.ptr == s.ptr && e.spec.dim0 == s.dim0 &&
               e.spec.dim1 == s.dim1 && e.spec.stride1 == s.stride1 &&
               e.spec.box0 == s.box0 && e.spec.box1 == s.box1 &&
               e.spec.batch == s.batch &&
               e.spec.batch_stride == s.batch_stride &&
               e.spec.swizzle_bits == s.swizzle_bits;
    }
    Entry entries_[kCap];
    int next_ = 0;
    std::mutex mu_;
};

inline TmaMapCache& tma_map_cache() {
    static TmaMapCache cache;
    return cache;
}

}  // namespace astrai
