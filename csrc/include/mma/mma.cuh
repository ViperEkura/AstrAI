#pragma once

#include <mma/sm80.cuh>
#include <mma/sm89.cuh>
#include <mma/sm120.cuh>
#include <utils/tensor.cuh>

namespace astrai {

#if defined(__CUDA_ARCH__)
inline constexpr int kMmaDeviceArch = __CUDA_ARCH__;
#else
inline constexpr int kMmaDeviceArch = 0;
#endif

template <typename T> struct MmaShapeFor : mma::WarpMma<T> {
    using type = typename mma::WarpMma<T>::Shape;
};

/* Accumulator mapping for the warp-wide m16n8 instruction family. */
struct Mma16x8Layout {
    static constexpr int kM = 16, kN = 8;
    static constexpr int kARegs = 4, kBRegs = 2, kCRegs = 4;
    static constexpr int kRowGroup = 8, kLaneGroup = 4;
    static DEVICE_FORCEINLINE int row(int lane, int half = 0) {
        return (lane >> log2_const<kLaneGroup>::value) + half * kRowGroup;
    }
    static DEVICE_FORCEINLINE int column(int lane, int pair = 0) {
        return (lane & (kLaneGroup - 1)) * 2 + pair;
    }
};

/* The warp MMA register contract shared by GEMM and attention. */
template <typename T> struct MmaOpImpl {
    using Traits = mma::WarpMma<T>;
    using AccT = typename Traits::AccT;
    using Shape = typename Traits::Shape;
    using Layout = Mma16x8Layout;
    static_assert(Shape::kM == Layout::kM && Shape::kN == Layout::kN,
                  "MMA atom requires a compatible fragment layout");
    static constexpr int kARegs = Layout::kARegs, kBRegs = Layout::kBRegs, kCRegs = Layout::kCRegs;
    using AFrag = ArrayEngine<unsigned, kARegs>;
    using BFrag = ArrayEngine<unsigned, kBRegs>;
    using CFrag = ArrayEngine<AccT, kCRegs>;

    static DEVICE_FORCEINLINE void fma(CFrag& d, const AFrag& a, const BFrag& b, const CFrag& c) {
        fma(d.storage, a.storage, b.storage, c.storage);
    }

    template <int Arch = kMmaDeviceArch>
    static DEVICE_FORCEINLINE void
    fma(AccT d[4], const unsigned a[4], const unsigned b[2], const AccT c[4]) {
        static_assert(Arch == 0 || Arch >= Traits::kMinArch,
                      "the build target does not support this warp MMA atom");
        Traits::fma(d, a, b, c);
    }
};

template <typename A, typename B, typename ShapeT> struct MmaOp;
template <typename T>
struct MmaOp<T, T, typename MmaShapeFor<T>::type> : MmaOpImpl<T> {};

template <typename T> struct MxMmaOp : MmaOpImpl<T> {
    using Base = MmaOpImpl<T>;
    using typename Base::AFrag;
    using typename Base::BFrag;
    using typename Base::CFrag;
    static constexpr int kMinArch = 1200;

    template <int Arch = kMmaDeviceArch>
    static DEVICE_FORCEINLINE void
    fma(CFrag& d, const AFrag& a, const BFrag& b, const CFrag& c) {
        fma<Arch>(d.storage, a.storage, b.storage, c.storage);
    }

    template <int Arch = kMmaDeviceArch>
    static DEVICE_FORCEINLINE void
    fma(float d[4], const unsigned a[4], const unsigned b[2], const float c[4]) {
        static_assert(Arch == 0 || mma::kSm120BlockScaled,
                      "block-scaled MMA requires an SM120 architecture or family target");
        mma::BlockScaledMma<T>::fma(d, a, b, c);
    }
};

template <typename T> struct mma_shape {
    static constexpr int k = MmaShapeFor<T>::type::kK;
};

template <typename T>
static DEVICE_FORCEINLINE void
mma_sync(float d[4], const unsigned a[4], const unsigned b[2], const float c[4]) {
    MmaOp<T, T, typename MmaShapeFor<T>::type>::fma(d, a, b, c);
}

} // namespace astrai
