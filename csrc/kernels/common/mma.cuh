// Shared mma.sync wrappers — pure CUDA, no torch.
//
// The instruction vocabulary is assembled from two trait layers (the
// humming codegen's compile-time format, hand-written):
//
//   MmaShapeFor<Dtype> — instruction shape per input dtype. The primary
//      template is UNDEFINED: a dtype with no tensor-core MMA is a compile
//      error at the use site, not a silent fallback. The K extent follows
//      the 256-bit A-fragment invariant (16B per lane row): bf16 k16, the
//      1-byte dtypes k32. kMinArch encodes each instruction's hardware
//      floor — the one place the requirement lives. (humming also keys on
//      an Arch tag for its sm_75 Turing thin-instruction correction; every
//      AstrAI target is sm_80+, where one shape serves all archs, so the
//      arch dimension collapses into kMinArch's build-time assert.)
//
//   MmaOp<A, B, Shape> — one specialization per instantiated
//      <dtype-pair, shape> cell: accumulator type, register counts and the
//      dedicated asm block. Mixed dtype pairs never reach the tensor core
//      (the gemm promotes both sides to MmaT first), so every cell is
//      symmetric; the A/B parameters keep the pairing explicit.
//
// All floating-point variants accumulate into fp32; the s8 pair accumulates
// into s32 (satfinite clamps the wrap that all-max-magnitude K~16k inputs
// could reach — the standard production int8-GEMM semantics). `d` may alias
// `c` (in-place accumulate, as the FP8 GEMM does).

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#include <cuda_runtime.h>
#include <type_traits>

#include "shape.cuh"
#include "tensor.cuh"

#define DEVICE_FORCEINLINE static __device__ __forceinline__

namespace astrai {

// Compute capability of the current compilation pass: 0 in the host pass,
// the numeric CC (e.g. 890) in device passes where __CUDA_ARCH__ is defined.
// Defined() cannot appear in expressions, so this macro lets the mma ops use
// the arch in a static_assert instead of per-branch #if guards.
#ifndef __CUDA_ARCH__
#define ASTRAI_DEVICE_ARCH 0
#else
#define ASTRAI_DEVICE_ARCH __CUDA_ARCH__
#endif

// Family (sm_100f/sm_120f/...) arch of the current pass, same trick: 0 in
// the host pass and on plain (non-'f', non-'a') targets, else the family
// CC (e.g. 1200). The value is usable in device expressions — the mx cell
// keys its asm on it (#if) and any cell can static_assert it.
#ifndef __CUDA_ARCH_FAMILY_SPECIFIC__
#define ASTRAI_ARCH_FAMILY 0
#else
#define ASTRAI_ARCH_FAMILY __CUDA_ARCH_FAMILY_SPECIFIC__
#endif

// --- <Dtype> -> instruction shape ------------------------------------------
// Primary template undefined: illegal <Dtype> combinations fail at compile
// time. Specializations spell one Shape<M, N, K> each.
template <typename Dtype>
struct MmaShapeFor;

template <>
struct MmaShapeFor<__nv_bfloat16> {
    using type = Shape<16, 8, 16>;   // f16/bf16 family: 256b / 16 bits
    static constexpr int kMinArch = 800;
};

template <>
struct MmaShapeFor<int8_t> {
    using type = Shape<16, 8, 32>;   // s8: 256b / 8 bits (sm_80 wide form)
    static constexpr int kMinArch = 800;
};

template <>
struct MmaShapeFor<__nv_fp8_e4m3> {
    using type = Shape<16, 8, 32>;
    static constexpr int kMinArch = 890;  // fp8 mma.sync, sm_89+ (Ada/Hopper)
};

template <>
struct MmaShapeFor<__nv_fp8_e5m2> {
    using type = Shape<16, 8, 32>;
    static constexpr int kMinArch = 890;
};

// --- <A, B, Shape> -> the mma op -------------------------------------------
// Primary template undefined: only the instantiated cells below exist.
//
// Each cell names its register cells as types (humming's ARegisters /
// BRegisters / CRegisters role): AFrag/BFrag/CFrag over common/tensor.cuh's
// ArrayEngine (cute's Array). The typed fma overload takes fragments BY
// REFERENCE, so
// fragment tensors index by semantic coordinates and no pointer arithmetic
// survives at the mma seam; the raw-array fma stays the core (the
// attention kernels' mma_sync and the C tests build on it).
template <typename A, typename B, typename ShapeT>
struct MmaOp;

template <>
struct MmaOp<__nv_bfloat16, __nv_bfloat16, Shape<16, 8, 16>> {
    using AccT = float;               // the accumulator type is derived, too
    static constexpr int kARegs = 4;  // A fragment: 4x b32
    static constexpr int kBRegs = 2;  // B fragment: 2x b32
    static constexpr int kCRegs = 4;  // C/D fragment: 4x f32
    using AFrag = ArrayEngine<unsigned, kARegs>;
    using BFrag = ArrayEngine<unsigned, kBRegs>;
    using CFrag = ArrayEngine<AccT, kCRegs>;
    DEVICE_FORCEINLINE void fma(CFrag& d, const AFrag& a, const BFrag& b,
                                const CFrag& c) {
        fma(d.storage, a.storage, b.storage, c.storage);
    }
    DEVICE_FORCEINLINE void fma(float d[4], const unsigned a[4],
                                       const unsigned b[2], const float c[4]) {
        static_assert(ASTRAI_DEVICE_ARCH == 0 || ASTRAI_DEVICE_ARCH >= 800,
                      "bf16 mma.sync requires sm_80+");
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    }
};

// The fp8 cells are sm_89+ instructions. fma is a template keyed on the
// pass arch so the kMinArch static_assert fires only when the cell is
// CALLED: member bodies of full specializations are checked in every
// including TU, and a bare assert would trip uncalled on the sm_80 passes
// (attention includes this header for the bf16 cell and ldmatrix).
template <>
struct MmaOp<__nv_fp8_e4m3, __nv_fp8_e4m3, Shape<16, 8, 32>> {
    using AccT = float;
    static constexpr int kARegs = 4;
    static constexpr int kBRegs = 2;
    static constexpr int kCRegs = 4;
    using AFrag = ArrayEngine<unsigned, kARegs>;
    using BFrag = ArrayEngine<unsigned, kBRegs>;
    using CFrag = ArrayEngine<AccT, kCRegs>;
    template <int Arch = ASTRAI_DEVICE_ARCH>
    DEVICE_FORCEINLINE void fma(CFrag& d, const AFrag& a, const BFrag& b,
                                const CFrag& c) {
        fma<Arch>(d.storage, a.storage, b.storage, c.storage);
    }
    template <int Arch = ASTRAI_DEVICE_ARCH>
    DEVICE_FORCEINLINE void fma(float d[4], const unsigned a[4],
                                       const unsigned b[2], const float c[4]) {
        static_assert(Arch == 0 || Arch >= 890,
                      "fp8 mma.sync requires sm_89+");
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    }
};

template <>
struct MmaOp<__nv_fp8_e5m2, __nv_fp8_e5m2, Shape<16, 8, 32>> {
    using AccT = float;
    static constexpr int kARegs = 4;
    static constexpr int kBRegs = 2;
    static constexpr int kCRegs = 4;
    using AFrag = ArrayEngine<unsigned, kARegs>;
    using BFrag = ArrayEngine<unsigned, kBRegs>;
    using CFrag = ArrayEngine<AccT, kCRegs>;
    template <int Arch = ASTRAI_DEVICE_ARCH>
    DEVICE_FORCEINLINE void fma(CFrag& d, const AFrag& a, const BFrag& b,
                                const CFrag& c) {
        fma<Arch>(d.storage, a.storage, b.storage, c.storage);
    }
    template <int Arch = ASTRAI_DEVICE_ARCH>
    DEVICE_FORCEINLINE void fma(float d[4], const unsigned a[4],
                                       const unsigned b[2], const float c[4]) {
        static_assert(Arch == 0 || Arch >= 890,
                      "fp8 mma.sync requires sm_89+");
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    }
};

template <>
struct MmaOp<int8_t, int8_t, Shape<16, 8, 32>> {
    using AccT = int32_t;
    static constexpr int kARegs = 4;
    static constexpr int kBRegs = 2;
    static constexpr int kCRegs = 4;
    using AFrag = ArrayEngine<unsigned, kARegs>;
    using BFrag = ArrayEngine<unsigned, kBRegs>;
    using CFrag = ArrayEngine<AccT, kCRegs>;
    DEVICE_FORCEINLINE void fma(CFrag& d, const AFrag& a, const BFrag& b,
                                const CFrag& c) {
        fma(d.storage, a.storage, b.storage, c.storage);
    }
    DEVICE_FORCEINLINE void fma(int32_t d[4], const unsigned a[4],
                                       const unsigned b[2], const int32_t c[4]) {
        static_assert(ASTRAI_DEVICE_ARCH == 0 || ASTRAI_DEVICE_ARCH >= 800,
                      "s8 mma.sync requires sm_80+");
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};

// --- full-rate fp8 via block_scale (sm_120 family) -------------------------
// The plain fp8 mma.sync decodes at half rate on sm_120 (measured issue
// rate 506 vs 1011 TFLOPS, RTX 5090); kind::mxf8f6f4 block_scale runs full
// rate with an identical A/B/C/D register contract and constant unit scales
// (every ue8m0 byte 0x7f = 2^0, selectors inert — the scale-factored
// product IS the plain product). Warp-level block_scale is sm_120-family
// only (100a/103a/110a reject it; datacenter Blackwell does MX through
// tcgen05), so fma dispatches on the family pass (the same call-time
// template trick as the plain fp8 cells) and falls back to the plain cell.
template <typename InT>
struct MxMmaOp;

template <>
struct MxMmaOp<__nv_fp8_e4m3> {
    using AccT = float;
    static constexpr int kARegs = 4;
    static constexpr int kBRegs = 2;
    static constexpr int kCRegs = 4;
    static constexpr int kMinArch = 1200;  // block_scale mxf8f6f4, sm_120 family
    using AFrag = ArrayEngine<unsigned, kARegs>;
    using BFrag = ArrayEngine<unsigned, kBRegs>;
    using CFrag = ArrayEngine<AccT, kCRegs>;
    template <int Family = ASTRAI_ARCH_FAMILY>
    DEVICE_FORCEINLINE void fma(CFrag& d, const AFrag& a, const BFrag& b,
                                const CFrag& c) {
        fma<Family>(d.storage, a.storage, b.storage, c.storage);
    }
    template <int Family = ASTRAI_ARCH_FAMILY>
    DEVICE_FORCEINLINE void fma(float d[4], const unsigned a[4],
                                       const unsigned b[2], const float c[4]) {
        if constexpr (Family >= 1200) {
            constexpr uint32_t sf_one = 0x7f7f7f7fu;  // ue8m0 1.0 x4
            const uint16_t sel = 0;
            asm volatile(
                "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
                "{%14}, {%15,%16}, {%17}, {%18,%19};"
                : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                  "r"(b[1]),
                  "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
                  "r"(sf_one), "h"(sel), "h"(sel), "r"(sf_one), "h"(sel),
                  "h"(sel));
        } else {
            MmaOp<__nv_fp8_e4m3, __nv_fp8_e4m3, Shape<16, 8, 32>>::fma(
                d, a, b, c);
        }
    }
};

template <>
struct MxMmaOp<__nv_fp8_e5m2> {
    using AccT = float;
    static constexpr int kARegs = 4;
    static constexpr int kBRegs = 2;
    static constexpr int kCRegs = 4;
    static constexpr int kMinArch = 1200;
    using AFrag = ArrayEngine<unsigned, kARegs>;
    using BFrag = ArrayEngine<unsigned, kBRegs>;
    using CFrag = ArrayEngine<AccT, kCRegs>;
    template <int Family = ASTRAI_ARCH_FAMILY>
    DEVICE_FORCEINLINE void fma(CFrag& d, const AFrag& a, const BFrag& b,
                                const CFrag& c) {
        fma<Family>(d.storage, a.storage, b.storage, c.storage);
    }
    template <int Family = ASTRAI_ARCH_FAMILY>
    DEVICE_FORCEINLINE void fma(float d[4], const unsigned a[4],
                                       const unsigned b[2], const float c[4]) {
        if constexpr (Family >= 1200) {
            constexpr uint32_t sf_one = 0x7f7f7f7fu;
            const uint16_t sel = 0;
            asm volatile(
                "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                "m16n8k32.row.col.f32.e5m2.e5m2.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
                "{%14}, {%15,%16}, {%17}, {%18,%19};"
                : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                  "r"(b[1]),
                  "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
                  "r"(sf_one), "h"(sel), "h"(sel), "r"(sf_one), "h"(sel),
                  "h"(sel));
        } else {
            MmaOp<__nv_fp8_e5m2, __nv_fp8_e5m2, Shape<16, 8, 32>>::fma(
                d, a, b, c);
        }
    }
};

// --- convenience views over the trait layers --------------------------------

// Compile-time facts of an input type's MMA, flattened for the layers that
// want plain ints (the attention kernels' KD/KT2 math, the gemm policy's
// register budgets). Derives entirely from the two traits above.
template <typename InT>
struct mma_shape {
    using Op = MmaOp<InT, InT, typename MmaShapeFor<InT>::type>;
    static constexpr int k = MmaShapeFor<InT>::type::kK;
    static constexpr int a_regs = Op::kARegs;
    static constexpr int b_regs = Op::kBRegs;
    static constexpr int min_arch = MmaShapeFor<InT>::kMinArch;
};

// d[4] = a[4] x b[2] + c[4], row-major A, col-major B — the fp32-accumulating
// family (bf16 / fp8 pairs). The s8 cell keeps its int32 accumulators and is
// reached through MmaOp directly. Building for a compute capability below
// MmaShapeFor<InT>::kMinArch is a **compile error** — the instruction does
// not exist there, and a silent no-op would produce wrong results.
template <typename InT>
DEVICE_FORCEINLINE void mma_sync(float d[4], const unsigned a[4],
                                 const unsigned b[2],
                                 const float c[4]) {
    static_assert(ASTRAI_DEVICE_ARCH == 0 ||
                      ASTRAI_DEVICE_ARCH >= MmaShapeFor<InT>::kMinArch,
                  "mma_sync: this MMA shape requires a newer compute "
                  "capability than the build target");
    MmaOp<InT, InT, typename MmaShapeFor<InT>::type>::fma(d, a, b, c);
}

#undef ASTRAI_DEVICE_ARCH

// ---------------------------------------------------------------------------
// ldmatrix — cooperatively load 8x8 b16 matrices from smem into registers.
//
// The instruction is identical for every 16-bit-storage element type: bf16
// maps 1:1 onto b16 slots; fp8 and s8 are stored packed two-per-slot (see
// gemm/mainloop.cuh), so one b16 slot holds two 1-byte values. `T` is the element
// type and only serves as a semantic tag.
//
//   x2 (single address): matrix0 = p (8 rows), matrix1 = p + 8*16 bytes
//   x4:                  four matrices at p, +128, +256, +384 bytes
//   Trans:               transpose variant (V fragments of attention)
//
// ldmatrix takes a *single* smem address per thread, but the addresses of
// the 32 lanes are *not* all the same: lane i supplies the start address of
// matrix-row i (modulo 8) for matrix (i/8) — lanes 0-7 feed matrix 0's rows,
// lanes 8-15 matrix 1's rows (x2/x4), lanes 16-23 / 24-31 matrix 2 / 3's rows
// (x4 only; their addresses are ignored by x2). Each matrix is 8 rows x 16
// bytes, and consecutive matrices of one instruction are contiguous at
// 128-byte strides. 1-byte fragment layouts in gemm/mainloop.cuh are arranged
// around this constraint.
// ---------------------------------------------------------------------------

// Per-lane-address cores: the caller supplies a raw shared-memory address
// per lane instead of one common pointer. Use when the fragment tiles are
// XOR-swizzled per 16B chunk so each lane must compute its own row and
// chunk address (see gemm/mainloop.cuh's a_lane_off / b_lane_off and the
// trans selectors for the m16n8k32 operand layouts). Trans selects the
// transposed load — the gemm's crosswise 16-bit staging ([K][rows] tiles)
// reads its fragments through it. (ldmatrix is a b16-only instruction:
// 8-bit crosswise operands keep the PRPT staging + plain loads.)
template <bool Trans = false>
DEVICE_FORCEINLINE void ldmatrix_x2_lane(unsigned r[2],
                                         unsigned addr) {
    if constexpr (Trans) {
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
            : "=r"(r[0]), "=r"(r[1])
            : "r"(addr));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(r[0]), "=r"(r[1])
                     : "r"(addr));
    }
}

template <bool Trans = false>
DEVICE_FORCEINLINE void ldmatrix_x4_lane(unsigned r[4],
                                         unsigned addr) {
    if constexpr (Trans) {
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
            : "r"(addr));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(addr));
    }
}

// Array-typed cores: the register cell IS the destination — fragment
// tensors hand their cells straight to the instruction, no decayed pointers
// at the seam. Same instructions, forwarding wrappers.
template <bool Trans = false>
DEVICE_FORCEINLINE void ldmatrix_x2_lane(ArrayEngine<unsigned, 2>& f,
                                         unsigned addr) {
    ldmatrix_x2_lane<Trans>(f.storage, addr);
}

template <bool Trans = false>
DEVICE_FORCEINLINE void ldmatrix_x4_lane(ArrayEngine<unsigned, 4>& f,
                                         unsigned addr) {
    ldmatrix_x4_lane<Trans>(f.storage, addr);
}

// Common-pointer wrappers over the per-lane cores (see the x2/x4 matrix
// layout notes above).
template <typename T, bool Trans = false>
DEVICE_FORCEINLINE void ldmatrix_x2(unsigned r[2], const T* p) {
    ldmatrix_x2_lane<Trans>(r, __cvta_generic_to_shared(p));
}

// Four matrices at p, p+128, p+256, p+384 bytes (16-byte row stride).
template <typename T, bool Trans = false>
DEVICE_FORCEINLINE void ldmatrix_x4(unsigned r[4], const T* p) {
    ldmatrix_x4_lane<Trans>(r, __cvta_generic_to_shared(p));
}


// ===========================================================================
// tcgen05 — 5th-gen tensor core (sm_100a/103a/110a/120a, Blackwell), the
// same instruction-vocabulary role as the mma.sync wrappers above but a
// different execution model: the MMA is issued by ONE thread, reads A/B
// from shared-memory descriptors, accumulates into TENSOR MEMORY (TMEM)
// instead of registers, and completion is observed through an mbarrier
// via tcgen05.commit (the discipline PipelineMbarrier carries).
//
// The PTX sites need an arch-specific sm_100+ target; declarations stay
// visible on every pass (ASTRAI_TCGEN05_ENABLED guards only the asm
// bodies) so __global__ templates can name them — non-Blackwell passes
// get no-op stubs that must never execute. Note sm_120a additionally
// needs CUDA 13.1+ to assemble (13.0's ptxas predates consumer tcgen05).
// Encoding constants adapted from humming's utils/ptx/tcgen05.cuh
// (BSD-3).
// ===========================================================================


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    defined(__CUDA_ARCH_SPECIFIC__)
#define ASTRAI_TCGEN05_ENABLED 1
#else
#define ASTRAI_TCGEN05_ENABLED 0
#endif
// tcgen05 needs an arch-specific ('a' suffix) sm_100+ target:
// __CUDA_ARCH_SPECIFIC__ is defined only for sm_100a/103a/110a/120a/...
// builds. Plain (non-'a') targets — including the production fatbin's
// sm_120 slice — get the inert stubs. Note sm_120a additionally needs
// CUDA 13.1+ to assemble (13.0's ptxas predates consumer tcgen05).

// --- allocation -----------------------------------------------------------
// Allocates kColumns TMEM columns (32-bit lanes, power of two in
// [32, 512]); the base TMEM address is written to the shared-memory slot
// `smem_addr`. Warp-wide (exactly one warp executes it); relinquish
// releases the CTA's allocation permit (we allocate once).
template <uint32_t kColumns>
__device__ __forceinline__ void tcgen05_alloc(uint32_t smem_addr) {
    static_assert(kColumns >= 32 && kColumns <= 512 &&
                      !(kColumns & (kColumns - 1)),
                  "TMEM columns must be a power of two in [32, 512]");
#if ASTRAI_TCGEN05_ENABLED
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        ::"r"(smem_addr), "n"(kColumns)
        : "memory");
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" :::
        "memory");
#else
    (void)smem_addr;
#endif
}

template <uint32_t kColumns>
__device__ __forceinline__ void tcgen05_dealloc(uint32_t tmem_addr) {
#if ASTRAI_TCGEN05_ENABLED
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 ::"r"(tmem_addr), "n"(kColumns)
                 : "memory");
#else
    (void)tmem_addr;
#endif
}

// --- ordering -------------------------------------------------------------
__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
#if ASTRAI_TCGEN05_ENABLED
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
#endif
}
__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
#if ASTRAI_TCGEN05_ENABLED
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#endif
}
__device__ __forceinline__ void tcgen05_wait_st() {
#if ASTRAI_TCGEN05_ENABLED
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#endif
}
__device__ __forceinline__ void tcgen05_wait_ld() {
#if ASTRAI_TCGEN05_ENABLED
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}
// Make generic-proxy shared writes (STS / cp.async) visible to the
// async proxy (the tensor core's smem view) before issuing an MMA.
__device__ __forceinline__ void fence_proxy_async_shared() {
#if ASTRAI_TCGEN05_ENABLED
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

// Completion: arrives on `mbar_addr` once every prior MMA of this thread
// is done — the TMEM-side handshake for the consumer's mbarrier wait.
__device__ __forceinline__ void tcgen05_commit(uint32_t mbar_addr) {
#if ASTRAI_TCGEN05_ENABLED
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster"
        ".b64 [%0];"
        ::"r"(mbar_addr)
        : "memory");
#else
    (void)mbar_addr;
#endif
}

// --- shared-memory matrix descriptor --------------------------------------
// K-major bf16, 64 elements (128B) per row, hardware 128B swizzle — the
// same physical layout Swizzle<3,3> stages (kTmaMode member), so the
// existing swizzled tiles feed tcgen05 unchanged. Encoded fields:
// [0:14) base>>4, [16:30) LBO (16B units), [32:46) SBO (16B units),
// bit 46 base-offset mode, [61:63) swizzle mode 2 (128B pattern).
__device__ __forceinline__ uint64_t tcgen05_smem_desc_bf16_k128(
    const void* ptr) {
#if ASTRAI_TCGEN05_ENABLED
    return ((uint64_t)(__cvta_generic_to_shared(ptr) >> 4) & 0x3fff) |
           (uint64_t(1) << 16) | (uint64_t(64) << 32) | (uint64_t(1) << 46) |
           (uint64_t(2) << 61);
#else
    (void)ptr;
    return 0;
#endif
}

// --- MMA ------------------------------------------------------------------
// C[m][n] (fp32, TMEM) += A[m][k] x B[n][k], bf16 operands from swizzled
// smem descriptors, K = 16 elements per issue (kind::f16). One thread
// issues; kM/kN are the MMA tile (M in {64,128} lanes, N a multiple of 8
// up to 256 columns). `accumulate` = false zero-inits D (loop peel).
// Instruction descriptor: bit4 D=fp32, bit7 A=bf16, bit10 B=bf16,
// [17:23) N/8, [24:29) M/16; transpose/LBO-mode fields zero (K-major).
template <uint32_t kM, uint32_t kN>
__device__ __forceinline__ void tcgen05_mma_bf16(uint32_t d_tmem,
                                                 uint64_t a_desc,
                                                 uint64_t b_desc,
                                                 bool accumulate) {
#if ASTRAI_TCGEN05_ENABLED
    constexpr uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                               ((kN / 8) << 17) | ((kM / 16) << 24);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %4, 0;\n"
        "  tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %5, "
        "%5, %5}, p;\n"
        "}"
        : : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
            "r"(uint32_t(accumulate)), "r"(0u)
        : "memory");
#else
    (void)d_tmem;
    (void)a_desc;
    (void)b_desc;
    (void)accumulate;
#endif
}

// --- TMEM read/write ------------------------------------------------------
// 32x32b: one warp reads 32 TMEM lanes (rows) x 32-bit columns; x8 = 8
// consecutive columns per lane. Lane l of warp w gets C[32w+l][n0..n0+7].
__device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t address,
                                                     uint32_t* values) {
#if ASTRAI_TCGEN05_ENABLED
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]),
          "=r"(values[3]), "=r"(values[4]), "=r"(values[5]),
          "=r"(values[6]), "=r"(values[7])
        : "r"(address)
        : "memory");
#else
    (void)address;
#endif
}

// 16x128b: one warp writes 16 lanes x 128 bits; x2 = two such rows.
__device__ __forceinline__ void tcgen05_st_16x128b_x2(uint32_t address,
                                                      const uint32_t* v) {
#if ASTRAI_TCGEN05_ENABLED
    asm volatile(
        "tcgen05.st.sync.aligned.16x128b.x2.b32 [%0], {%1, %2, %3, %4};"
        ::"r"(address), "r"(v[0]), "r"(v[2]), "r"(v[1]), "r"(v[3])
        : "memory");
#else
    (void)address;
    (void)v;
#endif
}


}  // namespace astrai

#undef DEVICE_FORCEINLINE
