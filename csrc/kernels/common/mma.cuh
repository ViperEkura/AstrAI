// Shared mma.sync wrappers — pure CUDA, no torch.
//
// One template for every tensor-core MMA used by the kernel families. The
// instruction shape follows from the input element type:
//   __nv_bfloat16        -> mma.sync.aligned.m16n8k16 (sm_80+), A = 4x b32, B = 2x b32
//   __nv_fp8_e4m3/e5m2   -> mma.sync.aligned.m16n8k32 (sm_89+), A = 4x b32, B = 2x b32
// All variants accumulate into fp32: d = a*b + c, with the PTX mnemonic and
// the K dimension differing per type. `d` may alias `c` (in-place accumulate,
// as the FP8 GEMM does).

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>


#define DEVICE_FORCEINLINE static __device__ __forceinline__

namespace astrai {

// Compute capability of the current compilation pass: 0 in the host pass,
// the numeric CC (e.g. 890) in device passes where __CUDA_ARCH__ is defined.
// Defined() cannot appear in expressions, so this macro lets mma_sync use
// the arch in a static_assert instead of per-branch #if guards.
#ifndef __CUDA_ARCH__
#define ASTRAI_DEVICE_ARCH 0
#else
#define ASTRAI_DEVICE_ARCH __CUDA_ARCH__
#endif

// Compile-time shape of the MMA instruction for an input element type.
// `min_arch` is the numeric compute capability the instruction requires —
// the single place that encodes the hardware floor for each type.
template <typename InT>
struct mma_shape {
    static constexpr int k = 16;         // m16n8k16
    static constexpr int a_regs = 4;     // A fragment: 4x b32
    static constexpr int b_regs = 2;     // B fragment: 2x b32
    static constexpr int min_arch = 800; // bf16 mma.sync, sm_80+
};

template <>
struct mma_shape<__nv_fp8_e4m3> {
    static constexpr int k = 32;         // m16n8k32
    static constexpr int a_regs = 4;
    static constexpr int b_regs = 2;
    static constexpr int min_arch = 890; // fp8 mma.sync, sm_89+ (Ada/Hopper)
};

template <>
struct mma_shape<__nv_fp8_e5m2> {
    static constexpr int k = 32;
    static constexpr int a_regs = 4;
    static constexpr int b_regs = 2;
    static constexpr int min_arch = 890;
};

// d[4] = a[4] x b[2] + c[4], row-major A, col-major B, fp32 accumulator.
// The PTX mnemonic is selected from InT. Building for a compute capability
// below `mma_shape<InT>::min_arch` is a **compile error** — the instruction
// does not exist there, and a silent no-op would produce wrong results.
template <typename InT>
DEVICE_FORCEINLINE void mma_sync(float d[4], const unsigned a[4],
                                 const unsigned b[2],
                                 const float c[4]) {
    static_assert(ASTRAI_DEVICE_ARCH == 0 ||
                      ASTRAI_DEVICE_ARCH >= mma_shape<InT>::min_arch,
                  "mma_sync: this MMA shape requires a newer compute "
                  "capability than the build target");
    if constexpr (std::is_same_v<InT, __nv_bfloat16>) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else if constexpr (std::is_same_v<InT, __nv_fp8_e5m2>) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    }
}

#undef ASTRAI_DEVICE_ARCH

// ---------------------------------------------------------------------------
// ldmatrix — cooperatively load 8x8 b16 matrices from smem into registers.
//
// The instruction is identical for every 16-bit-storage element type: bf16
// maps 1:1 onto b16 slots; fp8 is stored packed two-per-slot (see
// gemm/gemm.cuh), so one b16 slot holds two fp8 values. `T` is the element
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
// 128-byte strides. fp8 fragment layouts in gemm/gemm.cuh are arranged around
// this constraint.
// ---------------------------------------------------------------------------

// Per-lane-address cores: the caller supplies a raw shared-memory address
// per lane instead of one common pointer. Use when the fragment tiles are
// XOR-swizzled per 16B chunk so each lane must compute its own row and
// chunk address (see gemm/gemm.cuh's frag_addr + lane selectors for the
// m16n8k32 operand layouts). Trans selects the transposed load — the
// gemm's crosswise 16-bit staging ([K][rows] tiles) reads its fragments
// through it. (ldmatrix is a b16-only instruction: 8-bit crosswise
// operands keep the PRMT staging + plain loads.)
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
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
            : "r"(addr));
    }
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

}  // namespace astrai
