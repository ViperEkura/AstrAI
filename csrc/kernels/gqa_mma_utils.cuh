#pragma once

// Shared MMA utilities for tensor-core GQA kernels.
// mma.sync.m16n8k16 PTX wrappers, ldmatrix helpers, and bf16 packing.

// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
__device__ __forceinline__ void mma16816(float* d, const unsigned* a,
                                          const unsigned* b, const float* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// read two adjacent bf16 from smem as one packed .b32 (elem0 low, elem1 high)
__device__ __forceinline__ unsigned ld2(const bf16* p) {
    return *reinterpret_cast<const unsigned*>(p);
}

// pack two floats into one bf16x2 as .b32
__device__ __forceinline__ unsigned pk2(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<unsigned*>(&v);
}

// pack two (non-contiguous) bf16 into one .b32
__device__ __forceinline__ unsigned pkb(bf16 a, bf16 b) {
    __nv_bfloat162 v;
    v.x = a;
    v.y = b;
    return *reinterpret_cast<unsigned*>(&v);
}

// ldmatrix: cooperatively load mma fragments from smem (one instruction per
// 16x16 / 16x8 tile) with the exact register layout mma expects — replaces the
// scalar per-thread fragment packing, cutting shared-load instructions and bank
// conflicts. Each lane supplies the shared address of one 8-wide row.
__device__ __forceinline__ void ldmatrix_x4(unsigned* r, const bf16* p) {
    unsigned a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2(unsigned* r, const bf16* p) {
    unsigned a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2_trans(unsigned* r, const bf16* p) {
    unsigned a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
}
