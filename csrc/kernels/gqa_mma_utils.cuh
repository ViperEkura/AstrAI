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

// XOR swizzle for shared-memory column at 8-bf16 chunk granularity.
// Eliminates ldmatrix bank conflicts without LD padding: consecutive rows
// land in distinct bank groups. swiz_col(d, r, mask) = ((d>>3)^(r&mask))<<3 | (d&7).
// mask must cover log2(HEAD_DIM/8) chunk bits but stay within LD: use 7 for
// HEAD_DIM>=64 (8+ chunks), 3 for HEAD_DIM=32 (4 chunks). Default 7 keeps
// existing HEAD_DIM>=64 call sites working unchanged.
__device__ __forceinline__ int swiz_col(int d, int r, int mask = 7) {
    return ((d >> 3) ^ (r & mask)) << 3 | (d & 7);
}

// cp.async: copy 16 bytes (8 bf16) from global to shared memory directly,
// bypassing registers. Eliminates shared-store bank conflicts and cuts
// load-loop instruction count in half (1 cp.async vs 1 LDG + 1 STS).
// Requires sm_80+.
__device__ __forceinline__ void cp_async_16(bf16* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

// Wait until at most N commit groups are still in flight. Used for
// double-buffered pipelining: wait_group<1> lets the next tile's cp.async
// continue while ensuring the current tile's data is ready.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}
