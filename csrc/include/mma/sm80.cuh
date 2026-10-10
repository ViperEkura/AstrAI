#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <utils/define.cuh>
#include <utils/shape.cuh>

namespace astrai::mma {

template <typename T> struct WarpMma;

template <> struct WarpMma<__nv_bfloat16> {
    using Shape = astrai::Shape<16, 8, 16>;
    using AccT = float;
    static constexpr int kMinArch = 800;

    static DEVICE_FORCEINLINE void
    fma(float d[4], const unsigned a[4], const unsigned b[2], const float c[4]) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                       "f"(c[1]), "f"(c[2]), "f"(c[3]));
    }
};

template <> struct WarpMma<int8_t> {
    using Shape = astrai::Shape<16, 8, 32>;
    using AccT = int32_t;
    static constexpr int kMinArch = 800;

    static DEVICE_FORCEINLINE void
    fma(int32_t d[4], const unsigned a[4], const unsigned b[2], const int32_t c[4]) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                     : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]),
                       "r"(c[1]), "r"(c[2]), "r"(c[3]));
    }
};

} // namespace astrai::mma
