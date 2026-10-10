#pragma once

#include <cuda_fp8.h>
#include <mma/sm80.cuh>

namespace astrai::mma {

#define ASTRAI_FP8_MMA(Type, Format)                                                               \
    template <> struct WarpMma<Type> {                                                             \
        using Shape = astrai::Shape<16, 8, 32>;                                                    \
        using AccT = float;                                                                        \
        static constexpr int kMinArch = 890;                                                       \
        static DEVICE_FORCEINLINE void                                                             \
        fma(float d[4], const unsigned a[4], const unsigned b[2], const float c[4]) {              \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32." Format "." Format                \
                         ".f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "                            \
                         "{%10,%11,%12,%13};"                                                      \
                         : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])                          \
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),       \
                           "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));                            \
        }                                                                                          \
    };

ASTRAI_FP8_MMA(__nv_fp8_e4m3, "e4m3")
ASTRAI_FP8_MMA(__nv_fp8_e5m2, "e5m2")
#undef ASTRAI_FP8_MMA

} // namespace astrai::mma
