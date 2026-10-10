#pragma once

#include <cuda_fp8.h>
#include <utils/define.cuh>

namespace astrai::mma {

#if (defined(__CUDA_ARCH_FEAT_SM120_ALL) ||                                                        \
     (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))
inline constexpr bool kSm120BlockScaled = true;
#else
inline constexpr bool kSm120BlockScaled = false;
#endif

template <typename T> struct BlockScaledMma;

/* Unit UE8M0 scales preserve ordinary FP8 GEMM semantics. */
#define ASTRAI_MX_MMA(Type, Format)                                                                \
    template <> struct BlockScaledMma<Type> {                                                      \
        static DEVICE_FORCEINLINE void                                                             \
        fma(float d[4], const unsigned a[4], const unsigned b[2], const float c[4]) {              \
            constexpr unsigned one = 0x7f7f7f7fu;                                                  \
            constexpr unsigned short select = 0;                                                   \
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale."                            \
                         "scale_vec::1X.m16n8k32.row.col.f32." Format "." Format                   \
                         ".f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "                      \
                         "{%10,%11,%12,%13}, {%14}, {%15,%16}, {%17}, {%18,%19};"                  \
                         : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])                          \
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),       \
                           "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(one), "h"(select),      \
                           "h"(select), "r"(one), "h"(select), "h"(select));                       \
        }                                                                                          \
    };

ASTRAI_MX_MMA(__nv_fp8_e4m3, "e4m3")
ASTRAI_MX_MMA(__nv_fp8_e5m2, "e5m2")
#undef ASTRAI_MX_MMA

} // namespace astrai::mma
