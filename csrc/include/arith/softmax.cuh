/*
 * Shared online-softmax recurrence for split-KV and MMA paths. Scores and
 * running maxima stay pre-scale; scale*log2(e) is folded into exp2.
 *
 * Clamp the empty-state scaled max to 0 so masked -FLT_MAX scores weigh 0,
 * not exp2(0)=1. Fully masked rows therefore keep l=0 and normalize to 0.
 */

#pragma once

#include <cfloat>
#include <cuda_runtime.h>
#include <utils/define.cuh>

namespace astrai {
namespace attention {

// Flash-attention style running state: rescale-on-max (m, l) pair.
struct SoftmaxState {
    float m = -FLT_MAX;
    float l = 0.0f;
};

/* Scaled running max; clamp the empty-state sentinel (see above). */
DEVICE_FORCEINLINE float softmax_scaled_max(float m, float s2) {
    return (m == -FLT_MAX) ? 0.0f : m * s2;
}

/* Add the term with beta after rescaling by alpha; w=1 for keys, li for splits. */
DEVICE_FORCEINLINE void
softmax_step(SoftmaxState& s, float score, float w, float& alpha, float& beta, float s2) {
    float nm = fmaxf(s.m, score);
    alpha = exp2f((s.m - nm) * s2);
    beta = exp2f(score * s2 - softmax_scaled_max(nm, s2));
    s.l = s.l * alpha + w * beta;
    s.m = nm;
}

/* Subtract-first keeps corr=1 exactly when the raw max is unchanged. */
DEVICE_FORCEINLINE float softmax_remax(float& m, float cand, float& corr, float s2) {
    float nm = fmaxf(m, cand);
    corr = exp2f((m - nm) * s2);
    m = nm;
    return nm;
}

/* Warp-local row recurrence; mask policy supplies visibility, not addresses here. */
template <typename Traits> struct WarpSoftmax {
    using Layout = typename Traits::FragmentLayout;
    SoftmaxState rows[2];
    template <typename Mask>
    __device__ inline void update(int kv0, float scale_log2,
                                    typename Traits::ScoreFragment& Sacc,
                                    typename Traits::OutputFragment& Oacc,
                                    int lane, const Mask& mask) {
        float& m0 = rows[0].m;
        float& m1 = rows[1].m;
        float& l0 = rows[0].l;
        float& l1 = rows[1].l;

        int tid4 = lane & 3;

        float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
#pragma unroll
        for (int n8 = 0; n8 < Traits::NC8; n8++) {
            int cc = kv0 + n8 * Layout::kN + 2 * tid4;
            int c1 = cc + 1;
            bool b0 = mask.is_masked(0, cc);
            bool b1 = mask.is_masked(0, c1);
            bool b2 = mask.is_masked(1, cc);
            bool b3 = mask.is_masked(1, c1);
            float s0 = b0 ? -FLT_MAX : Sacc[n8][0];
            float s1 = b1 ? -FLT_MAX : Sacc[n8][1];
            float s2 = b2 ? -FLT_MAX : Sacc[n8][2];
            float s3 = b3 ? -FLT_MAX : Sacc[n8][3];
            Sacc[n8][0] = s0;
            Sacc[n8][1] = s1;
            Sacc[n8][2] = s2;
            Sacc[n8][3] = s3;
            rmax0 = fmaxf(rmax0, fmaxf(s0, s1));
            rmax1 = fmaxf(rmax1, fmaxf(s2, s3));
        }
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

        float corr0, corr1;
        float nm0 = softmax_remax(m0, rmax0, corr0, scale_log2);
        float nm1 = softmax_remax(m1, rmax1, corr1, scale_log2);

        /* Clamp the empty state's scaled max to 0. This keeps masked scores at
         * -FLT_MAX, so exp2 yields 0 rather than 1 for the softmax sentinel.
         */
        float nm2_0 = softmax_scaled_max(nm0, scale_log2);
        float nm2_1 = softmax_scaled_max(nm1, scale_log2);

        float rsum0 = 0.0f, rsum1 = 0.0f;
#pragma unroll
        for (int n8 = 0; n8 < Traits::NC8; n8++) {
            /* The compiler contracts x*s2 - nm2 into an FFMA feeding MUFU.EX2,
             * avoiding a separate scale multiply for each score.
             */
            float p0 = exp2f(Sacc[n8][0] * scale_log2 - nm2_0);
            float p1 = exp2f(Sacc[n8][1] * scale_log2 - nm2_0);
            float p2 = exp2f(Sacc[n8][2] * scale_log2 - nm2_1);
            float p3 = exp2f(Sacc[n8][3] * scale_log2 - nm2_1);
            Sacc[n8][0] = p0;
            Sacc[n8][1] = p1;
            Sacc[n8][2] = p2;
            Sacc[n8][3] = p3;
            rsum0 += p0 + p1;
            rsum1 += p2 + p3;
        }
        rsum0 += __shfl_xor_sync(0xFFFFFFFF, rsum0, 1);
        rsum0 += __shfl_xor_sync(0xFFFFFFFF, rsum0, 2);
        rsum1 += __shfl_xor_sync(0xFFFFFFFF, rsum1, 1);
        rsum1 += __shfl_xor_sync(0xFFFFFFFF, rsum1, 2);
        l0 = l0 * corr0 + rsum0;
        l1 = l1 * corr1 + rsum1;

        /* Skip O rescaling when the max is unchanged: corr is exactly 1.0f, so
         * multiplying by it would leave each value bit-identical.
         */
        if (corr0 != 1.0f) {
#pragma unroll
            for (int j = 0; j < Traits::DN8; j++) {
                Oacc[j][0] *= corr0;
                Oacc[j][1] *= corr0;
            }
        }
        if (corr1 != 1.0f) {
#pragma unroll
            for (int j = 0; j < Traits::DN8; j++) {
                Oacc[j][2] *= corr1;
                Oacc[j][3] *= corr1;
            }
        }
    }
};

} // namespace attention
} // namespace astrai
