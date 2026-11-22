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

// Share the same rounding in the register and workspace combine paths.
DEVICE_FORCEINLINE void merge_softmax(SoftmaxState& total,
                                      const SoftmaxState& part,
                                      float& alpha,
                                      float& beta,
                                      float scale_log2) {
    if (total.l == 0.0f) {
        total = part;
        alpha = 0.0f;
        beta = 1.0f;
        return;
    }
    const float m = fmaxf(total.m, part.m);
    alpha = exp2f(__fmul_rn(__fsub_rn(total.m, m), scale_log2));
    beta = exp2f(__fmul_rn(__fsub_rn(part.m, m), scale_log2));
    total.l = __fmaf_rn(total.l, alpha, __fmul_rn(part.l, beta));
    total.m = m;
}

DEVICE_FORCEINLINE float merge_output(float total, float part, float alpha, float beta) {
    return __fmaf_rn(total, alpha, __fmul_rn(part, beta));
}

// Four neighboring lanes contribute columns to the same MMA row.
DEVICE_FORCEINLINE float quad_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 1);
    return x + __shfl_xor_sync(0xffffffff, x, 2);
}

enum class SumReduction { PerTile, Deferred };

template <typename Traits> struct AttentionAccumulator {
    SoftmaxState rows[2];
    typename Traits::OutputFragment output{};

    template <SumReduction Reduction = SumReduction::PerTile>
    DEVICE_FORCEINLINE void update_softmax(typename Traits::ScoreFragment& scores,
                                           float scale_log2, int row_count = 2) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r >= row_count) {
#pragma unroll
                for (int n = 0; n < Traits::NC8; ++n)
                    scores[n][2 * r] = scores[n][2 * r + 1] = 0.0f;
                continue;
            }
            auto& state = rows[r];
            float m = state.m;
#pragma unroll
            for (int n = 0; n < Traits::NC8; ++n)
                m = fmaxf(m, fmaxf(scores[n][2 * r], scores[n][2 * r + 1]));
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
            const float corr = exp2f((state.m - m) * scale_log2);
            state.m = m;
            const float scaled_max = softmax_scaled_max(m, scale_log2);
            float sum = 0.0f;
#pragma unroll
            for (int n = 0; n < Traits::NC8; ++n) {
                auto& x = scores[n][2 * r];
                auto& y = scores[n][2 * r + 1];
                x = exp2f(x * scale_log2 - scaled_max);
                y = exp2f(y * scale_log2 - scaled_max);
                sum += x + y;
            }
            if constexpr (Reduction == SumReduction::PerTile)
                sum = quad_sum(sum);
            state.l = state.l * corr + sum;
            if (corr != 1.0f) {
#pragma unroll
                for (int n = 0; n < Traits::DN8; ++n) {
                    output[n][2 * r] *= corr;
                    output[n][2 * r + 1] *= corr;
                }
            }
        }
    }

    // The running maximum is shared within a row, so lane sums can be combined once.
    DEVICE_FORCEINLINE void finish_sum() {
#pragma unroll
        for (int r = 0; r < 2; ++r)
            rows[r].l = quad_sum(rows[r].l);
    }

    DEVICE_FORCEINLINE void merge(const AttentionAccumulator& part, float scale_log2) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (part.rows[r].l == 0.0f)
                continue;
            float alpha, beta;
            merge_softmax(rows[r], part.rows[r], alpha, beta, scale_log2);
#pragma unroll
            for (int n = 0; n < Traits::DN8; ++n) {
#pragma unroll
                for (int i = 2 * r; i < 2 * r + 2; ++i)
                    output[n][i] = merge_output(output[n][i], part.output[n][i], alpha, beta);
            }
        }
    }
};

} // namespace attention
} // namespace astrai
