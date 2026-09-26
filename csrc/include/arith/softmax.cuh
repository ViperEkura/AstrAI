// Shared online-softmax recurrence — pure CUDA, no torch.
//
// One sentinel policy for every attention consumer: the scalar prefill and
// decode kernels, the split-KV combine, and the MMA tile softmax
// (mma_utils.cuh).  While the running max is still -FLT_MAX (no valid key
// seen yet), __expf(score - max) == __expf(0) == 1 would admit masked-out
// terms with weight 1 — a fully-masked row/split must stay l == 0 so it
// normalises to 0 instead of mean(V).

#pragma once

#include <cfloat>
#include <cuda_runtime.h>

namespace astrai {
namespace attention {

// Flash-attention style running state: rescale-on-max (m, l) pair.
struct SoftmaxState {
    float m = -FLT_MAX;
    float l = 0.0f;
};

// Advance the state with one scored term of weight `w` — a plain key uses
// w = 1; the split-KV combine merges a (mi, li) partial with w = li.
// `alpha` rescales the caller's accumulator carrying the OLD max, `beta`
// weights the new term:  acc = acc * alpha + x * beta;  l likewise.
__device__ __forceinline__ void softmax_step(
    SoftmaxState& s, float score, float w, float& alpha, float& beta) {
    float nm = fmaxf(s.m, score);
    alpha = __expf(s.m - nm);
    beta = (nm == -FLT_MAX) ? 0.0f : __expf(score - nm);
    s.l = s.l * alpha + w * beta;
    s.m = nm;
}

// Running-max advance for consumers that reduce a whole tile before taking
// the exp (the MMA path): returns the new max, writes the old-state rescale
// factor and the 0/1 gate that zeroes the exp() terms of an all-masked row.
__device__ __forceinline__ float softmax_remax(
    float& m, float cand, float& corr, float& pn) {
    float nm = fmaxf(m, cand);
    corr = __expf(m - nm);
    pn = (nm == -FLT_MAX) ? 0.0f : 1.0f;
    m = nm;
    return nm;
}

}  // namespace attention
}  // namespace astrai
