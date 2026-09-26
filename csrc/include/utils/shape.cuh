// Static-geometry vocabulary (cute's Shape<> role), shared by every kernel
// family: one variadic type names one geometry, wherever a compile-time
// extent list is needed — the gemm policy's CTA tile Shape<M, N, K> and
// warp tile Shape<M, N>, the staging layouts' chunk grid Shape<Rows,
// Chunks>, the mma trait layer's instruction Shape<16, 8, 32>. Extracted
// from swizzle.cuh (which composes layouts over it) so the mma/policy/
// epilogue layers spell it without pulling the swizzle machinery in.

#pragma once

namespace astrai {

// log2 of a compile-time power of two.
template <int N, int Acc = 0>
struct log2_const : log2_const<(N >> 1), Acc + 1> {};
template <int Acc>
struct log2_const<1, Acc> {
    static constexpr int value = Acc;
};

// Static integer shape: kM/kN/kK read the leading extents (missing ones
// read 0), so tile recipes and instruction shapes share one spelling.
template <int... Ns>
struct Shape {
    static constexpr int kRank = sizeof...(Ns);
    static constexpr int kVals[kRank ? kRank : 1] = {Ns...};
    static constexpr int kM = kRank > 0 ? kVals[0] : 0;
    static constexpr int kN = kRank > 1 ? kVals[1] : 0;
    static constexpr int kK = kRank > 2 ? kVals[2] : 0;
};

}  // namespace astrai
