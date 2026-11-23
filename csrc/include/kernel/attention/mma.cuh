#pragma once
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <datatype/element.cuh>
#include <kernel/attention/kernel.cuh>
#include <memory/layout_policies.cuh>
#include <mma/ldmatrix.cuh>
#include <mma/mma.cuh>
#include <utils/define.cuh>

namespace astrai {
namespace attention {

template <int HEAD_DIM_, int BC_, int WARPS_, int STAGES_, typename T_ = bf16>
struct KernelTraits {
    using Elem = T_;
    using Atom = MmaOp<Elem, Elem, typename MmaShapeFor<Elem>::type>;
    using FragmentLayout = typename Atom::Layout;
    using SharedLayout = SharedTileLayout<HEAD_DIM_>;
    static_assert(sizeof(Elem) == 2 && Atom::Shape::kK == 16,
                  "attention tiles require a 16-bit m16n8k16 MMA atom");
    static_assert(HEAD_DIM_ % Atom::Shape::kK == 0 && BC_ % Atom::Shape::kK == 0,
                  "attention tiles must contain complete MMA cells");
    static_assert(WARPS_ > 0 && WARPS_ <= 32, "invalid warp count");
    static_assert(STAGES_ > 0 && (STAGES_ & (STAGES_ - 1)) == 0,
                  "attention pipeline stages must be a power of two");

    static constexpr int HEAD_DIM = HEAD_DIM_;
    static constexpr int BC = BC_;         // K/V tile size along seq dim
    static constexpr int WARPS = WARPS_;   // warps per block
    static constexpr int STAGES = STAGES_; // double-buffer stages (1 or 2)

    static constexpr int BR = Atom::Shape::kM; // Q rows per warp (mma M=16)

    /* Derived MMA tile counts use the shared mma_shape. Unsupported element
     * types fail at MmaShapeFor because they have no tensor-core cell.
     */
    static constexpr int KD = HEAD_DIM / Atom::Shape::kK;  // Q/K k-slides
    static constexpr int NC8 = BC / Atom::Shape::kN;       // S n-tiles (N=8)
    static constexpr int KT2 = BC / Atom::Shape::kK;       // P k-tiles (K=16)
    static constexpr int DN8 = HEAD_DIM / Atom::Shape::kN; // O n-tiles (N=8)

    static constexpr int LD = HEAD_DIM; // smem leading dim

    using QueryFragment = unsigned[KD][Atom::kARegs];
    using ScoreFragment = float[NC8][Atom::kCRegs];
    using OutputFragment = float[DN8][Atom::kCRegs];

    static constexpr int NUM_THREADS = WARPS * 32;
    static constexpr int VEC = 16 / (int)sizeof(Elem); // elements per cp.async unit
    static constexpr int TOTAL = BC * HEAD_DIM;        // total elements per tile
};

/* Typed tile operations compose the shared warp atom without owning storage or masks. */
template <typename Traits>
struct AttentionMma {
    using T = typename Traits::Elem;
    using Atom = typename Traits::Atom;
    using QueryFragment = typename Traits::QueryFragment;
    using ScoreFragment = typename Traits::ScoreFragment;
    using OutputFragment = typename Traits::OutputFragment;
    static DEVICE_FORCEINLINE void
    load_query(const AttentionParams& p, const QueryRows& q, int lane, QueryFragment& frag) {
        const int column = Traits::FragmentLayout::column(lane);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const auto& row = q.row[r];
            const T* base = static_cast<const T*>(p.q_ptr) + row.offset;
#pragma unroll
            for (int k = 0; k < Traits::KD; ++k) {
                const int d = k * Atom::Shape::kK + column;
                const auto* pair = reinterpret_cast<const unsigned*>(base + d);
                // The A fragment stores two rows at d, followed by the same rows at d + 8.
                frag[k][r] = row.valid ? pair[0] : 0u;
                frag[k][r + 2] = row.valid ? pair[4] : 0u;
            }
        }
    }

    // An x4 load supplies the B fragments for two adjacent m16n8 operations.
    template <bool Transpose>
    static DEVICE_FORCEINLINE void load_b_pair(unsigned (&b)[4], const T* tile, int row, int col) {
        const auto offset = row * Traits::LD + Traits::SharedLayout::column(col, row);
        astrai::ldmatrix_x4_lane<Transpose>(b, __cvta_generic_to_shared(tile + offset));
    }

    static DEVICE_FORCEINLINE void
    scores(const QueryFragment& query, const T* keys, int lane, ScoreFragment& scores) {
        constexpr int N = Atom::Shape::kN;
#pragma unroll
        for (int n = 0; n < Traits::NC8; n += 2) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                scores[n][i] = scores[n + 1][i] = 0.0f;
            // Lanes 0..15 address the first K fragment and 16..31 the next eight keys.
            const int row = n * N + (lane & 7) + (lane / 16) * N;
#pragma unroll
            for (int k = 0; k < Traits::KD; ++k) {
                unsigned b[4];
                load_b_pair<false>(b, keys, row, k * Atom::Shape::kK + (lane & 8));
                Atom::fma(scores[n], query[k], b, scores[n]);
                Atom::fma(scores[n + 1], query[k], b + 2, scores[n + 1]);
            }
        }
    }

    static __device__ inline void values(const ScoreFragment& weights,
                                         const typename Traits::Elem* __restrict__ sV,
                                         int lane,
                                         OutputFragment& Oacc) {
#pragma unroll
        for (int kt2 = 0; kt2 < Traits::KT2; kt2++) {
            unsigned Pa[Atom::kARegs];
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const auto& p = weights[2 * kt2 + n];
                Pa[2 * n] = ElemTrait<T>::pack2(p[0], p[1]);
                Pa[2 * n + 1] = ElemTrait<T>::pack2(p[2], p[3]);
            }
            const int row = kt2 * Atom::Shape::kK + (lane & 15);
#pragma unroll
            for (int n = 0; n < Traits::DN8; n += 2) {
                unsigned b[4];
                const int col = (n + lane / 16) * Atom::Shape::kN;
                load_b_pair<true>(b, sV, row, col);
                Atom::fma(Oacc[n], Pa, b, Oacc[n]);
                Atom::fma(Oacc[n + 1], Pa, b + 2, Oacc[n + 1]);
            }
        }
    }
};

} // namespace attention
} // namespace astrai
