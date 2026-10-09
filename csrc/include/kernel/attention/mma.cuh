#pragma once
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory/layout_policies.cuh>
#include <mma/ldmatrix.cuh>
#include <mma/mma.cuh>
#include <utils/define.cuh>
#include <datatype/element.cuh>


namespace astrai {
namespace attention {

template <int HEAD_DIM_, int BC_, int WARPS_, int STAGES_, typename T_ = bf16> struct KernelTraits {
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
    static constexpr int KD = HEAD_DIM / Atom::Shape::kK; // Q/K k-slides
    static constexpr int NC8 = BC / Atom::Shape::kN;         // S n-tiles (N=8)
    static constexpr int KT2 = BC / Atom::Shape::kK;      // P k-tiles (K=16)
    static constexpr int DN8 = HEAD_DIM / Atom::Shape::kN;   // O n-tiles (N=8)

    static constexpr int LD = HEAD_DIM; // smem leading dim

    using QueryFragment = unsigned[KD][Atom::kARegs];
    using ScoreFragment = float[NC8][Atom::kCRegs];
    using OutputFragment = float[DN8][Atom::kCRegs];

    static constexpr int NUM_THREADS = WARPS * 32;
    static constexpr int VEC = 16 / (int)sizeof(Elem); // elements per cp.async unit
    static constexpr int TOTAL = BC * HEAD_DIM;        // total elements per tile
};

/* Typed tile operations compose the shared warp atom without owning storage or masks. */
template <typename Traits> struct AttentionMma {
    using T = typename Traits::Elem;
    using Atom = typename Traits::Atom;
    using QueryFragment = typename Traits::QueryFragment;
    using ScoreFragment = typename Traits::ScoreFragment;
    using OutputFragment = typename Traits::OutputFragment;
    static __device__ inline void clear(OutputFragment& output) {
#pragma unroll
        for (int j = 0; j < Traits::DN8; ++j)
            output[j][0] = output[j][1] = output[j][2] = output[j][3] = 0.0f;
    }

    static __device__ inline void load_query(const T* __restrict__ qa,
                                              const T* __restrict__ qb,
                                              int stride_d,
                                              int off_a,
                                              int off_b,
                                              bool va,
                                              bool vb,
                                              int tid4,
                                              QueryFragment& Qa) {
#pragma unroll
        for (int kt = 0; kt < Traits::KD; kt++) {
            int c = kt * Atom::Shape::kK + tid4 * 2;
            const unsigned* pau = reinterpret_cast<const unsigned*>(&qa[off_a + c * stride_d]);
            const unsigned* pbu = reinterpret_cast<const unsigned*>(&qb[off_b + c * stride_d]);
            Qa[kt][0] = va ? pau[0] : 0u;
            Qa[kt][1] = vb ? pbu[0] : 0u;
            Qa[kt][2] = va ? pau[4] : 0u;
            Qa[kt][3] = vb ? pbu[4] : 0u;
        }
    }

    static __device__ inline void scores(const QueryFragment& Qa,
                                          const typename Traits::Elem* __restrict__ sK,
                                          int lane,
                                          ScoreFragment& Sacc) {
#pragma unroll
        for (int n8 = 0; n8 < Traits::NC8; n8++) {
            Sacc[n8][0] = Sacc[n8][1] = Sacc[n8][2] = Sacc[n8][3] = 0.0f;
            int krow_l = n8 * Atom::Shape::kN + (lane & 7);
            int kcol_h = (lane & 8) ? 8 : 0;
#pragma unroll
            for (int kt = 0; kt < Traits::KD; kt++) {
                unsigned b[Atom::kBRegs];
                astrai::ldmatrix_x2<typename Traits::Elem>(
                    b,
                    &sK[krow_l * Traits::LD + Traits::SharedLayout::column(kt * Atom::Shape::kK + kcol_h, krow_l)]);
                Atom::fma(Sacc[n8], Qa[kt], b, Sacc[n8]);
            }
        }
    }

    static __device__ inline void values(const ScoreFragment& Sacc,
                                          const typename Traits::Elem* __restrict__ sV,
                                          int lane,
                                          OutputFragment& Oacc) {
#pragma unroll
        for (int kt2 = 0; kt2 < Traits::KT2; kt2++) {
            unsigned Pa[Atom::kARegs];
            Pa[0] = ElemTrait<T>::pack2(Sacc[kt2 * 2][0], Sacc[kt2 * 2][1]);
            Pa[1] = ElemTrait<T>::pack2(Sacc[kt2 * 2][2], Sacc[kt2 * 2][3]);
            Pa[2] = ElemTrait<T>::pack2(Sacc[kt2 * 2 + 1][0], Sacc[kt2 * 2 + 1][1]);
            Pa[3] = ElemTrait<T>::pack2(Sacc[kt2 * 2 + 1][2], Sacc[kt2 * 2 + 1][3]);
            int vrow_l = kt2 * Atom::Shape::kK + (lane & 15);
#pragma unroll
            for (int dn8 = 0; dn8 < Traits::DN8; dn8++) {
                unsigned b[Atom::kBRegs];
                astrai::ldmatrix_x2<typename Traits::Elem, true>(
                    b, &sV[vrow_l * Traits::LD + Traits::SharedLayout::column(dn8 * Atom::Shape::kN, vrow_l)]);
                Atom::fma(Oacc[dn8], Pa, b, Oacc[dn8]);
            }
        }
    }

};

} // namespace attention
} // namespace astrai
