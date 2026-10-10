#pragma once
/* Typed GEMM query, layout routing, dispatch, and probe. */
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>
#include <utility>

#include <launcher/gemm_tiles.cuh>
#include <memory/tma.cuh>
#include <mma/mma.cuh>
#include <utils/device.cuh>

namespace astrai {
namespace gemm {

/*
 * Dtype-class derivation (returns plan_types.h's GemmPerfClass): the mma
 * promotion rule plus operand widths (mixed bf16xfp8 lands with W8A16 —
 * same bytes, same promoted bf16 k16 mma). int8 is tested FIRST: it
 * promotes to an int8 mma, so the "not bf16 -> fp8" arm would swallow it
 * and every W8A8 row would be unreachable.
 */
template <typename ElemA, typename ElemB> constexpr GemmPerfClass gemm_perf_class() {
    using MmaT = typename gemm_mma_traits<ElemA, ElemB>::MmaT;
    if constexpr (std::is_same_v<ElemA, int8_t> && std::is_same_v<ElemB, int8_t>) {
        return GemmPerfClass::kW8A8;
    } else if constexpr (!std::is_same_v<MmaT, __nv_bfloat16>) {
        return GemmPerfClass::kF8A8; // native fp8 symmetric pair
    } else if constexpr (std::is_same_v<ElemA, __nv_bfloat16> &&
                         std::is_same_v<ElemB, __nv_bfloat16>) {
        return GemmPerfClass::kW16A16;
    } else {
        return GemmPerfClass::kW8A16;
    }
}

static_assert(gemm_perf_class<int8_t, int8_t>() == GemmPerfClass::kW8A8,
              "int8 x int8 is its own class (see the ordering note above)");
static_assert(gemm_perf_class<__nv_fp8_e4m3, __nv_fp8_e4m3>() == GemmPerfClass::kF8A8,
              "fp8 x fp8 keys the F8A8 table");
static_assert(gemm_perf_class<__nv_bfloat16, __nv_bfloat16>() == GemmPerfClass::kW16A16,
              "bf16 x bf16 keys the W16A16 table");
static_assert(gemm_perf_class<__nv_bfloat16, int8_t>() == GemmPerfClass::kW8A16,
              "a quantized weight against bf16 activations keys W8A16");

/* Derive the query from typed operands and the launch geometry. */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename OutT = __nv_bfloat16,
          typename Schedule = MmaSync,
          typename LayoutOut = RowMajor>
PlanQuery plan_query(const GemmParams& p, const DeviceFacts& dev) {
    PlanQuery q;
    q.m = p.m;
    q.n = p.n;
    q.k = p.k;
    q.batch = p.batch;
    q.perf_class = (int)gemm_perf_class<ElemA, ElemB>();
    q.crosswise = crosswise_of<LayoutA, LayoutB>();
    q.ba = (int)sizeof(ElemA);
    q.bb = (int)sizeof(ElemB);
    q.out_elem_bytes = (int)sizeof(OutT);
    using MmaT = typename gemm_mma_traits<ElemA, ElemB>::MmaT;
    using MmaShape = typename astrai::MmaShapeFor<MmaT>::type;
    q.mma_k = MmaShape::kK;
    // Match the descriptor's alignment check before pricing TMA residency.
    // Driver encode can still reject a map; the launcher then falls back.
    q.tma = Schedule::kTma && crosswise_of<LayoutA, LayoutB>() == 0 && sizeof(ElemA) <= 2 &&
            sizeof(ElemB) <= 2 && dev.cc >= 90 && !gemm_tma_staging_disabled() &&
            astrai::tma_aligned16(p.a_ptr, p.a_ld * sizeof(ElemA),
                                  (p.batch > 1 ? p.a_batch_stride : 0) * sizeof(ElemA)) &&
            astrai::tma_aligned16(p.b_ptr, p.b_ld * sizeof(ElemB),
                                  (p.batch > 1 ? p.b_batch_stride : 0) * sizeof(ElemB));
    q.contiguous = q.crosswise == 0 && p.a_ld == p.k && p.b_ld == p.k &&
                   (reinterpret_cast<uintptr_t>(p.a_ptr) % 32) == 0 &&
                   (reinterpret_cast<uintptr_t>(p.b_ptr) % 32) == 0 &&
                   (p.batch <= 1 || ((p.a_batch_stride * sizeof(ElemA)) % 32 == 0 &&
                                     (p.b_batch_stride * sizeof(ElemB)) % 32 == 0));
    q.dev = dev;
    q.rank3a = p.batch > 1 && p.a_batch_stride > 0;
    q.rank3b = p.batch > 1 && p.b_batch_stride > 0;
    q.resources = &resources_for<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT, Schedule>;
    return q;
}

/* Carry the effective staging decision into the typed launch. */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename OutT = __nv_bfloat16,
          typename Schedule = MmaSync,
          typename LayoutOut = RowMajor>
LaunchPlan plan_dispatch_for(const GemmParams& p) {
    const PlanQuery q =
        plan_query<ElemA, ElemB, LayoutA, LayoutB, OutT, Schedule, LayoutOut>(p, device_facts());
    return {plan_dispatch(q), q.tma};
}

/*
 * Pure problem rewrite: dual-N-contiguous (NN) has no instantiation — it
 * runs as its transpose E = B^T A^T over swapped operands (CUTLASS-sm90's
 * is_swapAB) with a transposed-epilogue scatter into the [M][N] buffer.
 * The rare NN path pays a scalar-store scatter for it.
 */
inline void canonicalize_gemm(GemmParams& p, bool& trans_a, bool& trans_b) {
    if (!trans_a && !trans_b) {
        GemmParams s = p; // E = B^T * A^T: swap roles, M <-> N
        s.m = p.n;
        s.n = p.m;
        s.a_ptr = p.b_ptr;
        s.b_ptr = p.a_ptr;
        s.a_ld = p.b_ld;
        s.b_ld = p.a_ld;
        s.a_batch_stride = p.b_batch_stride;
        s.b_batch_stride = p.a_batch_stride;
        /*
         * The caller's [M][N] buffer read as E = B^T A^T: the epilogue
         * walks the caller's rows (kernel n) with the caller's N stride.
         */
        s.out_ld = p.n;
        p = s;
        trans_a = trans_b = true;
    }
}

/*
 * The (trans_a, trans_b) -> layout-tag ladder, ONE home for the branch both
 * the launch and the probe walk (a probe cannot answer for a layout the
 * launch does not take). fn gets three tags; the probe ignores the output
 * one. `swapped` marks the symmetric-NN rewrite's TT — the only arm whose
 * OUTPUT tag is transposed. A visitor, not a tag-returning function: a tag's
 * identity is its TYPE; every arm calls fn, keeping this total.
 */
template <typename ElemA, typename ElemB, typename F>
auto with_layout_tags(bool trans_a, bool trans_b, bool swapped, F&& fn) {
    constexpr bool kSymmetric = std::is_same_v<ElemA, ElemB>;
    if (trans_a && trans_b) {
        if constexpr (kSymmetric) {
            if (swapped)
                return fn(ColMajor{}, ColMajor{}, ColMajor{});
            return fn(ColMajor{}, ColMajor{}, RowMajor{});
        }
        return fn(ColMajor{}, ColMajor{}, RowMajor{});
    }
    if (trans_b) {
        // NT (the fused-linear shape), the production nn.Linear route.
        return fn(RowMajor{}, ColMajor{}, RowMajor{});
    }
    if (trans_a)
        return fn(ColMajor{}, RowMajor{}, RowMajor{});
    if constexpr (kSymmetric) {
        /*
         * Unreachable: canonicalize_gemm turns a symmetric NN into the TT arm
         * above, so both callers arrive here for a mixed pair only. The arm
         * stays total anyway — the probe returns fn's value and needs no dead
         * fallback — and names the instantiation the rewrite's own TT branch
         * already takes.
         */
        return fn(ColMajor{}, ColMajor{}, RowMajor{});
    } else {
        /*
         * Dual row-major: mixed only — symmetric NN was rewritten above
         * into the transposed TT kernel (if constexpr keeps this
         * instantiation out of symmetric builds).
         */
        return fn(RowMajor{}, RowMajor{}, RowMajor{});
    }
}

/*
 * Dtype-generic entry: canonicalize, plan, wire the tags. ElemA/ElemB/OutT
 * are independent knobs. The one asymmetry is NN: the swap rewrite assumes
 * a single element type, so symmetric NN rewrites to TT while mixed NN
 * instantiates the dual-row-major shape directly (A congruous, B
 * crosswise).
 */
template <typename ElemA,
          typename ElemB = ElemA,
          typename OutT = __nv_bfloat16,
          typename Schedule = MmaSync>
void gemm_dispatch(GemmParams p, cudaStream_t stream, bool trans_a, bool trans_b) {
    constexpr bool kSymmetric = std::is_same_v<ElemA, ElemB>;
    bool swapped = false;
    if constexpr (kSymmetric) {
        swapped = !trans_a && !trans_b; // canonicalize rewrites NN
        canonicalize_gemm(p, trans_a, trans_b);
    }
    /*
     * Each branch plans from its OWN layout tags, so the plan and the launch
     * below cannot disagree about the crosswise count, widths or perf class.
     * The tags ride empty tag instances; decltype recovers the types.
     */
    const auto launch = [&](auto la, auto lb, auto lout) {
        launch_plan<ElemA, ElemB, decltype(la), decltype(lb), decltype(lout), OutT, Schedule>(
            p,
            plan_dispatch_for<ElemA, ElemB, decltype(la), decltype(lb), OutT, Schedule,
                              decltype(lout)>(p),
            stream);
    };
    with_layout_tags<ElemA, ElemB>(trans_a, trans_b, swapped, launch);
}

/*
 * Explicit-instantiation spelling shared by the per-pair TUs (bare) and
 * gemm.cu's extern block: one place names the signature.
 */
#ifndef ASTRAI_GEMM_SCHEDULE
#define ASTRAI_GEMM_SCHEDULE MmaSync
#endif

#define ASTRAI_GEMM_INSTANTIATE(W, A)                                                              \
    template void gemm_dispatch<W, A, __nv_bfloat16, ASTRAI_GEMM_SCHEDULE>(                        \
        GemmParams, cudaStream_t, bool, bool);                                                     \
    template std::pair<PlanDecision, PlanQuery> plan_probe_for<W, A, ASTRAI_GEMM_SCHEDULE>(        \
        int64_t, int64_t, int64_t, int64_t, bool, bool, const DeviceFacts&)

/*
 * Host-only planner probe (the autotuner's coverage check): the decision
 * gemm_dispatch would make, without a launch or timing. The heuristic
 * queries compiled kernel resources on the current CUDA device.
 * The shared tag ladder (NN rewrite included) keeps a probe from
 * disagreeing with the real call's branch. Returns the decision plus the
 * query it answered.
 */
template <typename ElemA, typename ElemB, typename Schedule = MmaSync>
std::pair<PlanDecision, PlanQuery> plan_probe_for(int64_t m,
                                                  int64_t n,
                                                  int64_t k,
                                                  int64_t batch,
                                                  bool trans_a,
                                                  bool trans_b,
                                                  const DeviceFacts& dev) {
    GemmParams p{}; // contiguous operands with aligned base addresses
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.batch = static_cast<int>(batch);
    p.a_ld = trans_a ? m : k;
    p.b_ld = trans_b ? k : n;
    p.a_batch_stride = batch > 1 ? m * k : 0;
    p.b_batch_stride = batch > 1 ? n * k : 0;
    bool swapped = false;
    if constexpr (std::is_same_v<ElemA, ElemB>) {
        swapped = !trans_a && !trans_b;
        canonicalize_gemm(p, trans_a, trans_b); // symmetric NN -> transposed TT
    }
    return with_layout_tags<ElemA, ElemB>(
        trans_a, trans_b, swapped, [&](auto la, auto lb, auto lo) {
            PlanQuery q = plan_query<ElemA, ElemB, decltype(la), decltype(lb), __nv_bfloat16,
                                     Schedule, decltype(lo)>(p, dev);
            return std::make_pair(plan_dispatch(q), std::move(q));
        });
}

} // namespace gemm
} // namespace astrai
