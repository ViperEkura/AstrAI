/* GEMM typed entry, dtype/schedule selection, and planner probe. */

#include <c10/core/ScalarType.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "entry.h"
#include <api/dtype.h>
#include <api/gemm.h>
#include <gemm_build.h>
#include <launcher/gemm_dispatch.cuh>

using namespace astrai;
using namespace astrai::quant;

namespace astrai {
namespace gemm {

#define ASTRAI_GEMM_BASE_PAIRS(X)                                                                  \
    X(bf16, bf16)                                                                                  \
    X(bf16, int8_t)                                                                                \
    X(int8_t, int8_t)

#if ASTRAI_BUILD_FP8
#define ASTRAI_GEMM_FP8_PAIRS(X)                                                                   \
    X(fp8_e4m3, fp8_e4m3)                                                                          \
    X(fp8_e5m2, fp8_e5m2)
#else
#define ASTRAI_GEMM_FP8_PAIRS(X)
#endif
#define ASTRAI_GEMM_PAIRS(X) ASTRAI_GEMM_BASE_PAIRS(X) ASTRAI_GEMM_FP8_PAIRS(X)

#define GEMM_EXTERN(TA, TB)                                                                        \
    extern template void gemm_dispatch<TA, TB, __nv_bfloat16, MmaSync>(GemmParams, cudaStream_t,   \
                                                                       bool, bool);                \
    extern template std::pair<PlanDecision, PlanQuery> plan_probe_for<TA, TB, MmaSync>(            \
        int64_t, int64_t, int64_t, int64_t, bool, bool, const DeviceFacts&);
ASTRAI_GEMM_PAIRS(GEMM_EXTERN)
#undef GEMM_EXTERN
#if ASTRAI_BUILD_TMA
#define GEMM_EXTERN(TA, TB)                                                                        \
    extern template void gemm_dispatch<TA, TB, __nv_bfloat16, TmaMma>(GemmParams, cudaStream_t,    \
                                                                      bool, bool);                 \
    extern template std::pair<PlanDecision, PlanQuery> plan_probe_for<TA, TB, TmaMma>(             \
        int64_t, int64_t, int64_t, int64_t, bool, bool, const DeviceFacts&);
ASTRAI_GEMM_PAIRS(GEMM_EXTERN)
#undef GEMM_EXTERN
#endif
#if ASTRAI_BUILD_MX
#define GEMM_EXTERN(TA, TB)                                                                        \
    extern template void gemm_dispatch<TA, TB, __nv_bfloat16, Sm120Mma>(GemmParams, cudaStream_t,  \
                                                                        bool, bool);               \
    extern template std::pair<PlanDecision, PlanQuery> plan_probe_for<TA, TB, Sm120Mma>(           \
        int64_t, int64_t, int64_t, int64_t, bool, bool, const DeviceFacts&);
ASTRAI_GEMM_FP8_PAIRS(GEMM_EXTERN)
#undef GEMM_EXTERN
#endif

GemmCapabilities capabilities() {
    const int cc = device_facts().cc;
    return {cc,
            supports(build::kBase, cc),
            supports(build::kFp8, cc),
            supports(build::kTma, cc),
            supports(build::kMx, cc),
            build::kTargets};
}

namespace {

/*
 * The lookup key both stamped switches below pack their case labels with:
 * one u16 per pair, so every label is a compile-time constant and the switch
 * lowers to one indexed branch with no runtime-initialized state. An
 * unsupported pair raises with the actual operand dtypes in the message
 * instead of a hardcoded list that can drift.
 */
using GemmDispatchFn = void (*)(GemmParams, cudaStream_t, bool, bool);

constexpr uint16_t pack_dtypes(c10::ScalarType a, c10::ScalarType b) {
    return static_cast<uint16_t>(static_cast<uint8_t>(a)) << 8 | static_cast<uint8_t>(b);
}

/*
 * The unsupported-pair arm, one spelling for the two lookups below: the
 * error text is generated from ASTRAI_GEMM_PAIRS, matching both lookups.
 */
std::string unsupported_pair_message(c10::ScalarType a, c10::ScalarType b) {
    std::string instantiated;
#define ASTRAI_GEMM_PAIR_ROW(TA, TB)                                                               \
    instantiated += std::string(instantiated.empty() ? "" : ", ") + toString(scalar_type_v<TA>) +  \
                    " x " + toString(scalar_type_v<TB>);
    ASTRAI_GEMM_PAIRS(ASTRAI_GEMM_PAIR_ROW)
#undef ASTRAI_GEMM_PAIR_ROW
    return std::string("unsupported operand dtype pair ") + toString(a) + " x " + toString(b) +
           " (instantiated: " + instantiated + ")";
}

template <typename A_, typename B_> struct PairTag {
    using A = A_;
    using B = B_;
};

template <typename F> auto visit_gemm_pair(c10::ScalarType a, c10::ScalarType b, F&& fn) {
#define GEMM_CASE(TA, TB)                                                                          \
    case pack_dtypes(scalar_type_v<TA>, scalar_type_v<TB>):                                        \
        return fn(PairTag<TA, TB>{});
    switch (pack_dtypes(a, b)) { ASTRAI_GEMM_PAIRS(GEMM_CASE) }
#undef GEMM_CASE
    throw std::runtime_error(unsupported_pair_message(a, b));
}

/* All kernel variants have distinct schedule types and architecture images. */
template <typename A, typename B, typename F> auto with_schedule(F&& fn) {
    constexpr bool fp8 = sizeof(A) == 1 && !std::is_same_v<A, int8_t>;
    const GemmCapabilities caps = capabilities();
    TORCH_CHECK(fp8 ? caps.fp8 : caps.mma, "GEMM kernels were not built for this device/dtype (SM",
                caps.cc, ", compiled targets: ", caps.targets, ")");
#if ASTRAI_BUILD_MX
    if constexpr (fp8) {
        if (caps.mx && !gemm_mx_cell_disabled())
            return fn(Sm120Mma{});
    }
#endif
#if ASTRAI_BUILD_TMA
    if (caps.tma && !gemm_tma_staging_disabled())
        return fn(TmaMma{});
#endif
    return fn(MmaSync{});
}

GemmDispatchFn find_gemm_dispatch(c10::ScalarType a, c10::ScalarType b) {
    return visit_gemm_pair(a, b, [](auto pair) {
        using Pair = decltype(pair);
        return with_schedule<typename Pair::A, typename Pair::B>(
            [](auto schedule) -> GemmDispatchFn {
                return &gemm_dispatch<typename Pair::A, typename Pair::B, __nv_bfloat16,
                                      decltype(schedule)>;
            });
    });
}

/*
 * Host-only functions that run the planner without a launch (the
 * autotuner's coverage check).
 */
using GemmProbeFn = std::pair<PlanDecision, PlanQuery> (*)(
    int64_t, int64_t, int64_t, int64_t, bool, bool, const DeviceFacts&);

GemmProbeFn find_gemm_probe(c10::ScalarType a, c10::ScalarType b) {
    return visit_gemm_pair(a, b, [](auto pair) {
        using Pair = decltype(pair);
        return with_schedule<typename Pair::A, typename Pair::B>([](auto schedule) -> GemmProbeFn {
            return &plan_probe_for<typename Pair::A, typename Pair::B, decltype(schedule)>;
        });
    });
}

} // namespace

/*
 * Planner introspection: the Python tooling's C++ face. The heuristic reads
 * CUDA function metadata, but the probe launches nothing. Rows reach the planner
 * through configure()'s rows channel; injected rows rank BELOW the override
 * rows (plan_table.cpp), keeping the override tier authoritative.
 */

PlanProbe plan_probe(int64_t m,
                     int64_t n,
                     int64_t k,
                     at::ScalarType dt_a,
                     at::ScalarType dt_b,
                     bool trans_a,
                     bool trans_b,
                     int64_t batch) {
    TORCH_CHECK(m > 0 && n > 0 && k > 0, "GEMM probe: M, N and K must be greater than zero (got ",
                m, ", ", n, ", ", k, ")");
    const auto [decision, query] =
        find_gemm_probe(dt_a, dt_b)(m, n, k, batch, trans_a, trans_b, astrai::device_facts());
    PlanProbe r;
    r.source = decision.source;
    r.cta = decision.recipe.cta;
    r.k_stages = decision.recipe.k_stages;
    r.raster = decision.raster;
    r.k_tile = decision.recipe.k_tile;
    r.perf_class = query.perf_class;
    r.crosswise = query.crosswise;
    r.tma = query.tma;
    if (config_state().planner == "heuristic" && query.resources) {
        for (const auto& v : tile_vocabulary()) {
            if (v[0] != (query.crosswise > 0) || v[1] != query.ba || v[2] != query.bb)
                continue;
            GemmRecipe recipe{v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]};
            const auto resource = query.resources(recipe, query);
            const auto& e = resource.effective;
            r.resources.push_back({recipe.cta, recipe.k_stages, recipe.k_tile, e.bm, e.bn, e.k_tile,
                                   e.wm, e.wn, e.threads, resource.resident, resource.registers,
                                   resource.local_bytes});
        }
    }
    return r;
}

/*
 * quant_gemm's op entry: the ladder itself lives in entry.h (one kernel for
 * every cell, the only kernel-facing export); this wrapper is the one place
 * the family's dtype-pair lookup meets it — as a template argument, so the
 * header needs neither a forward declaration of a TU-internal symbol nor an
 * include-order contract. The scale contract is stated in the header and in
 * docs/developer/kernels/gemm.md, "Scales".
 */
torch::Tensor quant_gemm_impl(torch::Tensor a,
                              torch::Tensor b,
                              c10::optional<torch::Tensor> a_scale,
                              c10::optional<torch::Tensor> b_scale,
                              bool trans_a,
                              bool trans_b,
                              c10::optional<torch::Tensor> bias) {
    return quant_gemm_ladder<&find_gemm_dispatch>(a, b, a_scale, b_scale, trans_a, trans_b, bias);
}

} // namespace gemm
} // namespace astrai
