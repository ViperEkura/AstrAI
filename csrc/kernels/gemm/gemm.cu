// GEMM family, typed host layer (module `gemm`): the dtype-pair registry, the
// single quantized-GEMM entry, and the planner's C++ face — probe, row
// injection, runtime configuration and the tile vocabulary, all declared in
// gemm/api.h. The pybind surface (argument marshalling, the dict shapes, the
// module registration) lives in bindings.cu; this TU holds no py:: type.
// torch/extension.h is here for the torch::Tensor spelling only.
//
// The policy instantiation space compiles one explicit instantiation per dtype
// pair (one per .cu below), so the heavy template work runs as parallel
// nvcc jobs. This TU keeps the dtype-pair switch; the C tests instantiate from
// the headers instead.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm.cuh"
#include "gemm/api.h"
#include "quantize/checks.h"

using namespace astrai;
using namespace astrai::quant;

namespace astrai {
namespace gemm {

// The dtype-pair table: one line per supported pair — (torch ScalarType,
// element type) per operand — and the single source this TU's consumers
// stamp: the extern template declarations below and the dispatch / probe
// lookups further down. A pair is added once here and cannot reach one
// consumer without the other. (The extern block used to be hand-maintained
// beside the switch, and had already grown two bf16 x fp8 entries with no
// instantiation TU behind them.) The CMake gemm module entry carries one
// instantiation TU per line — kept together by hand, and a line with no TU
// behind it fails the link loudly.
#define ASTRAI_GEMM_PAIRS(X)                                                 \
    X(torch::kBFloat16, __nv_bfloat16, torch::kBFloat16, __nv_bfloat16)      \
    X(torch::kBFloat16, __nv_bfloat16, torch::kChar, int8_t)                 \
    X(torch::kChar, int8_t, torch::kChar, int8_t)                            \
    X(torch::kFloat8_e4m3fn, __nv_fp8_e4m3, torch::kFloat8_e4m3fn,           \
      __nv_fp8_e4m3)                                                         \
    X(torch::kFloat8_e5m2, __nv_fp8_e5m2, torch::kFloat8_e5m2, __nv_fp8_e5m2)

// The per-pair specializations are explicitly instantiated in their own TUs
// (gemm_bf16_bf16.cu etc.), one nvcc job per dtype pair. These extern
// template declarations keep the dispatch switch below from re-instantiating:
// the address-of forms are references to the externally defined symbols only.
// (They must sit here, outside the anonymous namespace — nvcc rejects
// extern template declarations in an anonymous namespace.)
#define ASTRAI_GEMM_EXTERN(SA, TA, SB, TB) \
    extern ASTRAI_GEMM_INSTANTIATE(TA, TB);
ASTRAI_GEMM_PAIRS(ASTRAI_GEMM_EXTERN)
#undef ASTRAI_GEMM_EXTERN

namespace {

// Inner-layout resolution for one GEMM operand. The user flag names the
// math (0 = last two dims are [rows][contract], 1 = transposed); the
// storage may independently be a col-major view (.t() of a contiguous
// buffer), which folds into the returned dispatch flag at zero copy — the
// kernel's LayoutA/LayoutB tags cover both storages. m/n/k derive from the
// user flag only. Tensors whose inner dims are neither natural layout fall
// back to .contiguous().
bool resolve_operand(const torch::Tensor& t_in, bool flag, int64_t& ld,
                     int64_t& batch_stride, torch::Tensor& storage) {
    torch::Tensor t = t_in;
    bool col_major = false;
    if (t.stride(-1) != 1) {
        if (t.stride(-2) == 1) {
            col_major = true;
        } else {
            t = t.contiguous();
        }
    }
    storage = t;
    ld = col_major ? t.stride(-1) : t.stride(-2);
    batch_stride = t.dim() == 3 ? t.stride(0) : 0;
    return flag ^ col_major;
}


}  // namespace

// ---------------------------------------------------------------------------
// quant_gemm: the single quantized-GEMM entry over the dtype-generic
// dispatch. Scale semantics follow GemmParams: one float32 element is the
// per-tensor device scalar; a 1-D float32 tensor of the operand's extent
// is per-row (activations) / per-channel (weights), applied in the epilogue.
// ---------------------------------------------------------------------------

namespace {

// A resolved dequant scale: n == 0 marks the per-tensor device scalar.
struct QuantScale {
    const float* ptr = nullptr;
    int n = 0;
};

QuantScale resolve_quant_scale(const torch::Tensor& s, int64_t extent,
                               const char* name) {
    TORCH_CHECK(s.defined(), name, " is required");
    TORCH_CHECK(s.is_cuda() && s.scalar_type() == torch::kFloat32 &&
                    s.is_contiguous(),
                name, " must be a contiguous CUDA float32 tensor");
    TORCH_CHECK(s.numel() == 1 || s.numel() == extent,
                name, " must hold 1 element (per-tensor) or ", extent,
                " (per-row/per-channel)");
    return {s.data_ptr<float>(), s.numel() == 1 ? 0 : (int)extent};
}

// The lookup key both stamped switches below pack their case labels with:
// one u16 per pair, so every label is a compile-time constant and the switch
// lowers to one indexed branch with no runtime-initialized state. An
// unsupported pair raises with the actual operand dtypes in the message
// instead of a hardcoded list that can drift.
using GemmDispatchFn = void (*)(GemmParams, cudaStream_t, bool, bool);

constexpr uint16_t pack_dtypes(c10::ScalarType a, c10::ScalarType b) {
    return static_cast<uint16_t>(static_cast<uint8_t>(a)) << 8 |
           static_cast<uint8_t>(b);
}

// The unsupported-pair arm, one spelling for the two lookups below: the
// switch's own default carries it, so the non-void lookups cannot fall off
// their end. The message names the operand dtypes it actually got instead of
// a hardcoded list that can drift.
#define ASTRAI_GEMM_UNSUPPORTED_PAIR(SA, SB)                            \
    TORCH_CHECK(false, "unsupported operand dtype pair ", toString(SA), \
                " x ", toString(SB),                                    \
                ": expected bf16 x int8 (W8A16), int8 x int8 (W8A8), "  \
                "bf16 x bf16 (W16A16), or matching fp8 x fp8")

GemmDispatchFn find_gemm_dispatch(c10::ScalarType a, c10::ScalarType b) {
#define GEMM_CASE(SA, TA, SB, TB) \
    case pack_dtypes(SA, SB):     \
        return &gemm_dispatch<TA, TB>;
    switch (pack_dtypes(a, b)) {
        ASTRAI_GEMM_PAIRS(GEMM_CASE)
    default:
        ASTRAI_GEMM_UNSUPPORTED_PAIR(a, b);
    }
#undef GEMM_CASE
}

// Host-only functions that run the planner without a launch (the
// autotuner's coverage check).
using GemmProbeFn =
    std::pair<PlanDecision, PlanQuery> (*)(int64_t, int64_t, int64_t, int64_t,
                                           bool, bool, const DeviceFacts&);

GemmProbeFn find_gemm_probe(c10::ScalarType a, c10::ScalarType b) {
#define PROBE_CASE(SA, TA, SB, TB) \
    case pack_dtypes(SA, SB):      \
        return &plan_probe_for<TA, TB>;
    switch (pack_dtypes(a, b)) {
        ASTRAI_GEMM_PAIRS(PROBE_CASE)
    default:
        ASTRAI_GEMM_UNSUPPORTED_PAIR(a, b);
    }
#undef PROBE_CASE
}

}  // namespace

// ---------------------------------------------------------------------------
// Planner introspection: the Python tooling's C++ face. The planner is
// GPU-free by design, so the probe launches nothing. Rows reach the planner
// through configure()'s rows channel; injected rows rank BELOW the override
// rows (plan_table.h), keeping the override tier authoritative.
// ---------------------------------------------------------------------------

PlanProbe plan_probe(int64_t m, int64_t n, int64_t k, at::ScalarType dt_a,
                     at::ScalarType dt_b, bool trans_a, bool trans_b,
                     int64_t batch) {
    const auto [decision, query] = find_gemm_probe(dt_a, dt_b)(
        m, n, k, batch, trans_a, trans_b, astrai::device_facts());
    PlanProbe r;
    r.source = decision.source;
    r.cta = decision.recipe.cta;
    r.stages = decision.recipe.stages;
    r.raster = decision.raster;
    r.kk = decision.recipe.kk;
    r.perf_class = query.perf_class;
    r.crosswise = query.crosswise;
    return r;
}

namespace {

// Install one row tier from a spec (a row-file path when one opens, else
// inline row text) and remember the spec, so the config state can hand back a
// value that re-installs it. `label` is what a parse error reports.
int install_rows(RowSource& tier, const char* label, const std::string& source) {
    std::vector<TableRow> rows;
    if (!parse_plan_table_file(source, rows))
        parse_plan_table_text(source, label, rows);
    const int installed = (int)rows.size();
    tier.set_from(source, std::move(rows));
    return installed;
}

RowSource& row_tier(RowTier tier) {
    return tier == RowTier::Injected ? plan_table_injected_source()
                                     : plan_table_override_source();
}

}  // namespace

// ---------------------------------------------------------------------------
// Runtime configuration: the backing of astrai.extension.plan. Every knob is
// tri-state — an absent patch field leaves it unchanged, an explicit value wins
// over the one-time env seed. Rows are addressed by tier (`rows` + `tier`), the
// all-tiers-off switch is its own field, and the staging keys are positive
// enables: tma=false forces cp.async staging, mx=false knocks the sm_120a
// block-scale cell out (the A/B knobs).
// ---------------------------------------------------------------------------

GemmConfigState config_state() {
    GemmConfigState s;
    s.planner = kPlannerModeNames[gemm_planner_mode()];  // resolves unset
    s.planner_mode = gemm_config().planner.load(std::memory_order_relaxed);
    s.log = gemm_plan_log_enabled();
    s.table_off = gemm_table_off();
    s.override_rows = (int)plan_table_override_source().size();
    s.override_source = plan_table_override_source().source();
    s.injected_rows = (int)plan_table_injected_source().size();
    s.injected_source = plan_table_injected_source().source();
    s.staging_tma = !gemm_tma_staging_disabled();
    s.staging_mx = !gemm_mx_cell_disabled();
    return s;
}

GemmConfigState configure(const GemmConfigPatch& patch) {
    gemm_config_seed_once();
    if (patch.planner_mode.has_value()) {
        const int mode = *patch.planner_mode;
        if (mode < -1 || mode >= kPlannerModeCount)
            throw std::invalid_argument("planner mode must be -1..2");
        gemm_config().planner = mode;
    }
    if (patch.log.has_value())
        gemm_config().log = *patch.log ? 1 : 0;
    if (patch.staging_tma.has_value())
        gemm_config().tma_disabled = *patch.staging_tma ? 0 : 1;
    if (patch.staging_mx.has_value())
        gemm_config().mx_disabled = *patch.staging_mx ? 0 : 1;
    if (patch.table_off.has_value())
        gemm_config().table_off = *patch.table_off ? 1 : 0;
    if (patch.rows.has_value()) {
        const RowTier which = patch.tier.value_or(RowTier::Override);
        if (patch.rows->empty()) {
            row_tier(which).clear();
        } else {
            install_rows(row_tier(which),
                         which == RowTier::Injected ? "injected rows"
                                                    : "override rows",
                         *patch.rows);
        }
    }
    return config_state();
}

// The recipe vocabulary per (crosswise, operand widths) — every
// (CTA class, stages, kK) the launch ladders instantiate for that staging
// pair, deduped on the dispatch key, in dispatch (manifest) order. Rows are
// (crosswise, ba, bb, cta, stages, kk, bm, bn, wm, wn, threads, smem);
// a row's numbers spell its canonical name,
// Tile_<bm>x<bn>x<kk>_W<wm>x<wn>_S<stages> — the bench's own spelling —
// which is how the Python tooling joins rows with dataset recipe strings
// without keeping a second copy of the vocabulary. (2,1) covers the mixed
// W8A16 class, whose congruent staging runs the conservative
// ladder too (manifest_kind's fallback); (1,2) matches no supported pair.
std::vector<std::vector<int>> tile_vocabulary() {
    const std::pair<int, int> widths[] = {{2, 2}, {2, 1}, {1, 1}};
    std::vector<std::vector<int>> out;
    for (int crosswise = 0; crosswise <= 1; ++crosswise)
        for (const auto& [ba, bb] : widths)
            for (const GemmRecipe& r : gemm_recipes_for(crosswise != 0, ba, bb))
                out.push_back({crosswise, ba, bb, r.cta, r.stages, r.kk,
                               r.bm, r.bn, r.wm, r.wn, r.threads, r.smem});
    return out;
}

// The TileClass spellings, in enum order — what a row's cta ordinal expands
// to in the compiled-in tables (the GENERATED block's paste target). Owned
// here so the sweep's C++ emitter needs no Python-side copy of the names.
std::vector<const char*> tile_class_names() {
    static constexpr const char* kNames[] = {
        "kSmall64", "kNarrow128x64", "kBig128", "kWide128x256",
        "kTall64x128"};
    static_assert((int)TileClass::kTall64x128 ==
                      (int)(sizeof(kNames) / sizeof(kNames[0])) - 1,
                  "kNames is indexed by TileClass: keep it in enum order");
    return std::vector<const char*>(kNames, kNames + sizeof(kNames) / sizeof(kNames[0]));
}

// ---------------------------------------------------------------------------
// The single quantized-GEMM entry (one kernel for every cell, the only
// kernel-facing export). The dtype pair picks the mma mode:
//   bf16 x bf16 (W16A16)        — no scales
//   bf16 x int8 (W8A16)         — b_scale required
//   int8 x int8 (W8A8)          — both scales required
//   fp8 x fp8, matching formats — both scales optional
// Scale arity is validated per side: int8 requires its dequant scale, fp8
// takes one optionally (per-tensor scalar or the operand's extent), bf16
// rejects one (nothing to dequant). The body packs GemmParams (batch
// broadcast rules, zero-copy transposed views, fused bf16 bias) and hands
// it to the dtype-pair dispatch.
// ---------------------------------------------------------------------------
torch::Tensor quant_gemm_impl(torch::Tensor a, torch::Tensor b,
                              c10::optional<torch::Tensor> a_scale,
                              c10::optional<torch::Tensor> b_scale,
                              bool trans_a, bool trans_b,
                              c10::optional<torch::Tensor> bias) {
    const auto dt_a = a.scalar_type(), dt_b = b.scalar_type();
    const bool i8a = dt_a == torch::kChar, i8b = dt_b == torch::kChar;
    const bool f8a = dt_a == torch::kFloat8_e4m3fn || dt_a == torch::kFloat8_e5m2;
    const bool f8b = dt_b == torch::kFloat8_e4m3fn || dt_b == torch::kFloat8_e5m2;
    const bool b16a = dt_a == torch::kBFloat16, b16b = dt_b == torch::kBFloat16;
    // The supported-pair set is validated once, by the dispatch switch's
    // default arm (find_gemm_dispatch), with the operand dtypes in the
    // message.
    if (f8a || f8b) {
        astrai::quant::check_fp8_device(a.device().index());
    }
    const int64_t m = trans_a ? a.size(-1) : a.size(-2);
    const int64_t n = trans_b ? b.size(-2) : b.size(-1);
    auto opt_scale = [&](const c10::optional<torch::Tensor>& s, int64_t extent,
                         const char* name, bool i8_side,
                         bool bf16_side) -> QuantScale {
        if (!s.has_value()) {
            TORCH_CHECK(!i8_side, "quant_gemm: ", name,
                        " is required for an int8 operand");
            return {nullptr, 0};
        }
        TORCH_CHECK(!bf16_side, "quant_gemm: ", name,
                    " given for a bf16 operand (nothing to dequant)");
        return resolve_quant_scale(*s, extent, name);
    };
    const QuantScale sa = opt_scale(a_scale, m, "a_scale", i8a, b16a);
    const QuantScale sb = opt_scale(b_scale, n, "b_scale", i8b, b16b);

    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK((a.dim() == 2 || a.dim() == 3) &&
                    (b.dim() == 2 || b.dim() == 3),
                "a and b must be 2D or 3D (batched)");
    TORCH_CHECK(a.device() == b.device(), "a and b must share device");
    torch::Tensor bias_t;
    if (bias.has_value()) bias_t = *bias;
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t batch_a = a.dim() == 3 ? a.size(0) : 1;
    const int64_t batch_b = b.dim() == 3 ? b.size(0) : 1;
    TORCH_CHECK(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                "batch dim mismatch (got ", batch_a, " and ", batch_b, ")");
    const int64_t batch = std::max(batch_a, batch_b);
    TORCH_CHECK(batch <= 65535, "batch dim exceeds the grid.z launch limit");

    torch::Tensor a_st, b_st;
    int64_t a_ld, b_ld, a_bstride, b_bstride;
    const bool tag_a = resolve_operand(a, trans_a, a_ld, a_bstride, a_st);
    const bool tag_b = resolve_operand(b, trans_b, b_ld, b_bstride, b_st);
    const int64_t k = trans_a ? a.size(-2) : a.size(-1);
    TORCH_CHECK(k == (trans_b ? b.size(-1) : b.size(-2)), "inner dim mismatch");
    TORCH_CHECK(sa.ptr == nullptr || sa.n == 0 || sa.n == m,
                "a_scale extent must match m");
    TORCH_CHECK(sb.ptr == nullptr || sb.n == 0 || sb.n == n,
                "w_scale extent must match n");

    const bool batched_out = a.dim() == 3 || b.dim() == 3;
    torch::Tensor output =
        batched_out
            ? torch::empty({batch, m, n}, a.options().dtype(torch::kBFloat16))
            : torch::empty({m, n}, a.options().dtype(torch::kBFloat16));
    GemmParams p;
    p.a_ptr = a_st.data_ptr();
    p.b_ptr = b_st.data_ptr();
    p.out_ptr = output.data_ptr();
    p.a_scale = sa.ptr;
    p.a_scale_m = sa.n;
    p.b_scale = sb.ptr;
    p.b_scale_n = sb.n;
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.a_ld = static_cast<int>(a_ld);
    p.b_ld = static_cast<int>(b_ld);
    if (bias_t.defined() && bias_t.numel() > 0) {
        TORCH_CHECK(bias_t.is_cuda() && bias_t.scalar_type() == torch::kBFloat16,
                    "quantized gemm bias must be a CUDA bf16 tensor");
        TORCH_CHECK(bias_t.dim() == 1 && bias_t.size(0) == n,
                    "quantized gemm bias must be 1D of length n=", n);
        TORCH_CHECK(bias_t.is_contiguous(), "bias must be contiguous");
        p.bias_ptr = bias_t.data_ptr();
    }
    p.batch = static_cast<int>(batch);
    p.a_batch_stride = (batch_a == 1 && batch > 1) ? 0 : a_bstride;
    p.b_batch_stride = (batch_b == 1 && batch > 1) ? 0 : b_bstride;
    p.out_batch_stride = m * n;
    p.out_ld = static_cast<int>(n);

    find_gemm_dispatch(dt_a, dt_b)(p, stream.stream(), tag_a, tag_b);
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

}  // namespace gemm
}  // namespace astrai
