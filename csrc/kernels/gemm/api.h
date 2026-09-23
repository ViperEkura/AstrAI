#pragma once
// C++ entry into the quantized GEMM family — the module's whole host surface,
// for composed callers inside this module (the fp8 linear op compiles into the
// gemm module so the dispatch state — plan table, planner mode, staging
// switches — stays single-source) and for the pybind layer (bindings.cu, which
// owns every py:: spelling in the family).
//
// Two rules keep this header usable from anywhere in the family:
//   * Declarations only, and deliberately template-free: including it must not
//     instantiate the dtype-pair kernels, which are explicitly instantiated in
//     their own TUs (one nvcc job each).
//   * No Python types: a `py::dict` in a signature makes the surface callable
//     from nowhere but bindings.cu — the flat structs below carry the same
//     fields the Python tooling reads, spelled once here and marshalled once in
//     bindings.cu.
//
// Device facts need no declaration here (`astrai::device_facts()` in
// common/device.cuh is already typed) and the vocabulary returns plain
// containers — the bindings marshall both directly.

#include <c10/util/Optional.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

namespace astrai {
namespace gemm {

// ---------------------------------------------------------------------------
// The single quantized-GEMM entry (one kernel for every cell). Semantics,
// validation and the dispatch are documented at the definition in gemm.cu; the
// pybind ``quant_gemm`` is a thin None-tolerant wrapper over this.
// ---------------------------------------------------------------------------
torch::Tensor quant_gemm_impl(torch::Tensor a, torch::Tensor b,
                              c10::optional<torch::Tensor> a_scale,
                              c10::optional<torch::Tensor> b_scale,
                              bool trans_a, bool trans_b,
                              c10::optional<torch::Tensor> bias);

// ---------------------------------------------------------------------------
// Planner introspection. The planner is GPU-free by design, so the probe
// launches nothing: `source` names the row tier that decided and the rest is
// the picked recipe in dispatch-key form. `crosswise` stays an int (the
// direct-load operand count the kernel takes), not a bool — the Python tooling
// reads it as 0/1.
// ---------------------------------------------------------------------------
struct PlanProbe {
    std::string source;
    int cta = 0;
    int stages = 0;
    int raster = 0;
    int kk = 0;
    int perf_class = -1;
    int crosswise = 0;
};

PlanProbe plan_probe(int64_t m, int64_t n, int64_t k,
                     at::ScalarType dt_a, at::ScalarType dt_b, 
                     bool trans_a, bool trans_b, 
                     int64_t batch);

// ---------------------------------------------------------------------------
// Runtime configuration (the backing of astrai.extension.plan). Every field is
// tri-state: absent leaves the knob unchanged, an explicit value wins over the
// one-time env seed. The pybind layer maps an absent dict key to "absent".
// ---------------------------------------------------------------------------

// Which row tier a `rows` patch addresses. The tiers rank in this order at
// lookup — override (the experimenter's) beats injected (the autotuner's)
// beats the compiled-in tables — so a user table always wins over a tuned
// winner.
enum class RowTier : int {
    Override = 0,
    Injected = 1,
};

struct GemmConfigPatch {
    // Row spec: a row-file path, or inline row text (one row per line:
    // m_min m_max n_min n_max perf_class crosswise cta stages raster [kk]).
    // The tier it addresses is replaced wholesale; the empty string clears
    // that tier, absent leaves both tiers alone.
    c10::optional<std::string> rows;
    c10::optional<RowTier> tier;  // which tier `rows` addresses
    // Every row tier off: the rows are skipped entirely and the planner chain
    // falls through to its model/degraded end.
    c10::optional<bool> table_off;
    // 0/1/2 = table / hybrid / model; -1 restores the unset state, where the
    // env seed decides (hybrid is what an unseeded process resolves to).
    c10::optional<int> planner_mode;
    c10::optional<bool> log;
    // Staging keys are positive enables: tma=false forces cp.async staging,
    // mx=false knocks the sm_120a block-scale cell out (the A/B knobs).
    c10::optional<bool> staging_tma;
    c10::optional<bool> staging_mx;
};

// The whole configuration as a value: configure(patch) returns it, and feeding
// its fields back re-installs exactly this state — each row tier rides the
// source spec it was installed from, so save/restore round-trips in one call.
struct GemmConfigState {
    std::string planner;    // the resolved planner mode name
    int planner_mode = -1;  // the raw knob: -1 = unset (the env seed decides)
    // Installed row counts: a mistyped path that parses to no rows shows up
    // here as 0 rather than staying silent.
    int override_rows = 0;
    int injected_rows = 0;
    std::string override_source;  // what that tier was last installed from
    std::string injected_source;

    bool log = false;
    bool table_off = false;
    bool staging_tma = true;
    bool staging_mx = true;
};

// Apply `patch` and return the resulting state — the binding hands both back
// in one call, so a caller never has to re-read the knobs it just set.
GemmConfigState configure(const GemmConfigPatch& patch);
GemmConfigState config_state();

// ---------------------------------------------------------------------------
// The dispatch's own vocabulary, for the Python tooling and the sweep's C++
// emitter — neither keeps a second copy of the names.
// ---------------------------------------------------------------------------

// Every (crosswise, operand widths) ladder's recipes, deduped on the dispatch
// key, in dispatch (manifest) order. Rows are (crosswise, ba, bb, cta, stages,
// kk, bm, bn, wm, wn, threads, smem); a row's numbers spell its canonical name,
// Tile_<bm>x<bn>x<kk>_W<wm>x<wn>_S<stages>, which is how the Python tooling
// joins rows with dataset recipe strings.
std::vector<std::vector<int>> tile_vocabulary();

// The TileClass spellings, in enum order — what a row's cta ordinal expands to
// in the compiled-in tables (the GENERATED block's paste target).
std::vector<const char*> tile_class_names();

}  // namespace gemm
}  // namespace astrai
