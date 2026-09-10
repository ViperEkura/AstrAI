#pragma once
// AOT dispatch table: measured best-recipe rows per (shape band, dtype
// class, layout class) that plan_gemm consults. Rows are data — the launch
// ladders resolve a row's CTA class through the manifest, so this header
// holds no kernel pointers or registration. Sources, override first:
//   - a runtime table file (ASTR_GEMM_TABLE=/path/to/rows.txt), so tile
//     tuning never needs a rebuild;
//   - the compiled-in GENERATED rows below (paste the row file the
//     measurement script emits; the script never writes source).
// An empty table makes every lookup miss and dispatch falls through to the
// degraded band rows.
// The sweep times the fused-linear (NT) layout, so pasted rows carry
// crosswise 0: non-NT shapes (TT, TN, mixed dual-row-major) take the same
// degraded fallback.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "policy.cuh"

namespace astrai {
namespace gemm {

// Ring K a row carries when its file omits the field, and the depth the
// forced-recipe knob pins. The k-tile depth is a row field now that the
// manifest holds kK twins (policy.cuh): the measured winner is kK=32 on most
// shapes, but not on all, so it has to be per row.
static constexpr int kTableRowK = 64;
// Fields the parser reads: the current row, and the 9-field one the sweep
// scripts still emit (its absent k field keeps kTableRowK).
static constexpr int kRowFields = 10;
static constexpr int kRowFieldsLegacyK = 9;

// Upper bound of the perf_class field. The GemmPerfClass ids live in gemm.cuh,
// which includes this header, so the enum cannot be named here: this mirrors
// its last enumerator (kF8A8).
static constexpr int kMaxPerfClass = 3;

// The k-tile depths the manifests actually carry as tiles. A row naming any
// other depth would match no tile and launch nothing, so plan_from_row rejects
// it and the degraded bands serve the shape.
inline constexpr bool row_k_supported(int kk) { return kk == 32 || kk == 64; }

// The ring depths the manifests carry as tiles — the stages twin of
// row_k_supported, enumerated rather than written as a range so that a gap in
// the set cannot be admitted as "still inside 2..5". Only the 64x64 class has
// s4/s5 (policy.cuh): whether a given operand pair has room for the ring is
// not this header's business, it is plan_from_row's smem gate, which checks
// against the real device ceiling.
inline constexpr bool row_stages_supported(int stages) {
    return stages == 2 || stages == 3 || stages == 4 || stages == 5;
}
// One tuned row. Bands are (min, max] on M and N — min exclusive,
// max inclusive, 0 = unbounded (humming's dispatch-table convention);
// a row matches when m > m_min && (m_max == 0 || m <= m_max) and the
// same for n. perf_class is the GemmPerfClass id (0 W16A16 / 1 W8A16 /
// 2 W8A8 / 3 F8A8; -1 = any): the same band can price a different
// recipe per dtype class. crosswise is the crosswise-operand count of
// the problem (0 = dual-congruous NT, 1 = TT, the NN swap and the mixed
// dual-row-major NN case, 2 = TN — the
// (trans_a ? 1 : 0) + (trans_b ? 0 : 1) of gemm_dispatch; -1 = any).
// raster: 0 = plan_raster with this row's CTA geometry at launch time.
// kk: the row's ring K; a row file that omits the trailing field keeps
// kTableRowK.
struct TableRow {
    TileClass cta;
    int64_t m_min;
    int64_t m_max;
    int64_t n_min;
    int64_t n_max;
    int perf_class;
    int crosswise;
    int stages;
    int raster;
    int kk = kTableRowK;
};

// CTA geometry of one class — the shapes dispatch_tile resolves from the
// manifest, so a row's smem budget can be priced here. Read off policy.cuh's
// kTileClassCta (which static_asserts itself against the tiles' CTA shapes),
// so the numbers cannot drift from what the ladders instantiate.
inline constexpr void plan_row_geometry(TileClass cta, int& bm, int& bn) {
    bm = kTileClassCta[(int)cta][0];
    bn = kTileClassCta[(int)cta][1];
}

// First matching row wins (generated tables are emitted non-overlapping;
// the override file can shadow the builtin rows when it precedes them).
inline const TableRow* plan_row_for(const TableRow* rows, int count,
                                    int64_t m, int64_t n, int perf_class,
                                    int crosswise) {
    for (int i = 0; i < count; ++i) {
        const TableRow& r = rows[i];
        if (m <= r.m_min) continue;
        if (r.m_max != 0 && m > r.m_max) continue;
        if (n <= r.n_min) continue;
        if (r.n_max != 0 && n > r.n_max) continue;
        if (r.perf_class != -1 && r.perf_class != perf_class) continue;
        if (r.crosswise != -1 && r.crosswise != crosswise) continue;
        return &r;
    }
    return nullptr;
}

// Row file format: one row per line, whitespace-separated
//   m_min m_max n_min n_max perf_class crosswise cta stages raster [k]
// ('#' starts a comment; 'perf_class' 0..3 / -1 any; 'crosswise'
// 0..2 / -1 any; 'cta' is the TileClass ordinal (0..kTileClassCount-1, i.e.
// the policy.cuh enum order);
// 'stages' 2..5 (only the 64x64 kK=64 geometry carries s4/s5, and only it has
// the smem room for them: plan_from_row rejects a deeper ring anywhere else.
// The deep rings are a dead end in practice — s2..s5 land within 1-2% at that
// tile, so no compiled-in row names one); the trailing 'k' is optional and
// defaults to kTableRowK,
// which is what the sweep scripts leave off). Invalid lines are warn-and-skip:
// tuning files are hand-edited between sweeps, and a malformed row must never
// block a launch the fallback would serve.

// lo <= v <= hi, for the fields whose legal values are a contiguous interval.
inline constexpr bool in_range(int v, int lo, int hi) {
    return v >= lo && v <= hi;
}

// A band is (min, max]: min exclusive, 0 = unbounded, so the sentinel stays
// out of the ordering test. max == min is an empty band — legal, and simply
// never matched.
inline constexpr bool row_band_ok(int64_t min, int64_t max) {
    return max == 0 || max >= min;
}

// Names the first field outside its range, or nullptr when the row is well
// formed: one check per line, because the warning has to say which column of
// a hand-edited file is wrong.
inline const char* plan_row_error(const TableRow& row, int fields) {
    if (fields != kRowFields && fields != kRowFieldsLegacyK)
        return "field count";
    if (row.m_min < 0 || row.n_min < 0) return "band min < 0";
    if (!row_k_supported(row.kk)) return "k (want 32 or 64)";
    if (!row_band_ok(row.m_min, row.m_max)) return "m band (max < min)";
    if (!row_band_ok(row.n_min, row.n_max)) return "n band (max < min)";
    if (!in_range(row.perf_class, -1, kMaxPerfClass)) return "perf_class";
    if (!in_range(row.crosswise, -1, 2)) return "crosswise (-1..2)";
    if (!row_stages_supported(row.stages)) return "stages (want 2..5)";
    return nullptr;
}

// Warn-and-skip one bad line: the parse continues, so a hand-edit typo costs
// that row and not the table.
inline void warn_bad_row(const std::string& path, int lineno, const char* why) {
    std::fprintf(stderr, "[gemm-plan-table] %s:%d: ignoring row: bad %s\n",
                 path.c_str(), lineno, why);
}

inline bool parse_plan_table_file(const std::string& path,
                                  std::vector<TableRow>& rows) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) return false;
    char line[256];
    int lineno = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        ++lineno;
        if (char* hash = std::strchr(line, '#'); hash != nullptr) *hash = '\0';
        long long m_min, m_max, n_min, n_max;
        int perf_class, crosswise, cta, stages, raster;
        // sscanf leaves a variable alone when its conversion fails, so a row
        // that omits the trailing k field keeps the default here: the legacy
        // field count needs no repair pass.
        int kk = kTableRowK;
        const int got =
            std::sscanf(line, " %lld %lld %lld %lld %d %d %d %d %d %d", &m_min,
                        &m_max, &n_min, &n_max, &perf_class, &crosswise, &cta,
                        &stages, &raster, &kk);
        if (got == EOF) continue;  // blank or comment-only line
        // cta is read as the TileClass ordinal, so it is the one field checked
        // before there is a row to validate; plan_row_error takes the rest.
        if (!in_range(cta, 0, kTileClassCount - 1)) {
            warn_bad_row(path, lineno, "cta index");
            continue;
        }
        const TableRow row{
            static_cast<TileClass>(cta), m_min, m_max, n_min, n_max,
            perf_class, crosswise, stages, raster, kk
        };
        if (const char* bad = plan_row_error(row, got); bad != nullptr) {
            warn_bad_row(path, lineno, bad);
            continue;
        }
        rows.push_back(row);
    }
    std::fclose(f);
    return true;
}

// Cache of the override file, re-parsed only when the env path changes
// (a single setenv per process in practice; the parse result is
// idempotent, so a concurrent writer races benignly like device_facts).
inline const std::vector<TableRow>& plan_table_override_rows() {
    static std::vector<TableRow> rows;
    static std::string loaded_path;
    const char* env = std::getenv("ASTR_GEMM_TABLE");
    const std::string path = env != nullptr ? std::string(env) : std::string();
    if (path != loaded_path) {
        rows.clear();
        if (!path.empty() && path != "-") parse_plan_table_file(path, rows);
        loaded_path = path;
    }
    return rows;
}

// BEGIN GENERATED
// Measured power-of-2 grid table (2026-09-10): a band-search partition of a
// sweep over M, N, K in 32..4096 powers of two, all seven dtype combos / four
// perf classes (gen_plan_table.py --full-coverage; the band-search pass that
// cut the M bands has since been dropped from the script), emitted as
// 42 rows and merged down to these 14 (abutting same-recipe rectangles
// joined, the catch-alls the open last N band already shadows dropped;
// verified decision-identical over 80656 probe points x 4 classes x 2
// crosswise counts, so the merge costs nothing at dispatch).
// The distillate these rows replace keyed the recipe on one N split at 1280;
// the measured recipe depends on N far more strongly than that for large M.
// It sent M>2560, N>1280 to the narrow CTA where the big CTA is 1.40x faster
// (4096x4096x4096 w16a16 101 -> 142 TFLOPS), and M>2560, N<=1280 to the big
// CTA where the small CTA is up to 3.6x faster (4096x64x4096 16.5 -> 57) — a
// 128-wide N tile wastes half its mma on a 64-column problem; the quantized
// classes had no rows at all and fell to the degraded bands. Measured on the
// 42-row form (validate_plan_table.py, interleaved A/B, 26 holdout shapes x 6
// combos): grid 1.199x, LLM shape list 1.097x, combined 1.109x; worst
// per-shape regression 0.84x.
// A floor, not an optimum: every row here is kK=64 with cta<=2, while the
// manifest carries more. On the narrow-N bands the kK=32 twin of the class
// these rows already name measures ~1.85x (RTX 4090, 4096x256x4096 w16a16
// 71 -> 131 TFLOPS), and the warp tiling is not addressable from a row at all
// (the dispatch key is class + stages + kK, see above). Rows naming either
// come from a row file, not from here.
static constexpr TableRow kBuiltinPlanTable[] = {
    {TileClass::kSmall64, 0, 0, 0, 768, 0, 0, 3, 0},
    {TileClass::kSmall64, 0, 768, 768, 1536, 0, 0, 3, 0},
    {TileClass::kBig128, 768, 0, 768, 1536, 0, 0, 2, 0},
    {TileClass::kSmall64, 0, 384, 1536, 3072, 0, 0, 3, 0},
    {TileClass::kBig128, 384, 0, 1536, 0, 0, 0, 2, 0},
    {TileClass::kSmall64, 0, 96, 3072, 0, 0, 0, 3, 0},
    {TileClass::kNarrow128x64, 96, 384, 3072, 0, 0, 0, 2, 0},
    {TileClass::kSmall64, 0, 0, 0, 1536, 1, 0, 3, 0},
    {TileClass::kSmall64, 0, 768, 1536, 3072, 1, 0, 3, 0},
    {TileClass::kBig128, 768, 0, 1536, 3072, 1, 0, 2, 0},
    {TileClass::kSmall64, 0, 384, 3072, 0, 1, 0, 3, 0},
    {TileClass::kBig128, 384, 0, 3072, 0, 1, 0, 2, 0},
    {TileClass::kSmall64, 0, 0, 0, 0, 2, 0, 3, 0},
    {TileClass::kSmall64, 0, 0, 0, 0, 3, 0, 3, 0},
};
// END GENERATED

static constexpr int kBuiltinPlanTableCount = (int)(sizeof(kBuiltinPlanTable) / sizeof(TableRow));

// Override file first, then the compiled-in rows. ASTR_GEMM_TABLE="-"
// is the explicit "AOT off" escape hatch: neither override nor builtin
// rows, so dispatch falls through to the degraded band rows (dev/bench).
inline const TableRow* plan_table_lookup(const GemmParams& p, int perf_class,
                                         int crosswise) {
    const char* env = std::getenv("ASTR_GEMM_TABLE");
    if (env != nullptr && std::strcmp(env, "-") == 0) return nullptr;
    const std::vector<TableRow>& rows = plan_table_override_rows();
    if (const TableRow* row = plan_row_for(rows.data(), (int)rows.size(), p.m,
                                           p.n, perf_class, crosswise);
        row != nullptr)
        return row;
    return plan_row_for(kBuiltinPlanTable, kBuiltinPlanTableCount, p.m, p.n,
                        perf_class, crosswise);
}

// The planner's row source: override file first, then the compiled-in rows.
inline std::optional<TableRow> table_row(const GemmParams& p, int perf,
                                         int crosswise) {
    if (const TableRow* row = plan_table_lookup(p, perf, crosswise);
        row != nullptr)
        return *row;
    return std::nullopt;
}

// Last-resort rows for a table miss with the model retired: the M band's
// dominant recipe from the full-coverage sweep (small for short M, narrow
// mid, big past mid) — a safe default, never best. Open N with -1 keys
// matches every shape, so planning stays a total function.
static constexpr TableRow kDegradedPlanRows[] = {
    {TileClass::kSmall64, 0, 512, 0, 0, -1, -1, 2, 0},
    {TileClass::kNarrow128x64, 512, 3072, 0, 0, -1, -1, 2, 0},
    {TileClass::kBig128, 3072, 0, 0, 0, -1, -1, 2, 0},
};

inline const TableRow& degraded_row_for(int64_t m) {
    if (const TableRow* row =
            plan_row_for(kDegradedPlanRows, 3, m, /*n=*/1, -1, -1);
        row != nullptr)
        return *row;
    return kDegradedPlanRows[0];  // the degenerate m=0 matches no band
}

}  // namespace gemm
}  // namespace astrai
