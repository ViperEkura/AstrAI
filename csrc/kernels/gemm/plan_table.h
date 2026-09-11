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
// Fields the parser reads: the current row, the 9-field one the sweep
// scripts still emit (its absent k field keeps kTableRowK), and the 12-field
// form that adds the contract-depth band.
static constexpr int kRowFields = 10;
static constexpr int kRowFieldsLegacyK = 9;
static constexpr int kRowFieldsKband = 12;

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
// k_min / k_max are the row's contract-depth band, same (min, max] rule as
// m and n, 0 = open. A row that omits both keeps an open band, so every
// existing row file and table keeps its meaning. The band exists because the
// recipe can flip with K: the wide CTA needs K past ~256 to beat the small
// one (a 128x256 tile's ring is only filled by the prologue for short K),
// while the kK=32 twin wins short K hardest. Without the band the two
// demands collide in one (M, N) cell and the row has to lose one of them.
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
    int64_t k_min = 0;
    int64_t k_max = 0;
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
// k <= 0 asks for the open-K reading, which is how the degraded rows and the
// callers that have no contract depth still resolve.
inline const TableRow* plan_row_for(const TableRow* rows, int count,
                                    int64_t m, int64_t n, int perf_class,
                                    int crosswise, int64_t k = 0) {
    for (int i = 0; i < count; ++i) {
        const TableRow& r = rows[i];
        if (m <= r.m_min) continue;
        if (r.m_max != 0 && m > r.m_max) continue;
        if (n <= r.n_min) continue;
        if (r.n_max != 0 && n > r.n_max) continue;
        if (r.perf_class != -1 && r.perf_class != perf_class) continue;
        if (r.crosswise != -1 && r.crosswise != crosswise) continue;
        // An open row (both bounds 0) matches any K, including the k <= 0
        // callers; a bounded row only matches a real depth.
        if (r.k_min != 0 || r.k_max != 0) {
            if (k <= 0) continue;
            if (k <= r.k_min) continue;
            if (r.k_max != 0 && k > r.k_max) continue;
        }
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
    if (fields != kRowFields && fields != kRowFieldsLegacyK &&
        fields != kRowFieldsKband)
        return "field count";
    if (row.m_min < 0 || row.n_min < 0) return "band min < 0";
    if (!row_k_supported(row.kk)) return "k (want 32 or 64)";
    if (!row_band_ok(row.m_min, row.m_max)) return "m band (max < min)";
    if (!row_band_ok(row.n_min, row.n_max)) return "n band (max < min)";
    if (!row_band_ok(row.k_min, row.k_max)) return "k band (max < min)";
    if (row.k_min < 0 || row.k_max < 0) return "k band min < 0";
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
        // that omits the trailing fields keeps the defaults here: the legacy
        // and k-less field counts need no repair pass.
        int kk = kTableRowK;
        long long k_min = 0, k_max = 0;
        const int got =
            std::sscanf(line, " %lld %lld %lld %lld %d %d %d %d %d %d %lld %lld",
                        &m_min, &m_max, &n_min, &n_max, &perf_class, &crosswise,
                        &cta, &stages, &raster, &kk, &k_min, &k_max);
        if (got == EOF) continue;  // blank or comment-only line
        // cta is read as the TileClass ordinal, so it is the one field checked
        // before there is a row to validate; plan_row_error takes the rest.
        if (!in_range(cta, 0, kTileClassCount - 1)) {
            warn_bad_row(path, lineno, "cta index");
            continue;
        }
        const TableRow row{
            static_cast<TileClass>(cta), m_min, m_max, n_min, n_max,
            perf_class, crosswise, stages, raster, kk, k_min, k_max
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
//
// v2 (2026-09-10, dense sweep + rectangle search): the first eight rows are
// measured additions in front of that floor, each one required to beat the
// row it shadows at EVERY sweep grid point inside its rectangle (min gain
// >= 1.00x, geomean >= 1.26x) — so a row here can only be an improvement.
//   1-2  pc2 wide CTA on the large-N rectangles the class-2 catch-all sent to
//        the 64x64 tile: 1.34-1.45x (w8a8 4096x11008x4096 304 -> 450 TFLOPS).
//   3-7  pc0 kK=32 on the narrow-N and mid-N bands: 1.31-1.78x geomean, up to
//        1.85x (w16a16 2048x512x4096 73.6 -> 136.3 TFLOPS). The kK=32 ring is
//        32KB at s3, under the 48KB two-CTA watermark, which is where the win
//        comes from; s2 and s3 measure within noise of each other.
//   8    pc1 big CTA on the large-M/small-N band the class-1 rows sent to the
//        64x64 tile: 1.27x.
// Validated interleaved against the pre-v2 table (validate_plan_table.py, 12
// holdout shapes x 7 combos): w16a16 -36.9/-39.2/-45.9% on three narrow-N
// holdouts, no regression above the ~4% per-shape noise floor (the one >=2%
// delta, f8a8_e4m3 on 2048x14336x4096, is the same tile in both tables).
// One builtin table per dtype class (GemmPerfClass): the class is the table,
// so a row tuned for one operand pair cannot fire on another. The single mixed
// table matched rows on a perf_class field instead, which made a row's reach a
// property of class ids alone — and int8 shared an id with fp8 until
// gemm_perf_class stopped bucketing it there, so every W8A8 row was dead while
// int8 dispatched on the F8A8 rows. Rows keep the field for the override-file
// path (one file still serves every class); inside a class table a row may
// only repeat its class's id or say -1, asserted below, so a typo cannot
// silently drop a row out of the table it sits in.
//
// First match wins, so order matters within a table: measured additions sit in
// front of the rows they shadow.
static constexpr TableRow kBuiltinPlanW16A16[] = {
    // 2026-09-11 power-of-2 grid sweep (M,N,K in 32..4096, GPU A/B over all
    // 512 points against the rows below): the kK=32 twin wins the whole
    // M,N >= 1024 region — -4.32% summed over the grid, 48 points better by
    // 5-44% (1024x1024x4096 111 -> 62us, 4096x4096x4096 1009 -> 941us); the
    // rest sit within the ~1us event-timer tick. kK=64 keeps the bands below,
    // where the 32-deep twin was not measured.
    {TileClass::kSmall64, 768, 0, 768, 1024, 0, 0, 3, 0, 32},
    {TileClass::kSmall64, 1024, 0, 1024, 4096, 0, 0, 3, 0, 32},
    {TileClass::kSmall64, 768, 1024, 3072, 4096, 0, 0, 3, 0, 32},
    // v2 (2026-09-10, dense sweep): kK=32 on the narrow- and mid-N bands where
    // the 64-deep ring spends issue slots the short K loop cannot use.
    {TileClass::kSmall64, 1536, 4096, 0, 768, 0, 0, 3, 0, 32},
    {TileClass::kSmall64, 768, 4096, 256, 768, 0, 0, 3, 0, 32},
    {TileClass::kSmall64, 384, 768, 1024, 3072, 0, 0, 3, 0, 32},
    {TileClass::kSmall64, 128, 384, 1536, 3072, 0, 0, 3, 0, 32},
    {TileClass::kSmall64, 128, 512, 3072, 0, 0, 0, 3, 0, 32},
    // v1 floor (2026-09-10, full-coverage sweep).
    {TileClass::kSmall64, 0, 0, 0, 768, 0, 0, 3, 0},
    {TileClass::kSmall64, 0, 768, 768, 1536, 0, 0, 3, 0},
    {TileClass::kBig128, 768, 0, 768, 1536, 0, 0, 2, 0},
    {TileClass::kSmall64, 0, 384, 1536, 3072, 0, 0, 3, 0},
    {TileClass::kBig128, 384, 0, 1536, 0, 0, 0, 2, 0},
    {TileClass::kSmall64, 0, 96, 3072, 0, 0, 0, 3, 0},
    {TileClass::kNarrow128x64, 96, 384, 3072, 0, 0, 0, 2, 0},
};

static constexpr TableRow kBuiltinPlanW8A16[] = {
    {TileClass::kBig128, 3072, 4096, 256, 2048, 1, 0, 2, 0, 64},
    {TileClass::kSmall64, 0, 0, 0, 1536, 1, 0, 3, 0},
    {TileClass::kSmall64, 0, 768, 1536, 3072, 1, 0, 3, 0},
    {TileClass::kBig128, 768, 0, 1536, 3072, 1, 0, 2, 0},
    {TileClass::kSmall64, 0, 384, 3072, 0, 1, 0, 3, 0},
    {TileClass::kBig128, 384, 0, 3072, 0, 1, 0, 2, 0},
};

static constexpr TableRow kBuiltinPlanW8A8[] = {
    // The wide CTA's win is a grid-fill property, not just a shape one: at
    // 128x256 per CTA the tile needs >= 1 wave of 128 SMs to pay off
    // (measured: <0.5 wave 0.79x, >=1 wave 1.08x, >=4 waves 1.42x against the
    // small CTA), so these bands start where M/128 * N/256 >= 128.
    //
    // Every wide row carries the K > 512 band. The crossover is epilogue
    // dependent — with per-tensor scales the wide tile is 1.24-1.27x *slower*
    // at K <= 256 and crosses over near 1K, while with the per-row/per-channel
    // scales W8A8 actually uses it is already even at K=32 (the scale loads
    // dominate the short-K epilogue either way) — so the band takes the
    // crossover that holds under both, which makes these rows a strict win
    // rather than a win that a benchmark harness with cheaper scales sees as a
    // regression.
    {TileClass::kWide128x256, 2048, 4096, 3072, 11008, 2, 0, 2, 0, 64, 512, 0},
    {TileClass::kWide128x256, 512, 4096, 8192, 11008, 2, 0, 2, 0, 64, 512, 0},
    // Mid-N at large M is the grid-sweep gap those bands leave: measured
    // 1.22-1.54x for K > 512 there, so the same tile and the same band.
    {TileClass::kWide128x256, 1024, 0, 1024, 0, 2, 0, 2, 0, 64, 512, 0},
    {TileClass::kSmall64, 0, 0, 0, 0, 2, 0, 3, 0},
};

static constexpr TableRow kBuiltinPlanF8A8[] = {
    // fp8 has no row of its own yet: the sweep's fp8 optima sat within the
    // small CTA's noise, and its roof is close (293 vs 313 TFLOPS measured on
    // this part), so the catch-all is the measured-best row on the grid.
    {TileClass::kSmall64, 0, 0, 0, 0, 3, 0, 3, 0},
};
// END GENERATED

// A builtin row must belong to the table it sits in: repeating the class id
// documents intent, -1 says "any", anything else means the row was filed
// under the wrong table and would be filtered out at lookup.
template <int Class>
constexpr bool builtin_rows_keyed_for(const TableRow* rows, int count) {
    for (int i = 0; i < count; ++i)
        if (rows[i].perf_class != -1 && rows[i].perf_class != Class)
            return false;
    return true;
}

#define ASTR_GEMM_ROWS_MATCH_CLASS(Table, Class)                              \
    static_assert(                                                            \
        builtin_rows_keyed_for<Class>(                                        \
            Table, (int)(sizeof(Table) / sizeof(TableRow))),                  \
        #Table " carries a row keyed for another dtype class")

ASTR_GEMM_ROWS_MATCH_CLASS(kBuiltinPlanW16A16, 0);
ASTR_GEMM_ROWS_MATCH_CLASS(kBuiltinPlanW8A16, 1);
ASTR_GEMM_ROWS_MATCH_CLASS(kBuiltinPlanW8A8, 2);
ASTR_GEMM_ROWS_MATCH_CLASS(kBuiltinPlanF8A8, 3);
#undef ASTR_GEMM_ROWS_MATCH_CLASS

// Builtin table for one dtype class; count receives its row count.
inline constexpr const TableRow* builtin_plan_table(int perf_class, int& count) {
    switch (perf_class) {
        case 0:
            count = (int)(sizeof(kBuiltinPlanW16A16) / sizeof(TableRow));
            return kBuiltinPlanW16A16;
        case 1:
            count = (int)(sizeof(kBuiltinPlanW8A16) / sizeof(TableRow));
            return kBuiltinPlanW8A16;
        case 2:
            count = (int)(sizeof(kBuiltinPlanW8A8) / sizeof(TableRow));
            return kBuiltinPlanW8A8;
        case 3:
            count = (int)(sizeof(kBuiltinPlanF8A8) / sizeof(TableRow));
            return kBuiltinPlanF8A8;
        default:
            count = 0;
            return nullptr;
    }
}

// Override file first (one file for every class, keyed by its perf_class
// column), then the class's own builtin table. ASTR_GEMM_TABLE="-"
// is the explicit "AOT off" escape hatch: neither override nor builtin
// rows, so dispatch falls through to the degraded band rows (dev/bench).
inline const TableRow* plan_table_lookup(const GemmParams& p, int perf_class,
                                         int crosswise) {
    const char* env = std::getenv("ASTR_GEMM_TABLE");
    if (env != nullptr && std::strcmp(env, "-") == 0) return nullptr;
    const std::vector<TableRow>& rows = plan_table_override_rows();
    if (const TableRow* row = plan_row_for(rows.data(), (int)rows.size(), p.m,
                                           p.n, perf_class, crosswise, p.k);
        row != nullptr)
        return row;
    int count = 0;
    const TableRow* builtin = builtin_plan_table(perf_class, count);
    if (builtin == nullptr) return nullptr;
    return plan_row_for(builtin, count, p.m, p.n, perf_class, crosswise, p.k);
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
