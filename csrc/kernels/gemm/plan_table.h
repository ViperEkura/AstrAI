#pragma once
// AOT dispatch table: measured best-recipe rows per (shape band, dtype class,
// layout class) that plan_gemm consults. Rows are data — the launch ladders
// resolve a row's CTA class through the manifest, so this header holds no
// kernel pointers or registration. Sources, override first: a runtime table
// file (ASTR_GEMM_TABLE=/path/to/rows.txt), so tile tuning never needs a
// rebuild, then the compiled-in GENERATED rows below (paste what the
// measurement script emits; the script never writes source). An empty table
// makes every lookup miss, and dispatch falls through to the degraded rows.
// The sweep times the fused-linear (NT) layout, so pasted rows carry crosswise
// 0 and the other layout classes take that same degraded fallback. Row files
// and their field order: parse_plan_table_file below; measurements, sweeps and
// the open questions: AGENTS.md (Plan table tuning log).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "common/device.cuh"
#include "policy.cuh"

namespace astrai {
namespace gemm {

// Ring K a row carries when its file omits the field. Per row rather than
// global because the manifest holds kK twins (policy.cuh) and the measured
// winner is kK=32 on most shapes but not all.
static constexpr int kTableRowK = 64;
// Fields the parser reads: the current row, the 9-field one the sweep
// scripts still emit (its absent k field keeps kTableRowK), the 12-field
// form that adds the contract-depth band, and the 13-field form that adds
// the wave gate.
static constexpr int kRowFields = 10;
static constexpr int kRowFieldsLegacyK = 9;
static constexpr int kRowFieldsKband = 12;
static constexpr int kRowFieldsWave = 13;
static constexpr int kRowFieldsWavePermille = 14;

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
// One tuned row. Every band is (min, max] — min EXCLUSIVE, max inclusive,
// 0 = unbounded (humming's dispatch-table convention) — and the same rule
// applies to the K band. perf_class is the GemmPerfClass id (-1 = any);
// crosswise the crosswise-operand count (-1 = any, see crosswise_of);
// raster 0 means plan_raster with this row's CTA geometry at launch time.
// kk is the row's ring K, defaulting to kTableRowK when a file omits it.
// The K band exists because the recipe flips with K — the wide CTA needs K
// past ~256 (its ring is only filled by the prologue for short K) while the
// kK=32 twin wins short K hardest — so without it the two demands would
// collide in one (M, N) cell and the row would have to lose one of them.
//
// min_ctas_per_sm / min_wave_permille are the row's WAVE GATES: the row
// matches only while its own CTA tile's grid — ceil(m/bm) * ceil(n/bn) *
// batch — covers that much of the machine, the bare form being
// grid >= n * sms and the wave form grid * 1000 >= n * sms * resident, with
// resident priced per device from the row's own ring. 0 = no gate (every row
// that predates the field, so the format stays backward compatible). The wave
// form is the one to prefer: it is dimensionless, so it survives both a
// different SM count and a different smem per SM, where a literal M bound is
// calibrated to one device. Rows whose bound is NOT wave arithmetic — a
// measured latency crossover, or a table property like "a row above already
// answers this band" — keep literals and want a re-measure on a new device.
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
    int min_ctas_per_sm = 0;
    int min_wave_permille = 0;
};

// CTA geometry of one class — the shapes dispatch_tile resolves from the
// manifest, so a row's smem budget can be priced here. Read off policy.cuh's
// kTileClassCta (which static_asserts itself against the tiles' CTA shapes),
// so the numbers cannot drift from what the ladders instantiate.
inline constexpr void plan_row_geometry(TileClass cta, int& bm, int& bn) {
    bm = kTileClassCta[(int)cta][0];
    bn = kTileClassCta[(int)cta][1];
}

// Everything one plan decision is priced against, assembled once per launch
// and passed as one value: the problem's shape, the class it keys the table
// on, the operand widths the ring and the load path depend on, and the
// device. A struct because these facts used to travel as eight positional
// arguments, each planner entry in its own order, with GemmParams and loose
// scalars carrying the same numbers — a call site read
// `plan_row_for(rows, count, m, n, perf, crosswise, k, ba, bb, dev, batch)`
// and nothing at the call site said which argument was which. The device is
// the queried DeviceFacts rather than a planner-private view: a bare SM count
// cannot carry a wave bound (a wave is sms * resident, and resident is the
// minimum of three per-SM resource limits), and a second device type with
// most of the same fields only invited the two to drift.
struct PlanQuery {
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    int64_t batch = 1;
    int perf_class = -1;  // GemmPerfClass id; -1 matches any
    int crosswise = 0;    // direct-load operand count, see gemm_dispatch
    int ba = 2;           // operand element bytes
    int bb = 2;
    DeviceFacts dev{};
};

// CTAs of one plan's tile that fit on an SM, or 0 when the plan cannot be
// priced (no device facts) or its ring cannot be launched at all (the same
// ceiling plan_from_row rejects past). Two limits, minimum wins: the device's
// smem per SM over the plan's ring, and the launch-bounds hint —
// __launch_bounds__(threads, hint) is what makes the compiler fit `hint` CTAs
// into the register file, so the hint is a FLOOR on the real count and not the
// count itself (the exact figure needs
// cudaOccupancyMaxActiveBlocksPerMultiprocessor, a driver query a pure planner
// cannot make; the test asserts the floor instead). A thread term is left out
// on purpose: no manifested tile's thread ceiling is under its register floor
// (see DeviceFacts).
// The direction of that floor matters: resident_model <= resident_true puts
// the wave threshold at or BELOW the physical one, so a device that packs more
// CTAs of this ring than the hint would fire the gate early. It is exact for
// the 512-thread tiles, where the register file is what binds (64 regs x 512
// threads x 2 = 65536 of 64K, on every supported part) — which is why the
// only gated ring today is the 128x128 kK=32 one.
inline int plan_resident_ctas(TileClass cta, int stages, int kk,
                              const PlanQuery& q) {
    const DeviceFacts& dev = q.dev;
    if (dev.smem_per_sm <= 0 || dev.regs_per_sm <= 0) return 0;
    int bm = 0, bn = 0;
    plan_row_geometry(cta, bm, bn);
    const int ring = ring_smem_bytes(bm, bn, kk, stages, q.ba, q.bb);
    if (ring > dev.smem_max) return 0;
    return std::min(dev.smem_per_sm / ring, min_ctas_for_ring(ring));
}

// First matching row wins (generated tables are emitted non-overlapping; the
// override file can shadow the builtin rows when it precedes them). q.k <= 0
// asks for the open-K reading, which is how the degraded rows and callers with
// no contract depth still resolve. q.dev.sms <= 0 means "no device facts": a
// gated row is then skipped rather than guessed at, and the lookup still ends
// at the degraded rows, so planning stays total.
inline const TableRow* plan_row_for(const TableRow* rows, int count,
                                    const PlanQuery& q) {
    for (int i = 0; i < count; ++i) {
        const TableRow& r = rows[i];
        if (q.m <= r.m_min) continue;
        if (r.m_max != 0 && q.m > r.m_max) continue;
        if (q.n <= r.n_min) continue;
        if (r.n_max != 0 && q.n > r.n_max) continue;
        if (r.perf_class != -1 && r.perf_class != q.perf_class) continue;
        if (r.crosswise != -1 && r.crosswise != q.crosswise) continue;
        // An open row (both bounds 0) matches any K, including the k <= 0
        // callers; a bounded row only matches a real depth.
        if (r.k_min != 0 || r.k_max != 0) {
            if (q.k <= 0) continue;
            if (q.k <= r.k_min) continue;
            if (r.k_max != 0 && q.k > r.k_max) continue;
        }
        // Wave gates: the row's own geometry prices the grid, so a row cannot
        // state a fill it could not itself satisfy. min_ctas_per_sm is the
        // bare CTAs-per-SM form; min_wave_permille the wave form, whose
        // resident term is priced from the row's ring on this device.
        if (r.min_ctas_per_sm > 0 || r.min_wave_permille > 0) {
            if (q.dev.sms <= 0) continue;
            int bm, bn;
            plan_row_geometry(r.cta, bm, bn);
            const int64_t grid = ((q.m + bm - 1) / bm) * ((q.n + bn - 1) / bn) *
                                 q.batch;
            if (r.min_ctas_per_sm > 0 &&
                grid < (int64_t)r.min_ctas_per_sm * q.dev.sms)
                continue;
            if (r.min_wave_permille > 0) {
                const int resident = plan_resident_ctas(r.cta, r.stages, r.kk, q);
                if (resident <= 0) continue;
                if (grid * 1000 <
                    (int64_t)r.min_wave_permille * q.dev.sms * resident)
                    continue;
            }
        }
        return &r;
    }
    return nullptr;
}

// Row file format: one row per line, whitespace-separated
//   m_min m_max n_min n_max perf_class crosswise cta stages raster
//   [k [k_min k_max [min_ctas_per_sm [min_wave_permille]]]]
// '#' starts a comment; perf_class 0..3 or -1; crosswise 0..2 or -1; cta is the
// TileClass ordinal (the policy.cuh enum order); stages 2..5, of which only the
// 64x64 kK=64 geometry carries s4/s5 (plan_from_row rejects a deeper ring
// anywhere else, and no compiled-in row names one — s2..s5 land within 1-2% at
// that tile). The trailing 'k' defaults to kTableRowK, which is what the sweep
// scripts leave off.
// The trailing forms are additive — a row that omits them behaves exactly as
// it did before they existed, which is what keeps hand-edited tuning files and
// older sweeps valid. Invalid lines are warn-and-skip: a malformed row must
// never block a launch the fallback would serve.

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
        fields != kRowFieldsKband && fields != kRowFieldsWave &&
        fields != kRowFieldsWavePermille)
        return "field count";
    if (row.m_min < 0 || row.n_min < 0) return "band min < 0";
    if (!row_k_supported(row.kk)) return "k (want 32 or 64)";
    if (!row_band_ok(row.m_min, row.m_max)) return "m band (max < min)";
    if (!row_band_ok(row.n_min, row.n_max)) return "n band (max < min)";
    if (!row_band_ok(row.k_min, row.k_max)) return "k band (max < min)";
    if (row.k_min < 0 || row.k_max < 0) return "k band min < 0";
    if (row.min_ctas_per_sm < 0) return "min_ctas_per_sm < 0";
    if (row.min_wave_permille < 0) return "min_wave_permille < 0";
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
        int min_ctas_per_sm = 0;
        int min_wave_permille = 0;
        const int got =
            std::sscanf(line,
                        " %lld %lld %lld %lld %d %d %d %d %d %d %lld %lld %d %d",
                        &m_min, &m_max, &n_min, &n_max, &perf_class, &crosswise,
                        &cta, &stages, &raster, &kk, &k_min, &k_max,
                        &min_ctas_per_sm, &min_wave_permille);
        if (got == EOF) continue;  // blank or comment-only line
        // cta is read as the TileClass ordinal, so it is the one field checked
        // before there is a row to validate; plan_row_error takes the rest.
        if (!in_range(cta, 0, kTileClassCount - 1)) {
            warn_bad_row(path, lineno, "cta index");
            continue;
        }
        const TableRow row{
            static_cast<TileClass>(cta), m_min, m_max, n_min, n_max,
            perf_class, crosswise, stages, raster, kk, k_min, k_max,
            min_ctas_per_sm, min_wave_permille
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
// Measured rows, distilled (2026-09-10, power-of-2 grid sweep over M, N, K in
// 32..4096): 42 rows merged to 14. A row here is a FLOOR, not an optimum —
// every one is kK=64 with cta <= 2 while the manifest carries more (the kK=32
// twin of the same band measures ~1.85x on narrow N, and the warp tiling is
// not addressable from a row at all: the dispatch key is class + stages + kK).
// One table per dtype class, so a row tuned for one operand pair cannot fire
// on another; the sweeps, the merge proof and the per-shape numbers are in
// AGENTS.md (Plan table tuning log).
//
// First match wins, so order matters: measured additions sit in front of the
// rows they shadow.
static constexpr TableRow kBuiltinPlanW16A16[] = {
    // Ring-residency rows (2026-09-11, RTX 4090), all on the 128x128 kK=32 s2
    // ring: its 48KB ring keeps TWO CTAs resident where the kK=64 twin's 96KB
    // keeps one, so the epilogue (which scatters through the reclaimed rings)
    // overlaps instead of being exposed, and its 16 warps of 32x32 double the
    // warps per partition at the same 64-register budget. Worth 7-14% on the
    // large shapes; sweeps and per-shape numbers in AGENTS.md's tuning log.
    // The M<=512 band keeps kK=64 s2 instead: there the K loop is too short
    // for the 64x64 CTA's 3 resident CTAs to lose.
    {TileClass::kBig128, 512, 0, 4096, 0, 0, 0, 2, 0, 32},
    // The wave bound of the narrow-N band (N <= 1536), replacing an M literal
    // that was hand-calibrated twice and wrong twice — the M that fills the
    // machine moves with N (grid = m_tiles * n_tiles), so no literal holds it.
    // Measured against the 64x64 kK=32 row below on a 128-SM part (256 slots
    // at this ring's resident 2): 288 CTAs = 1.13 waves and 312 = 1.22 and
    // 336 = 1.31 all went to the 64x64 tile (up to 23% ahead), 360 = 1.41 and
    // 384 = 1.50 to this one. So 1360 permille, and resident is priced from
    // the ring per device (plan_resident_ctas), so a part that packs four CTAs
    // of this ring moves the crossover with it. Table and controls: doc.
    //
    // N in (1536,3072] stays on the literal row below: the wave rule splits
    // 2-2 across the four mid-N points measured — and the two it gets WRONG
    // are the two best-filled grids (2560x2560 at 1.56 waves is 15% for the
    // 64x64 tile), so fill is not what decides that band. Left open rather
    // than guessed at; the doc records the four points.
    {TileClass::kBig128, 0, 0, 1024, 1536, 0, 0, 2, 0, 32, 0, 0, 0, 1360},
    // The mid-N half of that band keeps its calibrated literal (29 M-tiles,
    // see above) and its bare CTAs-per-SM gate.
    {TileClass::kBig128, 3712, 0, 1536, 3072, 0, 0, 2, 0, 32, 0, 0, 2},
    {TileClass::kBig128, 512, 0, 3072, 4096, 0, 0, 2, 0, 32},
    // Small M drops the wide band to the 64x64 tile: a 128x128 grid is
    // ceil(M/128) x n_tiles — 32 CTAs at M <= 128 — and streaming B from a
    // thin grid costs more than the big tile's reuse pays. The 384/512 flip
    // and the kk split are measured literals, not wave arithmetic (at M 512
    // the big tile wins on a half wave); the crossover table lives in the
    // tuning log, per the TableRow comment on literal vs wave bounds.
    {TileClass::kSmall64, 0, 128, 3072, 0, 0, 0, 3, 0, 64, 4096, 0},
    {TileClass::kSmall64, 0, 128, 3072, 0, 0, 0, 3, 0, 32, 2048, 0},
    {TileClass::kSmall64, 0, 384, 3072, 0, 0, 0, 3, 0, 32, 2048, 0},
    {TileClass::kBig128, 384, 512, 3072, 0, 0, 0, 2, 0, 32, 2048, 0},
    // The same residency effect as a k-tile depth: N in (768,1536] resolved to
    // the kK=64 small tile below (64KB ring, one resident CTA) where the kK=32
    // twin is 32KB. M > 1536 keeps the big-tile row (5.3% ahead there) and
    // M <= 128 keeps kK=64 (the grid is too thin for residency to pay and the
    // deeper ring amortizes better).
    {TileClass::kSmall64, 128, 1536, 768, 1536, 0, 0, 3, 0, 32},
    // Same swap for N in (1536,3072] and N <= 768, which resolved to the kK=64
    // floors below (64KB ring vs the twin's 32KB): worth 5-37% where the grid
    // is thin. The M bounds keep the rows off the shapes where kK=64 measured
    // ahead instead (3072x2048x1536 +2%, 256x768x3072 +4%).
    {TileClass::kSmall64, 384, 1024, 1536, 3072, 0, 0, 3, 0, 32},
    {TileClass::kSmall64, 384, 0, 0, 768, 0, 0, 3, 0, 32},
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
    // The wide CTA (128x256, resident 1) needs >= 1 wave to pay off — measured
    // <0.5 wave 0.79x, >=1 wave 1.08x, >=4 waves 1.42x against the small CTA —
    // so these bands start where M/128 * N/256 >= 128, i.e. one wave here.
    // Every wide row also carries the K > 512 band: that crossover is epilogue
    // dependent, so it takes the value that holds under both per-tensor and
    // per-row/per-channel scales (a strict win rather than one a cheap-scale
    // harness reads as a regression).
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
// dev/batch are the running device and the problem's batch, the inputs a
// row's wave gates need; dev.sms <= 0 (no device facts) skips gated rows
// instead of inventing a count. ba/bb are the operand widths.
inline const TableRow* plan_table_lookup(const PlanQuery& q) {
    const char* env = std::getenv("ASTR_GEMM_TABLE");
    if (env != nullptr && std::strcmp(env, "-") == 0) return nullptr;
    const std::vector<TableRow>& rows = plan_table_override_rows();
    if (const TableRow* row =
            plan_row_for(rows.data(), (int)rows.size(), q);
        row != nullptr)
        return row;
    int count = 0;
    const TableRow* builtin = builtin_plan_table(q.perf_class, count);
    if (builtin == nullptr) return nullptr;
    return plan_row_for(builtin, count, q);
}

// The planner's row source: override file first, then the compiled-in rows.
inline std::optional<TableRow> table_row(const PlanQuery& q) {
    if (const TableRow* row = plan_table_lookup(q); row != nullptr) return *row;
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
    // Only the M band decides here: the degraded rows are open on N and K with
    // -1 keys and carry no gate, so the rest of the query is left at its
    // defaults (k = 0 asks the open-K reading, and an empty device skips
    // nothing because nothing is gated).
    PlanQuery q;
    q.m = m;
    if (const TableRow* row = plan_row_for(kDegradedPlanRows, 3, q);
        row != nullptr)
        return *row;
    return kDegradedPlanRows[0];  // the degenerate m=0 matches no band
}

}  // namespace gemm
}  // namespace astrai
