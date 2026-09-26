#pragma once
// AOT dispatch table: measured best-recipe rows per (shape band, dtype class,
// layout class). Sources, override first: ASTR_GEMM_TABLE rows file (tuning
// without a rebuild), then the compiled-in GENERATED rows (paste what the
// measurement script emits; it never writes source). An empty table misses
// every lookup and dispatch falls to the degraded rows. The sweep times the
// NT layout, so pasted rows carry crosswise 0. Field order: the parser below.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <utils/device.cuh>
#include <policy.cuh>

namespace astrai {
namespace gemm {

// Ring K when the row file omits the field (the manifest holds kK twins;
// the measured winner is kK=32 on most shapes but not all).
static constexpr int kTableRowK = 64;
// Field counts the parser reads: current, legacy 9 (no k), +k-band 12,
// +wave-gate 13, +wave-permille 14.
static constexpr int kRowFields = 10;
static constexpr int kRowFieldsLegacyK = 9;
static constexpr int kRowFieldsKband = 12;
static constexpr int kRowFieldsWave = 13;
static constexpr int kRowFieldsWavePermille = 14;

// Upper bound of the perf_class field, mirroring GemmPerfClass's last
// enumerator (kF8A8, policy.cuh — where the enum now lives).
static constexpr int kMaxPerfClass = 3;

// The k-tile depths the manifests carry; any other depth matches no tile
// and launches nothing, so it is rejected.
inline constexpr bool row_k_supported(int kk) { return kk == 32 || kk == 64; }

// Ring depths a row may name. Enumerated, not a range, so a gap cannot pass
// as "inside 2..5": s4/s5 stay parseable for old sweep files, but no ladder
// instantiates past s3 (the s4/s5 twins were a measured wash, removed).
inline constexpr bool row_stages_supported(int stages) {
    return stages == 2 || stages == 3 || stages == 4 || stages == 5;
}
// One tuned row. Bands are (min, max] — min exclusive, 0 unbounded — K band
// included. perf_class is the GemmPerfClass id, crosswise the operand count
// (-1 = any for both); raster 0 resolves through plan_raster at this row's
// geometry; kk defaults to kTableRowK. The K band exists because the recipe
// flips with K (wide CTA wants K > ~256, kK=32 twin wins short K hardest).
//
// The wave gates match only while this tile's grid covers that much of the
// machine: bare form grid >= n * sms, wave form grid * 1000 >= n * sms *
// resident (resident priced from the row's ring). 0 = no gate. Prefer the
// wave form — dimensionless, survives a different SM count or smem; literal
// bounds (a measured crossover, "a row above answers this band") are
// device-calibrated and want a re-measure elsewhere.
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

// CTA geometry of one class, off kTileClassCta (self-asserted against the
// tiles), so the numbers cannot drift from what the ladders instantiate.
inline constexpr void plan_row_geometry(TileClass cta, int& bm, int& bn) {
    bm = kTileClassCta[(int)cta][0];
    bn = kTileClassCta[(int)cta][1];
}

// Everything a plan decision is priced against is PlanQuery — now policy.cuh
// (the kernel-side vocabulary), so this header compiles against types the
// launchers already see.

// CTAs of one plan's tile per SM, or 0 when unpriced/unlaunchable. Minimum
// of smem-per-SM over the ring and the __launch_bounds__ hint — the hint is
// a FLOOR on the real count (the exact figure needs a driver query a pure
// planner cannot make), so resident_model <= resident_true fires the wave
// gate early, never late. Exact for the 512-thread tiles where the register
// file binds (64 regs x 512 x 2 = 64K) — the only gated ring today.
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

// First matching row wins (generated tables are non-overlapping). q.k <= 0
// is the open-K reading (degraded rows, no-depth callers); q.dev.sms <= 0
// skips gated rows rather than guessing, and the lookup still ends at the
// degraded rows — planning stays total.
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
// TileClass ordinal (the policy.cuh enum order); stages 2..5 parse, but no
// ladder instantiates a tile past s3 — plan_from_row rejects a deeper ring
// outright, so a stale sweep row naming s4/s5 falls to the next source. The
// trailing 'k' defaults to kTableRowK, which is what the sweep
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

// First out-of-range field, or nullptr: one check per line so the warning
// names the hand-edited column that is wrong.
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

// Warn-and-skip: a hand-edit typo costs one row, not the table.
inline void warn_bad_row(const std::string& path, int lineno, const char* why) {
    std::fprintf(stderr, "[gemm-plan-table] %s:%d: ignoring row: bad %s\n",
                 path.c_str(), lineno, why);
}

// One row-file line (label names the source in warnings). Mutated in place
// (the '#' comment cut); both the file and runtime-injection readers go
// through here so the two cannot drift.
inline void parse_plan_table_line(char* line, const char* label, int lineno,
                                  std::vector<TableRow>& rows) {
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
    if (got == EOF) return;  // blank or comment-only line
    // cta is read as the TileClass ordinal, so it is the one field checked
    // before there is a row to validate; plan_row_error takes the rest.
    if (!in_range(cta, 0, kTileClassCount - 1)) {
        warn_bad_row(label, lineno, "cta index");
        return;
    }
    const TableRow row{
        static_cast<TileClass>(cta), m_min, m_max, n_min, n_max,
        perf_class, crosswise, stages, raster, kk, k_min, k_max,
        min_ctas_per_sm, min_wave_permille
    };
    if (const char* bad = plan_row_error(row, got); bad != nullptr) {
        warn_bad_row(label, lineno, bad);
        return;
    }
    rows.push_back(row);
}

inline bool parse_plan_table_file(const std::string& path,
                                  std::vector<TableRow>& rows) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) return false;
    char line[256];
    int lineno = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        ++lineno;
        parse_plan_table_line(line, path.c_str(), lineno, rows);
    }
    std::fclose(f);
    return true;
}

// The same parser over in-memory row text: the runtime channel and the file
// path accept identical syntax. Returns the surviving row count.
inline int parse_plan_table_text(const std::string& text, const char* label,
                                 std::vector<TableRow>& rows) {
    const int before = (int)rows.size();
    std::string line;
    int lineno = 0;
    for (std::size_t pos = 0; pos <= text.size(); ++pos) {
        const char c = pos < text.size() ? text[pos] : '\n';
        if (c != '\n' && c != '\r') {
            line.push_back(c);
            continue;
        }
        ++lineno;
        line.push_back('\0');
        parse_plan_table_line(&line[0], label, lineno, rows);
        line.clear();
        // The trailing newline of the final chunk loops once more with an
        // empty string; an empty line parses to nothing, so the extra pass
        // is harmless.
    }
    return (int)rows.size() - before;
}

// One row tier's backing store: mutex-guarded, replaced wholesale (never
// mutated in place), lookups copy the row out under the lock — a concurrent
// install cannot dangle a pointer a launched plan holds. Two instances:
// override (the configure rows channel, outranks everything) and injected
// (the autotuner's measured winners; ranked below the env file, which is
// the experimenter's). `source` is the install spec, kept so the config
// state can re-install exactly (re-emitting parsed rows would lose the
// gates the text format cannot express).
class RowSource {
public:
    void set_from(std::string source, std::vector<TableRow> rows) {
        std::lock_guard<std::mutex> g(mutex_);
        rows_ = std::move(rows);
        source_ = std::move(source);
    }
    void clear() { set_from({}, {}); }
    std::string source() const {
        std::lock_guard<std::mutex> g(mutex_);
        return source_;
    }
    std::optional<TableRow> lookup(const PlanQuery& q) const {
        std::lock_guard<std::mutex> g(mutex_);
        const TableRow* row =
            plan_row_for(rows_.data(), (int)rows_.size(), q);
        if (row == nullptr) return std::nullopt;
        return *row;
    }
    size_t size() const {
        std::lock_guard<std::mutex> g(mutex_);
        return rows_.size();
    }

private:
    mutable std::mutex mutex_;
    std::vector<TableRow> rows_;
    std::string source_;
};

inline RowSource& plan_table_override_source() {
    static RowSource source;
    return source;
}

inline RowSource& plan_table_injected_source() {
    static RowSource source;
    return source;
}


// The planner-rank vocabulary, one place: the strings configure() takes
// and config_state() returns for GemmConfig::planner.
inline constexpr const char* kPlannerModeNames[] = {"table", "hybrid",
                                                    "model"};
inline constexpr int kPlannerModeCount = 3;
inline bool parse_planner_mode(const std::string& name, int& out) {
    for (int i = 0; i < kPlannerModeCount; ++i)
        if (name == kPlannerModeNames[i]) {
            out = i;
            return true;
        }
    return false;
}

// Legacy ASTR_GEMM_* env vars, read once on first touch; configure() writes
// the atomics directly and bypasses this.
inline void gemm_config_seed_once() {
    static const bool seeded = [] {
        GemmConfig& c = gemm_config();
        auto env = [](const char* name) {
            const char* e = std::getenv(name);
            return e == nullptr ? std::string() : std::string(e);
        };
        if (const std::string v = env("ASTR_GEMM_MODEL"); !v.empty())
            c.planner = std::atoi(v.c_str());
        if (const std::string v = env("ASTR_GEMM_PLAN"); !v.empty() && v != "0")
            c.log = 1;
        if (env("ASTR_GEMM_NO_TMA") == "1") c.tma_disabled = 1;
        if (env("ASTR_GEMM_NO_MX") == "1") c.mx_disabled = 1;
        if (const std::string v = env("ASTR_GEMM_TABLE"); !v.empty()) {
            if (v == "-") {
                c.table_off = 1;
            } else {
                std::vector<TableRow> rows;
                if (parse_plan_table_file(v, rows))
                    plan_table_override_source().set_from(v, std::move(rows));
            }
        }
        return true;
    }();
    (void)seeded;
}

// Resolved views (unset falls to the default, never to a later env read);
// the three launch-side knobs' resolved views are policy.cuh's.
inline int gemm_planner_mode() {
    gemm_config_seed_once();
    const int v = gemm_config().planner.load(std::memory_order_relaxed);
    return v < 0 ? 1 : v;  // default: hybrid (model fills what no row owns)
}
inline bool gemm_table_off() {
    gemm_config_seed_once();
    return gemm_config().table_off.load(std::memory_order_relaxed) > 0;
}

// BEGIN GENERATED
// Rows are the measured DIFF of the model, not full coverage: emitted only
// where a recipe beat the model's dispatch by >=2% in interleaved A/B
// (four rounds/point; the sweep's model reference runs last, at the hottest
// clock — 10-20% underprice on heavy shapes, measured 2026-09-14). Ties and
// unmeasured bands serve the model. Measured 2026-09-14, this box: sm_120 /
// RTX 5090 / 170 SMs, M {1..4096} x N {1024..28672} x K {1536,4096,8192},
// 2% production-semantics holdout.
//
// Crosswise F8A8 rows (2026-09-19): the wide CTA joined the byte ladders
// for crosswise staging, moving the L2 re-read wall — the 64x64 model pick
// streams 4.8GB of operands through L2 on the qkv cell (84.7% L2 SOL, 48%
// compute) vs 2.4GB (128-row) and 1.8GB (wide). m=16384, astrai_1b
// projections, four A/B rounds/point. NN rows band the canonicalized
// aspect; (16384,1536) splits by exact k band into square (k=1536) and
// mlp_down (k=6912). Bands are the measured points only. The kK=32 unlock
// replaced three rows after a four-round confirm: square NN/TT and mlp_down
// NN run the kK=32 big CTA (the kK=32 narrow lost everywhere, -8..-34%);
// kK=64 keeps the other nine points.
//
// A stale row is worse than none (2026-09-14 +33% lesson): the tier is
// signature-guarded by kBuiltinPlanMeasuredOn — mismatch serves nothing,
// chain runs override -> injected -> [no builtin] -> model -> degraded.
// Another part gets its own sweep, never these rows.

static constexpr std::array<TableRow, 29> kBuiltinPlanW16A16 = {{
    // Prepended so its band wins over the 128x128 kk32 row below, 3.0-3.2x
    // off here: n <= 256 gives that tile 40 blocks over 170 SMs where this
    // fills 160 (interleaved A/B, 2026-09-14, n=256, k {2048,4096},
    // m {1792..2560}; the band is the measured one).
    {TileClass::kSmall64, 1536, 3072, 0, 256, 0, 0, 3, 0, 64},
    {TileClass::kTall64x128, 3072, 0, 5120, 8576, 0, 0, 2, 0, 32},
    {TileClass::kTall64x128, 0, 12, 8576, 19840, 0, 0, 3, 0, 32},
    {TileClass::kTall64x128, 12, 96, 8576, 19840, 0, 0, 2, 0, 32},
    {TileClass::kTall64x128, 0, 4, 19840, 0, 0, 0, 3, 0, 32},
    {TileClass::kSmall64, 0, 768, 0, 1280, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 768, 1536, 0, 1280, 0, 0, 3, 0, 64},
    {TileClass::kBig128, 1536, 3072, 0, 1280, 0, 0, 3, 0, 32},
    {TileClass::kSmall64, 0, 384, 1280, 2816, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 384, 768, 1280, 2816, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 3072, 0, 1280, 2816, 0, 0, 2, 0, 32},
    {TileClass::kSmall64, 0, 192, 2816, 5120, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 192, 384, 2816, 5120, 0, 0, 3, 0, 64},
    {TileClass::kBig128, 384, 768, 2816, 5120, 0, 0, 2, 0, 64},
    {TileClass::kSmall64, 0, 24, 5120, 8576, 0, 0, 3, 0, 64},
    {TileClass::kSmall64, 24, 48, 5120, 8576, 0, 0, 2, 0, 64},
    {TileClass::kSmall64, 48, 96, 5120, 8576, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 96, 192, 5120, 8576, 0, 0, 3, 0, 64},
    {TileClass::kSmall64, 192, 384, 5120, 8576, 0, 0, 2, 0, 32},
    {TileClass::kNarrow128x64, 768, 1536, 5120, 8576, 0, 0, 2, 0, 32},
    {TileClass::kSmall64, 0, 4, 8576, 19840, 0, 0, 3, 0, 64},
    {TileClass::kSmall64, 4, 12, 8576, 19840, 0, 0, 2, 0, 32},
    {TileClass::kNarrow128x64, 1536, 3072, 8576, 19840, 0, 0, 2, 0, 64},
    {TileClass::kBig128, 3072, 0, 8576, 19840, 0, 0, 2, 0, 64},
    {TileClass::kSmall64, 0, 96, 19840, 0, 0, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 96, 192, 19840, 0, 0, 0, 3, 0, 64},
    {TileClass::kBig128, 192, 384, 19840, 0, 0, 0, 2, 0, 64},
    {TileClass::kNarrow128x64, 384, 768, 19840, 0, 0, 0, 2, 0, 64},
    {TileClass::kBig128, 768, 0, 19840, 0, 0, 0, 2, 0, 64},
}};
static constexpr std::array<TableRow, 23> kBuiltinPlanW8A16 = {{
    // Same narrow-N pathology and fix as the W16A16 row above: the 128x128
    // kk64 row below is 2.9-3.1x off at n=256, m 1792..2560 (A/B, 2026-09-14,
    // k=4096); the band is the measured one.
    {TileClass::kSmall64, 1536, 3072, 0, 256, 1, 0, 3, 0, 64},
    {TileClass::kTall64x128, 0, 96, 8576, 19840, 1, 0, 3, 0, 32},
    {TileClass::kTall64x128, 0, 4, 19840, 0, 1, 0, 2, 0, 32},
    {TileClass::kTall64x128, 96, 192, 19840, 0, 1, 0, 3, 0, 32},
    {TileClass::kNarrow128x64, 768, 1536, 0, 1280, 1, 0, 3, 0, 64},
    {TileClass::kBig128, 1536, 0, 0, 1280, 1, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 384, 768, 1280, 2816, 1, 0, 3, 0, 64},
    {TileClass::kSmall64, 768, 3072, 1280, 2816, 1, 0, 2, 0, 64},
    {TileClass::kNarrow128x64, 3072, 0, 1280, 2816, 1, 0, 2, 0, 32},
    {TileClass::kNarrow128x64, 192, 384, 2816, 5120, 1, 0, 3, 0, 64},
    {TileClass::kBig128, 384, 1536, 2816, 5120, 1, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 1536, 0, 2816, 5120, 1, 0, 2, 0, 32},
    {TileClass::kNarrow128x64, 96, 192, 5120, 8576, 1, 0, 3, 0, 64},
    {TileClass::kSmall64, 192, 768, 5120, 8576, 1, 0, 2, 0, 64},
    {TileClass::kNarrow128x64, 768, 0, 5120, 8576, 1, 0, 2, 0, 32},
    {TileClass::kSmall64, 0, 768, 8576, 19840, 1, 0, 2, 0, 64},
    {TileClass::kNarrow128x64, 768, 3072, 8576, 19840, 1, 0, 2, 0, 32},
    {TileClass::kBig128, 3072, 0, 8576, 19840, 1, 0, 3, 0, 64},
    {TileClass::kSmall64, 4, 96, 19840, 0, 1, 0, 2, 0, 64},
    {TileClass::kNarrow128x64, 96, 192, 19840, 0, 1, 0, 3, 0, 64},
    {TileClass::kBig128, 192, 384, 19840, 0, 1, 0, 3, 0, 64},
    {TileClass::kNarrow128x64, 384, 768, 19840, 0, 1, 0, 2, 0, 32},
    {TileClass::kBig128, 768, 0, 19840, 0, 1, 0, 3, 0, 64},
}};
static constexpr std::array<TableRow, 29> kBuiltinPlanW8A8 = {{
    // The narrow-N pathology of the W16A16/W8A16 rows above, same band and
    // same winner: the 128x128 kk64 row below measured 1.56-1.74x off at n=256,
    // m 1792..2560 (interleaved A/B, 2026-09-14, k=4096).
    {TileClass::kSmall64, 1536, 3072, 0, 256, 2, 0, 3, 0, 64},
    {TileClass::kSmall64, 12, 768, 0, 1280, 2, 0, 3, 0, 64},
    {TileClass::kBig128, 1536, 3072, 0, 1280, 2, 0, 3, 0, 64},
    {TileClass::kWide128x256, 3072, 0, 0, 1280, 2, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 192, 1280, 2816, 2, 0, 3, 0, 64},
    {TileClass::kSmall64, 192, 384, 1280, 2816, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 768, 1536, 1280, 2816, 2, 0, 3, 0, 64},
    {TileClass::kWide128x256, 1536, 3072, 1280, 2816, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 3072, 0, 1280, 2816, 2, 0, 3, 0, 64},
    {TileClass::kSmall64, 12, 48, 2816, 5120, 2, 0, 3, 0, 64},
    {TileClass::kSmall64, 48, 96, 2816, 5120, 2, 0, 2, 0, 64},
    {TileClass::kSmall64, 96, 192, 2816, 5120, 2, 0, 3, 0, 64},
    {TileClass::kBig128, 384, 768, 2816, 5120, 2, 0, 3, 0, 64},
    {TileClass::kWide128x256, 768, 3072, 2816, 5120, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 3072, 0, 2816, 5120, 2, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 96, 5120, 8576, 2, 0, 3, 0, 64},
    {TileClass::kBig128, 192, 384, 5120, 8576, 2, 0, 3, 0, 64},
    {TileClass::kWide128x256, 384, 768, 5120, 8576, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 768, 3072, 5120, 8576, 2, 0, 2, 0, 64},
    {TileClass::kWide128x256, 3072, 0, 5120, 8576, 2, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 24, 8576, 19840, 2, 0, 3, 0, 64},
    {TileClass::kSmall64, 24, 96, 8576, 19840, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 96, 192, 8576, 19840, 2, 0, 2, 0, 64},
    {TileClass::kWide128x256, 192, 384, 8576, 19840, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 384, 768, 8576, 19840, 2, 0, 3, 0, 64},
    {TileClass::kBig128, 768, 3072, 8576, 19840, 2, 0, 2, 0, 64},
    {TileClass::kWide128x256, 3072, 0, 8576, 19840, 2, 0, 2, 0, 64},
    {TileClass::kBig128, 192, 384, 19840, 0, 2, 0, 3, 0, 64},
    {TileClass::kWide128x256, 384, 0, 19840, 0, 2, 0, 2, 0, 64},
}};
static constexpr std::array<TableRow, 41> kBuiltinPlanF8A8 = {{
    // The same narrow-N pathology, band and winner as above: the 128x128
    // kk64 row below measured 1.72-1.91x off at n=256, m 1792..2560 (A/B,
    // 2026-09-14, k=4096).
    {TileClass::kSmall64, 1536, 3072, 0, 256, 3, 0, 3, 0, 64},
    {TileClass::kSmall64, 12, 768, 0, 1280, 3, 0, 3, 0, 64},
    {TileClass::kBig128, 1536, 3072, 0, 1280, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 3072, 0, 0, 1280, 3, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 384, 1280, 2816, 3, 0, 3, 0, 64},
    {TileClass::kBig128, 768, 1536, 1280, 2816, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 1536, 3072, 1280, 2816, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 3072, 0, 1280, 2816, 3, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 192, 2816, 5120, 3, 0, 3, 0, 64},
    {TileClass::kBig128, 384, 768, 2816, 5120, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 768, 1536, 2816, 5120, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 1536, 3072, 2816, 5120, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 3072, 0, 2816, 5120, 3, 0, 3, 0, 64},
    {TileClass::kSmall64, 12, 24, 5120, 8576, 3, 0, 2, 0, 64},
    {TileClass::kSmall64, 24, 96, 5120, 8576, 3, 0, 3, 0, 64},
    {TileClass::kBig128, 192, 384, 5120, 8576, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 384, 768, 5120, 8576, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 768, 3072, 5120, 8576, 3, 0, 2, 0, 64},
    {TileClass::kWide128x256, 3072, 0, 5120, 8576, 3, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 24, 8576, 19840, 3, 0, 3, 0, 64},
    {TileClass::kSmall64, 24, 96, 8576, 19840, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 96, 192, 8576, 19840, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 192, 384, 8576, 19840, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 384, 0, 8576, 19840, 3, 0, 2, 0, 64},
    {TileClass::kSmall64, 12, 96, 19840, 0, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 192, 384, 19840, 0, 3, 0, 3, 0, 64},
    {TileClass::kWide128x256, 384, 768, 19840, 0, 3, 0, 2, 0, 64},
    {TileClass::kBig128, 768, 1536, 19840, 0, 3, 0, 2, 0, 64},
    {TileClass::kWide128x256, 1536, 0, 19840, 0, 3, 0, 2, 0, 64},
    // Crosswise rows, 2026-09-19 (block comment above): qkv. NN bands the
    // swapped aspect (canonicalize_gemm plans NN as 6144x16384).
    {TileClass::kBig128, 6143, 6144, 16383, 16384, 3, 1, 2, 0, 64, 1535, 1536},
    {TileClass::kBig128, 16383, 16384, 6143, 6144, 3, 1, 2, 0, 64, 1535, 1536},
    {TileClass::kWide128x256, 16383, 16384, 6143, 6144, 3, 2, 2, 0, 64, 1535, 1536},
    // square (k=1536) and mlp_down (k=6912) share the (16384,1536) aspect;
    // the exact k band splits them. kK=32 big CTA carries NN/TT (deeper 32KB
    // ring, same 2.4GB of L2 traffic at higher occupancy), TN keeps wide
    // kK=64.
    {TileClass::kBig128, 1535, 1536, 16383, 16384, 3, 1, 3, 0, 32, 1535, 1536},
    {TileClass::kBig128, 16383, 16384, 1535, 1536, 3, 1, 3, 0, 32, 1535, 1536},
    {TileClass::kWide128x256, 16383, 16384, 1535, 1536, 3, 2, 2, 0, 64, 1535, 1536},
    // mlp_up.
    {TileClass::kBig128, 6911, 6912, 16383, 16384, 3, 1, 2, 0, 64, 1535, 1536},
    {TileClass::kBig128, 16383, 16384, 6911, 6912, 3, 1, 2, 0, 64, 1535, 1536},
    {TileClass::kWide128x256, 16383, 16384, 6911, 6912, 3, 2, 2, 0, 64, 1535, 1536},
    // mlp_down (k=6912).
    {TileClass::kBig128, 1535, 1536, 16383, 16384, 3, 1, 3, 0, 32, 6911, 6912},
    {TileClass::kWide128x256, 16383, 16384, 1535, 1536, 3, 1, 2, 0, 64, 6911, 6912},
    {TileClass::kWide128x256, 16383, 16384, 1535, 1536, 3, 2, 2, 0, 64, 6911, 6912},
}};

// The device the rows above were measured on.
static constexpr DeviceFacts kBuiltinPlanMeasuredOn = {
    /*sms=*/170, /*smem_max=*/101376, /*smem_per_sm=*/102400,
    /*regs_per_sm=*/65536, /*l2_bytes=*/100663296, /*cc=*/120};

inline bool builtin_rows_match_device(const DeviceFacts& dev) {
    const DeviceFacts& m = kBuiltinPlanMeasuredOn;
    return dev.cc == m.cc && dev.sms == m.sms && dev.smem_max == m.smem_max &&
           dev.smem_per_sm == m.smem_per_sm && dev.regs_per_sm == m.regs_per_sm &&
           dev.l2_bytes == m.l2_bytes;
}

// END GENERATED


// Builtin table for one dtype class; count receives its row count. An empty
// table (the shipped default) returns a valid pointer and a zero count, so
// plan_row_for matches nothing and the chain falls through to the model.
inline constexpr const TableRow* builtin_plan_table(int perf_class, int& count) {
    switch (perf_class) {
        case 0:
            count = (int)kBuiltinPlanW16A16.size();
            return kBuiltinPlanW16A16.data();
        case 1:
            count = (int)kBuiltinPlanW8A16.size();
            return kBuiltinPlanW8A16.data();
        case 2:
            count = (int)kBuiltinPlanW8A8.size();
            return kBuiltinPlanW8A8.data();
        case 3:
            count = (int)kBuiltinPlanF8A8.size();
            return kBuiltinPlanF8A8.data();
        default:
            count = 0;
            return nullptr;
    }
}

// Last-resort rows for the chain's tail: the M band's dominant recipe from
// the full-coverage sweep (small for short M, narrow mid, big past mid) —
// a safe default, never best. Open N with -1 keys matches every shape, so
// planning stays a total function; the RowSetPlanner over them reads the
// M band alone (the m-only query below carries the matcher's "strictly
// past the min" caveat: an n of 0 sits ON the open bound, so a 1 stands
// for "some real n").
static constexpr TableRow kDegradedPlanRows[] = {
    {TileClass::kSmall64, 0, 512, 0, 0, -1, -1, 2, 0},
    {TileClass::kNarrow128x64, 512, 3072, 0, 0, -1, -1, 2, 0},
    {TileClass::kBig128, 3072, 0, 0, 0, -1, -1, 2, 0},
};

}  // namespace gemm
}  // namespace astrai
