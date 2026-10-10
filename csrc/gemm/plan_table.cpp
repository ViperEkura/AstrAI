/* GEMM row parsing, storage, built-in data and runtime configuration. */
#include "plan_table.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <api/gemm.h>
#include <policy.cuh>

namespace astrai {
namespace gemm {

/*
 * Field counts the parser reads: current, legacy 9 (no k), +k-band 12,
 * +wave-gate 13, +wave-permille 14.
 */
static constexpr int kRowFields = 10;
static constexpr int kRowFieldsLegacyK = 9;
static constexpr int kRowFieldsKband = 12;
static constexpr int kRowFieldsWave = 13;
static constexpr int kRowFieldsWavePermille = 14;

/*
 * Upper bound of the perf_class field, mirroring GemmPerfClass's last
 * enumerator (kF8A8, launcher/plan_types.h — where the enum lives).
 */
static constexpr int kMaxPerfClass = 3;

/*
 * The k-tile depths the manifests carry; any other depth matches no tile
 * and launches nothing, so it is rejected.
 */
inline constexpr bool row_k_supported(int k_tile) {
    return k_tile == 32 || k_tile == 64 || k_tile == 128;
}

/*
 * Pipeline depths a row may name. Enumerated, not a range, so a gap cannot pass
 * as "inside 2..5": s4/s5 stay parseable for old sweep files, but no ladder
 * instantiates past s3 (the s4/s5 twins were a measured wash, removed).
 */
inline constexpr bool row_stages_supported(int k_stages) {
    return k_stages == 2 || k_stages == 3 || k_stages == 4 || k_stages == 5;
}
/*
 * CTA geometry of one class, off kTileClassCta (self-asserted against the
 * tiles), so the numbers cannot drift from what the ladders instantiate.
 */
inline constexpr void plan_row_geometry(TileClass cta, int& bm, int& bn) {
    bm = kTileClassCta[(int)cta][0];
    bn = kTileClassCta[(int)cta][1];
}

/*
 * CTAs of one plan's tile per SM, or 0 when unpriced/unlaunchable. Minimum
 * of smem-per-SM over the ring and the __launch_bounds__ hint — the hint is
 * a FLOOR on the real count (the exact figure needs a driver query a pure
 * planner cannot make), so resident_model <= resident_true fires the wave
 * gate early, never late. Exact for the 512-thread tiles where the register
 * file binds (64 regs x 512 x 2 = 64K) — the only gated ring today.
 */
int plan_resident_ctas(TileClass cta, int k_stages, int k_tile, const PlanQuery& q) {
    const DeviceFacts& dev = q.dev;
    if (dev.smem_per_sm <= 0 || dev.regs_per_sm <= 0)
        return 0;
    int bm = 0, bn = 0;
    plan_row_geometry(cta, bm, bn);
    const int ring = ring_smem_bytes(bm, bn, k_tile, k_stages, q.ba, q.bb);
    const int smem = q.tma ? tma_smem_bytes(ring, k_stages) : ring;
    if (smem > dev.smem_max)
        return 0;
    return std::min(dev.smem_per_sm / smem, min_ctas_for_ring(ring));
}

/*
 * First matching row wins (generated tables are non-overlapping). q.k <= 0
 * is the open-K reading for no-depth callers; q.dev.sms <= 0 skips
 * gated rows rather than guessing. No match returns nullptr.
 */
inline const TableRow* plan_row_for(const TableRow* rows, int count, const PlanQuery& q) {
    for (int i = 0; i < count; ++i) {
        const TableRow& r = rows[i];
        if (q.m <= r.m_min)
            continue;
        if (r.m_max != 0 && q.m > r.m_max)
            continue;
        if (q.n <= r.n_min)
            continue;
        if (r.n_max != 0 && q.n > r.n_max)
            continue;
        if (r.perf_class != -1 && r.perf_class != q.perf_class)
            continue;
        if (r.crosswise != -1 && r.crosswise != q.crosswise)
            continue;
        /*
         * An open row (both bounds 0) matches any K, including the k <= 0
         * callers; a bounded row only matches a real depth.
         */
        if (r.k_min != 0 || r.k_max != 0) {
            if (q.k <= 0)
                continue;
            if (q.k <= r.k_min)
                continue;
            if (r.k_max != 0 && q.k > r.k_max)
                continue;
        }
        /*
         * Wave gates: the row's own geometry prices the grid, so a row cannot
         * state a fill it could not itself satisfy. min_ctas_per_sm is the
         * bare CTAs-per-SM form; min_wave_permille the wave form, whose
         * resident term is priced from the row's ring on this device.
         */
        if (r.min_ctas_per_sm > 0 || r.min_wave_permille > 0) {
            if (q.dev.sms <= 0)
                continue;
            int bm, bn;
            plan_row_geometry(r.cta, bm, bn);
            const int64_t grid = ((q.m + bm - 1) / bm) * ((q.n + bn - 1) / bn) * q.batch;
            if (r.min_ctas_per_sm > 0 && grid < (int64_t)r.min_ctas_per_sm * q.dev.sms)
                continue;
            if (r.min_wave_permille > 0) {
                const int resident = plan_resident_ctas(r.cta, r.k_stages, r.k_tile, q);
                if (resident <= 0)
                    continue;
                if (grid * 1000 < (int64_t)r.min_wave_permille * q.dev.sms * resident)
                    continue;
            }
        }
        return &r;
    }
    return nullptr;
}

/*
 * Row file format: one row per line, whitespace-separated
 *   m_min m_max n_min n_max perf_class crosswise cta k_stages raster
 *   [k_tile [k_min k_max [min_ctas_per_sm [min_wave_permille]]]]
 * '#' starts a comment; perf_class 0..3 or -1; crosswise 0..2 or -1; cta is the
 * TileClass ordinal (the policy/manifest.cuh enum order); k_stages 2..5 parse, but no
 * ladder instantiates a tile past s3 — plan_from_row rejects a deeper ring
 * outright, so a stale sweep row naming s4/s5 falls to the next source. The
 * trailing 'k_tile' defaults to kTableRowK, which is what the sweep
 * scripts leave off.
 * The trailing forms are additive — a row that omits them behaves exactly as
 * it did before they existed, which is what keeps hand-edited tuning files and
 * older sweeps valid. Invalid lines are warn-and-skip: a malformed row must
 * never block a launch the fallback would serve.
 */

// lo <= v <= hi, for the fields whose legal values are a contiguous interval.
inline constexpr bool in_range(int v, int lo, int hi) { return v >= lo && v <= hi; }

/*
 * A band is (min, max]: min exclusive, 0 = unbounded, so the sentinel stays
 * out of the ordering test. max == min is an empty band — legal, and simply
 * never matched.
 */
inline constexpr bool row_band_ok(int64_t min, int64_t max) { return max == 0 || max >= min; }

/*
 * First out-of-range field, or nullptr: one check per line so the warning
 * names the hand-edited column that is wrong.
 */
inline const char* plan_row_error(const TableRow& row, int fields) {
    if (fields != kRowFields && fields != kRowFieldsLegacyK && fields != kRowFieldsKband &&
        fields != kRowFieldsWave && fields != kRowFieldsWavePermille)
        return "field count";
    if (row.m_min < 0 || row.n_min < 0)
        return "band min < 0";
    if (!row_k_supported(row.k_tile))
        return "k (want 32, 64 or 128)";
    if (!row_band_ok(row.m_min, row.m_max))
        return "m band (max < min)";
    if (!row_band_ok(row.n_min, row.n_max))
        return "n band (max < min)";
    if (!row_band_ok(row.k_min, row.k_max))
        return "k band (max < min)";
    if (row.k_min < 0 || row.k_max < 0)
        return "k band min < 0";
    if (row.min_ctas_per_sm < 0)
        return "min_ctas_per_sm < 0";
    if (row.min_wave_permille < 0)
        return "min_wave_permille < 0";
    if (!in_range(row.perf_class, -1, kMaxPerfClass))
        return "perf_class";
    if (!in_range(row.crosswise, -1, 2))
        return "crosswise (-1..2)";
    if (!row_stages_supported(row.k_stages))
        return "k_stages (want 2..5)";
    return nullptr;
}

// Warn-and-skip: a hand-edit typo costs one row, not the table.
inline void warn_bad_row(const std::string& path, int lineno, const char* why) {
    std::fprintf(stderr, "[gemm-plan-table] %s:%d: ignoring row: bad %s\n", path.c_str(), lineno,
                 why);
}

/*
 * One row-file line (label names the source in warnings). Mutated in place
 * (the '#' comment cut); both the file and runtime-injection readers go
 * through here so the two cannot drift.
 */
inline void
parse_plan_table_line(char* line, const char* label, int lineno, std::vector<TableRow>& rows) {
    if (char* hash = std::strchr(line, '#'); hash != nullptr)
        *hash = '\0';
    long long m_min, m_max, n_min, n_max;
    int perf_class, crosswise, cta, k_stages, raster;
    /*
     * sscanf leaves a variable alone when its conversion fails, so a row
     * that omits the trailing fields keeps the defaults here: the legacy
     * and k-less field counts need no repair pass.
     */
    int k_tile = kTableRowK;
    long long k_min = 0, k_max = 0;
    int min_ctas_per_sm = 0;
    int min_wave_permille = 0;
    const int got =
        std::sscanf(line, " %lld %lld %lld %lld %d %d %d %d %d %d %lld %lld %d %d", &m_min, &m_max,
                    &n_min, &n_max, &perf_class, &crosswise, &cta, &k_stages, &raster, &k_tile,
                    &k_min, &k_max, &min_ctas_per_sm, &min_wave_permille);
    if (got == EOF)
        return; // blank or comment-only line
    if (got < kRowFieldsLegacyK) {
        warn_bad_row(label, lineno, "field count");
        return;
    }
    /*
     * cta is read as the TileClass ordinal, so it is the one field checked
     * before there is a row to validate; plan_row_error takes the rest.
     */
    if (!in_range(cta, 0, kTileClassCount - 1)) {
        warn_bad_row(label, lineno, "cta index");
        return;
    }
    const TableRow row{static_cast<TileClass>(cta),
                       m_min,
                       m_max,
                       n_min,
                       n_max,
                       perf_class,
                       crosswise,
                       k_stages,
                       raster,
                       k_tile,
                       k_min,
                       k_max,
                       min_ctas_per_sm,
                       min_wave_permille};
    if (const char* bad = plan_row_error(row, got); bad != nullptr) {
        warn_bad_row(label, lineno, bad);
        return;
    }
    rows.push_back(row);
}

inline bool parse_plan_table_file(const std::string& path, std::vector<TableRow>& rows) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr)
        return false;
    char line[256];
    int lineno = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        ++lineno;
        parse_plan_table_line(line, path.c_str(), lineno, rows);
    }
    std::fclose(f);
    return true;
}

/*
 * The same parser over in-memory row text: the runtime channel and the file
 * path accept identical syntax. Returns the surviving row count.
 */
inline int
parse_plan_table_text(const std::string& text, const char* label, std::vector<TableRow>& rows) {
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
        /*
         * The trailing newline of the final chunk loops once more with an
         * empty string; an empty line parses to nothing, so the extra pass
         * is harmless.
         */
    }
    return (int)rows.size() - before;
}

// Row tiers copy lookup results under a mutex; installs replace the whole table.
class RowSource {
  public:
    void set_from(std::string source, std::vector<TableRow> rows) {
        std::lock_guard<std::mutex> g(mutex_);
        rows_ = std::move(rows);
        source_ = std::move(source);
        has_rows_.store(!rows_.empty(), std::memory_order_release);
    }
    void clear() { set_from({}, {}); }
    std::string source() const {
        std::lock_guard<std::mutex> g(mutex_);
        return source_;
    }
    std::optional<TableRow> lookup(const PlanQuery& q) const {
        if (!has_rows_.load(std::memory_order_acquire))
            return std::nullopt;
        std::lock_guard<std::mutex> g(mutex_);
        const TableRow* row = plan_row_for(rows_.data(), (int)rows_.size(), q);
        if (row == nullptr)
            return std::nullopt;
        return *row;
    }
    size_t size() const {
        std::lock_guard<std::mutex> g(mutex_);
        return rows_.size();
    }

  private:
    mutable std::mutex mutex_;
    std::atomic<bool> has_rows_{false};
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

/*
 * The planner-rank vocabulary, one place: the strings configure() takes
 * and config_state() returns for GemmConfig::planner.
 */
inline constexpr const char* kPlannerModeNames[] = {"table", "hybrid", "model", "heuristic"};
inline constexpr int kPlannerModeCount = sizeof(kPlannerModeNames) / sizeof(kPlannerModeNames[0]);
bool parse_planner_mode(const std::string& name, int& out) {
    for (int i = 0; i < kPlannerModeCount; ++i)
        if (name == kPlannerModeNames[i]) {
            out = i;
            return true;
        }
    return false;
}

/*
 * Resolved views (unset falls to the default, never to a later env read);
 * the three launch-side knobs' resolved views are plan_types.h's.
 */
int gemm_planner_mode() {
    gemm_config_seed_once();
    const int v = gemm_config().planner.load(std::memory_order_relaxed);
    return v < 0 ? 2 : v; // default: legacy model
}
bool gemm_table_off() {
    gemm_config_seed_once();
    return gemm_config().table_off.load(std::memory_order_relaxed) > 0;
}

// Read legacy ASTR_GEMM_* defaults once; configure() can override them.
void gemm_config_seed_once() {
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
        if (env("ASTR_GEMM_NO_TMA") == "1")
            c.tma_disabled = 1;
        if (env("ASTR_GEMM_NO_MX") == "1")
            c.mx_disabled = 1;
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

namespace {

/*
 * Install one row tier from a spec (a row-file path when one opens, else
 * inline row text) and remember the spec, so the config state can hand back a
 * value that re-installs it. `label` is what a parse error reports.
 */
void install_rows(RowSource& tier, const char* label, const std::string& source) {
    std::vector<TableRow> rows;
    if (!parse_plan_table_file(source, rows))
        parse_plan_table_text(source, label, rows);
    tier.set_from(source, std::move(rows));
}

RowSource& row_tier(RowTier tier) {
    return tier == RowTier::Injected ? plan_table_injected_source() : plan_table_override_source();
}

} // namespace

/*
 * Runtime configuration: the backing of astrai.extension.policy.gemm.plan. Every knob is
 * tri-state — an absent patch field leaves it unchanged, an explicit value wins
 * over the one-time env seed. Rows are addressed by tier (`rows` + `tier`), the
 * all-tiers-off switch is its own field, and the staging keys are positive
 * enables: tma=false forces cp.async staging, mx=false knocks the sm_120a
 * block-scale cell out (the A/B knobs).
 */

GemmConfigState config_state() {
    GemmConfigState s;
    s.planner = kPlannerModeNames[gemm_planner_mode()]; // resolves unset
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
                         which == RowTier::Injected ? "injected rows" : "override rows",
                         *patch.rows);
        }
    }
    return config_state();
}

std::optional<TableRow> plan_override_row(const PlanQuery& q) {
    return plan_table_override_source().lookup(q);
}

std::optional<TableRow> plan_injected_row(const PlanQuery& q) {
    return plan_table_injected_source().lookup(q);
}

std::optional<TableRow> plan_builtin_row(const PlanQuery& q) {
    if (!builtin_rows_match_device(q.dev))
        return std::nullopt;
    int count = 0;
    const TableRow* rows = builtin_plan_table(q.perf_class, count);
    const TableRow* row = rows ? plan_row_for(rows, count, q) : nullptr;
    return row ? std::optional<TableRow>(*row) : std::nullopt;
}

} // namespace gemm
} // namespace astrai
