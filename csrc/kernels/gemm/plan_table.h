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
// and their field order: parse_plan_table_file below.

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

// The ring depths a row file may NAME (the stages twin of row_k_supported,
// enumerated rather than written as a range so that a gap in the set cannot
// be admitted as "still inside 2..5"): s4/s5 stay parseable so old sweep
// files keep meaning what they meant, but no ladder instantiates a tile
// past s3 (the s4/s5 twins were a measured wash and were removed) —
// plan_from_row rejects those rows, and the degraded bands serve the shape.
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
    int out_elem_bytes = 2;  // output element bytes the model cost's output
                             // term prices; 2 = the bf16 fused-linear
                             // default (plan_query's OutT parameter — an
                             // fp32-out caller is priced at 4, not 2)
    bool tma = true;  // the staging this launch will take (plan_query fills
                      // it from launch_plan_impl's predicate): the planner
                      // prices residency per variant — the sign flips with
                      // staging (TMA shares bandwidth, cp.async's software
                      // ring IS the latency hiding)
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

// One line of a row file (the label names the source in warnings: a path or
// an in-memory text marker). The line buffer is mutated in place — the '#'
// comment cut — and parses into `rows` or is warn-skipped, exactly as the
// file loop always did; both the file and the runtime-injection readers go
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

// The same parser over an in-memory row text (newline-separated), so the
// runtime rows channel and the file path accept identical row syntax.
// Returns the number of rows that survived.
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

// Which source served a lookup — the probe binding reports it so the Python
// autotuner can top up only the shapes nothing else covers (the env file is
// the experimenter's, the builtin table is the AOT baseline; neither is the
// autotuner's to shadow, which is also why injected rows rank below the env
// file at lookup). kModel names the analytical planner (gemm.cuh's
// plan_model_scan): it is not a row source, but the probe's question —
// "who served this shape" — has the same answer shape for it.
// One row tier's backing store: a mutex-guarded row container. The
// container is replaced wholesale (set/clear), never mutated in place, and
// lookups copy the row out under the mutex — a concurrent install therefore
// cannot dangle a pointer a launched plan still holds (installs happen
// during serving warmups). Two mutable instances exist: the override source
// (the configure rows channel, which outranks everything) and the injected
// source (the same channel with tier=injected, where the autotuner's measured
// winners land); the builtin and degraded tiers read static rows and need no
// container.
//
// `source` is the spec the rows were last installed from (a row-file path or
// inline row text) — kept so the config state can hand back a value that
// re-installs them exactly. TableRow carries gates (k bands, occupancy
// floors) the text format cannot express, so re-emitting parsed rows would
// be lossy.
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


// Runtime configuration: the backing state of the runtime plan API
// (astrai.extension.ops.gemm's set_* functions and the gemm ``configure``
// binding). One knob per launch-time switch, each a tri-state atomic —
// -1 means "unset, the one-time env seed decides", any other value is
// explicit and wins. The environment is consulted exactly once per
// process (a migration seed; the vars are documented as deprecated),
// never per call: sweeps toggle this state through the binding instead,
// which is the same one-set-per-launch cadence the env file supported
// without paying getenv on the hot path.
struct GemmConfig {
    std::atomic<int> planner{-1};      // 0 table-only, 1 hybrid (table -> model), 2 model-only
    std::atomic<int> log{-1};          // [gemm-plan] stderr log on/off
    std::atomic<int> tma_disabled{-1};  // cp.async staging forced everywhere
    std::atomic<int> mx_disabled{-1};   // sm_120a block-scale cell knocked out
    std::atomic<int> table_off{-1};    // 1 = "-" (no override, no injected, no builtin rows)
};

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

inline GemmConfig& gemm_config() {
    static GemmConfig cfg;
    return cfg;
}

// The migration seed: legacy ASTR_GEMM_* variables read once, on the first
// planner/table touch. Explicit configure() calls bypass it entirely — they
// write the atomics directly.
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

// Resolved views (unset falls to the default, never to a later env read).
inline int gemm_planner_mode() {
    gemm_config_seed_once();
    const int v = gemm_config().planner.load(std::memory_order_relaxed);
    return v < 0 ? 1 : v;  // default: hybrid (model fills what no row owns)
}
inline bool gemm_plan_log_enabled() {
    gemm_config_seed_once();
    return gemm_config().log.load(std::memory_order_relaxed) > 0;
}
inline bool gemm_tma_staging_disabled() {
    gemm_config_seed_once();
    return gemm_config().tma_disabled.load(std::memory_order_relaxed) > 0;
}
inline bool gemm_mx_cell_disabled() {
    gemm_config_seed_once();
    return gemm_config().mx_disabled.load(std::memory_order_relaxed) > 0;
}
inline bool gemm_table_off() {
    gemm_config_seed_once();
    return gemm_config().table_off.load(std::memory_order_relaxed) > 0;
}

// BEGIN GENERATED
// Compiled-in rows are the measured DIFF of the analytical model, not a
// full-coverage table: a row is emitted only where its recipe beat the
// model's own dispatch by >=2% in an INTERLEAVED A/B re-measure (four
// alternating trials per point) — the sweep itself always measures the
// model reference last at each point, i.e. at the hottest clock state,
// which underprices it by 10-20% on heavy shapes and made batch-ordered
// diff tables overclaim (measured 2026-09-14; the same bias explains
// phase-ordered validate swings on the llama-sized holdouts). Ties and
// unmeasured bands therefore serve the model tier, and the rows below
// are exactly the model's confirmed errors, measured 2026-09-14 on this
// box's sm_120 / RTX 5090 / 170 SMs over M {1..4096} x N {1024..28672}
// x K {1536,4096,8192}, gated by a production-semantics holdout at 2%.
//
// The crosswise F8A8 rows below are the 2026-09-19 addition: the wide CTA
// joined the byte ladders for crosswise staging (manifest_kind routes
// 1-byte pairs to TileManifestByte regardless of staging), which moved the
// L2 re-read wall — the 64x64 model pick streams 4.8GB of operands through
// L2 on the qkv cell (84.7% L2 SOL, 48% compute) where a 128-row CTA
// streams 2.4GB and the wide 1.8GB. Measured at the production micro-batch
// m=16384 over the astrai_1b projections, four alternating A/B rounds per
// point (shipped model dispatch vs the pinned recipe). NN rows band the
// CANONICALIZED aspect — canonicalize_gemm plans symmetric NN as the
// transposed problem — and the (16384,1536) aspect is split by an exact k
// band into square (k=1536) and mlp_down (k=6912). Bands are the measured
// points only: another m or projection needs a sweep of its own. The same
// session's kK=32 unlock (byte ladder gains the 32-deep-ring big CTA and
// the kK=32 narrow) replaced three of these rows after a four-round confirm:
// square NN/TT and mlp_down NN run the kK=32 big CTA (the 18KB-ring kK=32
// narrow lost everywhere, -8..-34%, its packed grid half-busy); the kK=64
// rows keep the other nine points.
//
// A row is calibrated to the part it was measured on, and a stale one is
// worse than no row at all (the 2026-09-14 +33% lesson). The tier is
// therefore signature-guarded: kBuiltinPlanMeasuredOn below must equal
// the running device's facts or the builtin tier serves nothing and the
// chain runs override -> injected (autotuner cache, device-keyed) ->
// [no builtin] -> model -> degraded. Another part gets its rows from a
// sweep of its own, never from these.

static constexpr std::array<TableRow, 29> kBuiltinPlanW16A16 = {{
    // Prepended so its band wins first over the 128x128 kk32 row below,
    // which is 3.0-3.2x off here: at n <= 256 that tile has 40 blocks over
    // 170 SMs (a quarter-full machine), where this one fills 160. Measured
    // in an interleaved A/B (2026-09-14, this box, n=256 x k {2048,4096} x
    // m {1792..2560}); the band is the measured one, so n > 256 keeps the
    // older row until a sweep of its own says otherwise.
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
    // Same narrow-N pathology as the W16A16 row above it, same fix and same
    // band: the 128x128 kk64 row below is 2.9-3.1x off at n=256, m 1792..2560
    // (interleaved A/B, 2026-09-14, k=4096), where the small kk=64 cell is
    // the winner. The band is the measured one; n > 256 is unmeasured here.
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
    // The narrow-N pathology of the W16A16/W8A16 rows above, same band and
    // same winner: the 128x128 kk64 row below measured 1.72-1.91x off at n=256,
    // m 1792..2560 (interleaved A/B, 2026-09-14, k=4096).
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
    // Crosswise rows, 2026-09-19 (see the block comment): qkv. NN's row
    // bands the swapped aspect (canonicalize_gemm plans NN as 6144x16384).
    {TileClass::kBig128, 6143, 6144, 16383, 16384, 3, 1, 2, 0, 64, 1535, 1536},
    {TileClass::kBig128, 16383, 16384, 6143, 6144, 3, 1, 2, 0, 64, 1535, 1536},
    {TileClass::kWide128x256, 16383, 16384, 6143, 6144, 3, 2, 2, 0, 64, 1535, 1536},
    // square (k=1536) and mlp_down (k=6912) share the (16384,1536) aspect;
    // the exact k band splits them. The kK=32 big CTA carries NN/TT here
    // (its deeper 32KB ring prices the same 2.4GB of L2 traffic at higher
    // occupancy than the kK=64 twins), TN keeps the wide CTA at kK=64.
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

// The device the rows above were measured on; see the block comment.
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
