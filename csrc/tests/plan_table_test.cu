// Compile + behavior check for the per-class plan tables and the int8 class
// fix. Standalone, like the other csrc tests:
//   nvcc -std=c++17 -arch=sm_89 -I csrc/kernels -o /tmp/plan_classes plan_classes_check.cu
// Prints the dtype class each operand pair keys, and which builtin row the
// planner matches for a few representative shapes, then exits non-zero if any
// expectation fails.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "gemm/gemm.cuh"

using namespace astrai;
using namespace astrai::gemm;

static int failures = 0;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL: ");                           \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++failures;                                      \
        }                                                    \
    } while (0)

namespace {

// The device every structural probe is priced against — this container's
// RTX 4090, written down rather than queried so an expectation cannot change
// with the machine running the test. Named fields, not an aggregate: the
// point of the fixture is that a reader can see which number is which.
DeviceFacts test_dev(int sms = 128, int smem_per_sm = 102400) {
    DeviceFacts dev;
    dev.sms = sms;
    dev.smem_max = 101376;
    dev.smem_per_sm = smem_per_sm;
    dev.regs_per_sm = 65536;
    dev.l2_bytes = 75497472;
    dev.cc = 89;
    return dev;
}

// A query for a bare shape on the test device; the probes that care about
// operand widths or the device set those fields themselves.
PlanQuery shape_query(int64_t m, int64_t n, int64_t k,
                                int perf_class = 0,
                                const DeviceFacts& dev = test_dev()) {
    PlanQuery q;
    q.m = m;
    q.n = n;
    q.k = k;
    q.perf_class = perf_class;
    q.dev = dev;
    return q;
}

const char* class_name(GemmPerfClass c) {
    switch ((int)c) {
        case 0: return "W16A16";
        case 1: return "W8A16";
        case 2: return "W8A8";
        case 3: return "F8A8";
        default: return "?";
    }
}

const char* cta_name(TileClass c) {
    switch ((int)c) {
        case 0: return "small64";
        case 1: return "narrow128x64";
        case 2: return "big128";
        case 3: return "wide128x256";
        default: return "?";
    }
}

struct Probe {
    int64_t m, n, k;
    int perf_class;
    TileClass want_cta;
    int want_stages, want_kk;
    const char* note;
};

}  // namespace

int main() {
    // 1. Dtype classification: each pair keys its own class. The classes are
    //    named constants first — a template comma inside a macro argument
    //    would split it.
    constexpr GemmPerfClass kBF16BF16 = gemm_perf_class<__nv_bfloat16, __nv_bfloat16>();
    constexpr GemmPerfClass kBF16I8 = gemm_perf_class<__nv_bfloat16, int8_t>();
    constexpr GemmPerfClass kI8I8 = gemm_perf_class<int8_t, int8_t>();
    constexpr GemmPerfClass kFP8FP8 = gemm_perf_class<__nv_fp8_e4m3, __nv_fp8_e4m3>();
    constexpr GemmPerfClass kBF16FP8 = gemm_perf_class<__nv_bfloat16, __nv_fp8_e4m3>();
    CHECK(kBF16BF16 == GemmPerfClass::kW16A16, "bf16xbf16 -> %s", class_name(kBF16BF16));
    CHECK(kBF16I8 == GemmPerfClass::kW8A16, "bf16xint8 -> %s", class_name(kBF16I8));
    CHECK(kI8I8 == GemmPerfClass::kW8A8,
          "int8xint8 -> %s (the bug: it used to key F8A8)", class_name(kI8I8));
    CHECK(kFP8FP8 == GemmPerfClass::kF8A8, "fp8xfp8 -> %s", class_name(kFP8FP8));
    CHECK(kBF16FP8 == GemmPerfClass::kW8A16, "bf16xfp8 -> %s", class_name(kBF16FP8));

    // 2. Per-class tables: the class picks the table, and a row cannot leak
    //    across classes the way the single mixed table let it.
    const Probe probes[] = {
        // int8 large square: the wide CTA row the class fix unlocked.
        {4096, 4096, 4096, 2, TileClass::kWide128x256, 2, 64, "w8a8 wide CTA, grid fits"},
        // int8 small: the class's catch-all, not an fp8/wide row.
        {512, 512, 512, 2, TileClass::kSmall64, 3, 64, "w8a8 small"},
        // fp8 must not see the int8 wide rows.
        {4096, 4096, 4096, 3, TileClass::kSmall64, 3, 64, "f8a8 has no wide row"},
        // 2026-09-11 grid-sweep additions.
        {1024, 1024, 4096, 0, TileClass::kSmall64, 3, 32, "w16a16 kK=32, new row"},
        {4096, 1024, 4096, 0, TileClass::kSmall64, 3, 32, "w16a16 kK=32, new row"},
        // The v2 narrow-N rows survive the split.
        {2048, 512, 4096, 0, TileClass::kSmall64, 3, 32, "w16a16 v2 narrow-N kK=32"},
        // 2026-09-11 ring-residency rows: every wide band takes the 128x128
        // kK=32 s2 twin (48KB ring -> 2 CTAs/SM, 16 warps of 32x32).
        {100000, 100000, 4096, 0, TileClass::kBig128, 2, 32, "w16a16 wide band kK=32"},
        {2048, 6144, 1536, 0, TileClass::kBig128, 2, 32, "w16a16 wide band kK=32"},
        {2048, 4096, 4096, 0, TileClass::kBig128, 2, 32, "w16a16 N=4096 band kK=32"},
        {4096, 1536, 4096, 0, TileClass::kBig128, 2, 32, "w16a16 narrow-N, large M kK=32"},
        {512, 11008, 4096, 0, TileClass::kBig128, 2, 32, "w16a16 small-M wide band kK=32"},
        // The band opens at M 512 (exclusive), so the octave above it is this
        // row's: 768 and 1024 sat on the 64x64 rows until it did.
        {768, 11008, 4096, 0, TileClass::kBig128, 2, 32, "w16a16 M=768 band kK=32"},
        // The narrow-N k-tile-depth row: N in (768,1536] drops from the kK=64
        // ring (one resident CTA) to the kK=32 twin, below M 1536 only.
        {1024, 1536, 1536, 0, TileClass::kSmall64, 3, 32, "w16a16 narrow-N band kK=32"},
        {512, 1024, 1024, 0, TileClass::kSmall64, 3, 32, "w16a16 narrow-N band kK=32"},
        {128, 1536, 1536, 0, TileClass::kSmall64, 3, 64, "w16a16 M<=128 keeps kK=64"},
        {4096, 1536, 1536, 0, TileClass::kBig128, 2, 32, "w16a16 M>1536 keeps its row"},
        // The narrow-N floor sits at 29 M-tiles, not at the 256-slot mark:
        // 26/28 M-tiles (312/336 CTAs over 256 slots) still hand the shape to
        // the 64x64 row, and the big tile only takes over at 30.
        {3328, 1536, 1536, 0, TileClass::kSmall64, 3, 32, "w16a16 narrow-N M=3328 -> small"},
        {3584, 1536, 1536, 0, TileClass::kSmall64, 3, 32, "w16a16 narrow-N M=3584 -> small"},
        {3712, 1536, 1536, 0, TileClass::kSmall64, 3, 32, "w16a16 narrow-N M=3712 -> small"},
        {3840, 1536, 1536, 0, TileClass::kBig128, 2, 32, "w16a16 narrow-N M=3840 -> big"},
        // The two kK=64 floors below lose their kK=32 twins in one band each.
        // Above M 1024 the mid-N shapes already resolve to the kK=32 tile
        // through the grid-sweep rows, so the new band only has to cover
        // M in (384,1024].
        {1024, 3072, 1536, 0, TileClass::kSmall64, 3, 32, "w16a16 N=(1536,3072] band kK=32"},
        {768, 3072, 1536, 0, TileClass::kSmall64, 3, 32, "w16a16 N=(1536,3072] band kK=32"},
        {512, 768, 3072, 0, TileClass::kSmall64, 3, 32, "w16a16 N<=768 band kK=32"},
        {768, 512, 2048, 0, TileClass::kSmall64, 3, 32, "w16a16 N<=768 band kK=32"},
        {256, 768, 3072, 0, TileClass::kSmall64, 3, 64, "w16a16 M<=384 keeps kK=64"},
        {256, 3072, 1536, 0, TileClass::kSmall64, 3, 32, "w16a16 M in (128,384] kK=32"},
        {3072, 2048, 1536, 0, TileClass::kSmall64, 3, 32, "w16a16 M>1024 mid-N kK=32"},
        // Bands the additions must not shadow: kK=64 floor, and the two edges
        // of the wide bands (N stays on the row below, M and K too).
        {64, 64, 64, 0, TileClass::kSmall64, 3, 64, "w16a16 tiny stays kK=64"},
        // Small-M wide band: 64x64 tile below the measured 384/512 flip, kk=32
        // on the K=4096 side (the kk twins tie at M 64; kk=32 is 44% ahead at
        // M 128) and kk=64 above K 4096.
        {64, 4096, 4096, 0, TileClass::kSmall64, 3, 32, "w16a16 small-M wide band, K=4096"},
        {64, 28672, 8192, 0, TileClass::kSmall64, 3, 64, "w16a16 small-M wide band, large K"},
        {192, 4096, 4096, 0, TileClass::kSmall64, 3, 32, "w16a16 small-M wide band kK=32"},
        {256, 28672, 8192, 0, TileClass::kSmall64, 3, 32, "w16a16 mid-M wide band, large K"},
        {512, 4096, 4096, 0, TileClass::kBig128, 2, 32, "w16a16 small-M wide band kK=32"},
        {512, 100000, 4096, 0, TileClass::kBig128, 2, 32, "w16a16 small-M wide band kK=32"},
        {512, 6144, 1536, 0, TileClass::kSmall64, 3, 32, "w16a16 small-M short-K keeps its row"},
        {2048, 1536, 1536, 0, TileClass::kSmall64, 3, 32, "w16a16 mid-M narrow-N keeps its row"},
        // W8A16 row that is not class 0/2/3.
        {4096, 1024, 4096, 1, TileClass::kBig128, 2, 64, "w8a16 big-M/small-N"},
        // The K band on the mid-N w8a8 row: short K falls through to the
        // catch-all, past the band the wide tile takes over.
        {2048, 2048, 256, 2, TileClass::kSmall64, 3, 64, "w8a8 mid-N short K -> small"},
        {4096, 2048, 256, 2, TileClass::kSmall64, 3, 64, "w8a8 mid-N short K -> small"},
        {2048, 2048, 512, 2, TileClass::kSmall64, 3, 64, "w8a8 at band edge (K=512) -> small"},
        {2048, 2048, 1024, 2, TileClass::kWide128x256, 2, 64, "w8a8 mid-N K>512 -> wide"},
        {4096, 2048, 2048, 2, TileClass::kWide128x256, 2, 64, "w8a8 mid-N K>512 -> wide"},
    };
    for (const Probe& t : probes) {
        int count = 0;
        const TableRow* table = builtin_plan_table(t.perf_class, count);
        CHECK(table != nullptr, "class %d has no builtin table", t.perf_class);
        if (table == nullptr) continue;
        // A fixed device, not the running one: these probes pin the table's
        // STRUCTURE, and a shape's expected recipe must not change with the
        // machine the test runs on. The wave gates' device dependence has its
        // own section below, with explicit counts and resources.
        const TableRow* row =
            plan_row_for(table, count, shape_query(t.m, t.n, t.k, t.perf_class));
        if (row == nullptr) {
            std::printf("FAIL: %s: no row for %lldx%lldx%lld class %d\n", t.note,
                        (long long)t.m, (long long)t.n, (long long)t.k, t.perf_class);
            ++failures;
            continue;
        }
        CHECK(row->cta == t.want_cta && row->stages == t.want_stages && row->kk == t.want_kk,
              "%s: %lldx%lldx%lld class %d -> cta %s s%d k%d (want %s s%d k%d)", t.note,
              (long long)t.m, (long long)t.n, (long long)t.k, t.perf_class, cta_name(row->cta),
              row->stages, row->kk, cta_name(t.want_cta), t.want_stages, t.want_kk);
        std::printf("ok   %-34s %6lldx%-6lldx%-6lld class %-6s -> %-12s s%d k%d\n", t.note,
                    (long long)t.m, (long long)t.n, (long long)t.k,
                    class_name((GemmPerfClass)t.perf_class), cta_name(row->cta),
                    row->stages, row->kk);
    }

    // 2b. The wave gate: a row whose bound is wave arithmetic states it in
    //     CTAs per SM, so the same table crosses over at a different M on a
    //     part with a different SM count. Passing the count in is the only
    //     way to test a second device from one machine.
    {
        int count = 0;
        const TableRow* table = builtin_plan_table(0, count);
        struct WaveProbe {
            int64_t m, n, k;
            int sms;
            TileClass want_cta;
            int want_stages, want_kk;
            const char* note;
        };
        const WaveProbe waves[] = {
            // The N=1536 band is gated in WAVES now: this ring leaves two CTAs
            // resident, so a wave is 256 CTAs and the row asks 1360 permille =
            // 348. 4096x1536 (32 x 12 = 384) clears it on both counts; a
            // 170-SM part needs 1.36 * 170 * 2 = 463 and the same shape still
            // clears (384 < 463 would not — the count is what moves).
            {4096, 1536, 1536, 128, TileClass::kBig128, 2, 32, "wave permille fires at 128 sms"},
            // The same shape on a 170-SM part needs 1.36 * 170 * 2 = 463 CTAs
            // and its 384 no longer fill it, so the kK=32 row underneath
            // serves the shape. That is the device dependence the wave form
            // buys: one table, a different crossover per part.
            {4096, 1536, 1536, 170, TileClass::kSmall64, 3, 32, "wave permille holds off at 170 sms"},
            // 5120x1152 (40 x 9 = 360) is past the measured crossover; 30
            // M-tiles (3840x1152) is not, even though the old M literal let it
            // through — the N dependence the literal could not carry.
            {5120, 1152, 1536, 128, TileClass::kBig128, 2, 32, "wave permille fires at N=1152"},
            {3840, 1152, 1536, 128, TileClass::kSmall64, 3, 32, "wave permille holds off at 30 M-tiles"},
            // 4096x1152 (32 x 9 = 288 = 1.13 waves) is the shape the literal
            // sent to the big tile at a 23% loss.
            {4096, 1152, 1536, 128, TileClass::kSmall64, 3, 32, "wave permille holds off at 288 CTAs"},
            // The mid-N half of the band is NOT gated in waves (the measured
            // split left no rule) and keeps the literal plus the bare
            // CTAs-per-SM gate: 4096x2048 clears both (32 x 16 = 512 >= 256)
            // and keeps its big tile, while 3072x2048 sits under the literal
            // floor and stays on the kK=32 row — the conservative half of the
            // open 3072x2048 (+6% for the big tile) question.
            {4096, 2048, 1536, 128, TileClass::kBig128, 2, 32, "mid-N literal keeps the big tile"},
            {3072, 2048, 1536, 128, TileClass::kSmall64, 3, 32, "mid-N literal floor holds"},
            // No device facts: a gated row cannot be priced, so it is skipped
            // rather than assumed to hold (the lookup still lands on the
            // degraded rows, so planning stays total).
            {4096, 1536, 1536, 0, TileClass::kSmall64, 3, 32, "no sms skips the gated row"},
        };
        for (const WaveProbe& w : waves) {
            const TableRow* row = plan_row_for(
                table, count, shape_query(w.m, w.n, w.k, 0, test_dev(w.sms)));
            if (row == nullptr) {
                std::printf("FAIL: %s: no row for %lldx%lldx%lld sms=%d\n", w.note,
                            (long long)w.m, (long long)w.n, (long long)w.k, w.sms);
                ++failures;
                continue;
            }
            CHECK(row->cta == w.want_cta && row->stages == w.want_stages &&
                      row->kk == w.want_kk,
                  "%s: %lldx%lldx%lld sms=%d -> cta %s s%d k%d (want %s s%d k%d)",
                  w.note, (long long)w.m, (long long)w.n, (long long)w.k, w.sms,
                  cta_name(row->cta), row->stages, row->kk, cta_name(w.want_cta),
                  w.want_stages, w.want_kk);
            std::printf("ok   %-36s sms=%-4d %6lldx%-6lld -> %-12s s%d k%d\n", w.note,
                        w.sms, (long long)w.m, (long long)w.n, cta_name(row->cta),
                        row->stages, row->kk);
        }
    }

    // 3. Rows in a class table must be keyed for it (this is what the
    //    static_asserts in plan_table.h enforce at compile time).
    CHECK(builtin_rows_keyed_for<0>(kBuiltinPlanW16A16,
                                    (int)(sizeof(kBuiltinPlanW16A16) / sizeof(TableRow))),
          "W16A16 table keyed correctly");
    CHECK(builtin_rows_keyed_for<2>(kBuiltinPlanW8A8,
                                    (int)(sizeof(kBuiltinPlanW8A8) / sizeof(TableRow))),
          "W8A8 table keyed correctly");

    // 4. Row files: the 13-field form carries the K band and the wave gate,
    //    and the 12-, 10- and 9-field forms keep their open bands so existing
    //    sweeps keep parsing (the fields are additive, so an older file means
    //    exactly what it meant before the gate existed).
    const char* path = "/tmp/plan_classes_rowfile.txt";
    if (FILE* f = std::fopen(path, "w")) {
        std::fprintf(f, "# probe rows\n");
        std::fprintf(f, "1024 0 1024 0 2 0 3 2 0 64 512 0\n");  // k-banded wide
        std::fprintf(f, "0 0 0 0 0 0 0 3 0 32\n");              // 10-field
        std::fprintf(f, "0 0 0 0 0 0 0 3 0\n");                 // 9-field
        std::fprintf(f, "2048 0 1024 0 0 0 2 2 0 32 0 0 2\n");  // 13-field, gated
        std::fprintf(f, "2048 0 1024 0 0 0 2 2 0 32 0 0 0 1360\n");  // 14-field, wave gated
        std::fprintf(f, "0 0 0 0 99 0 0 3 0\n");                // bad perf_class
        std::fprintf(f, "0 0 0 0 0 0 0 3 0 400 0\n");           // bad kK
        std::fprintf(f, "0 0 0 0 0 0 0 3 0 64 0 0 -1\n");       // bad wave gate
        std::fprintf(f, "0 0 0 0 0 0 0 3 0 64 0 0 0 -1\n");     // bad wave permille
        std::fclose(f);
    }
    std::vector<TableRow> parsed;
    CHECK(parse_plan_table_file(path, parsed), "row file parses");
    CHECK((int)parsed.size() == 5, "row file: %d rows survived (want 5; the 4 bad ones skip)",
          (int)parsed.size());
    if ((int)parsed.size() == 5) {
        CHECK(parsed[0].k_min == 512 && parsed[0].k_max == 0 && parsed[0].kk == 64,
              "row file: K band parsed (k_min=%lld k_max=%lld)", (long long)parsed[0].k_min,
              (long long)parsed[0].k_max);
        CHECK(parsed[0].cta == TileClass::kWide128x256, "row file: cta ordinal 3 -> wide CTA");
        CHECK(parsed[0].min_ctas_per_sm == 0, "row file: absent wave gate reads as no gate");
        CHECK(parsed[1].k_min == 0 && parsed[1].k_max == 0 && parsed[1].kk == 32,
              "row file: 10-field form keeps an open K band");
        CHECK(parsed[2].k_min == 0 && parsed[2].k_max == 0 && parsed[2].kk == 64,
              "row file: 9-field form keeps kK=64 and an open K band");
        CHECK(parsed[2].min_ctas_per_sm == 0, "row file: 9-field form keeps no gate");
        CHECK(parsed[3].min_ctas_per_sm == 2, "row file: wave gate parsed (%d)",
              parsed[3].min_ctas_per_sm);
        auto hit_for = [&](const TableRow* row, int64_t k,
                           const DeviceFacts& dev) {
            return plan_row_for(row, 1, shape_query(2048, 2048, k, 2, dev));
        };
        CHECK(hit_for(&parsed[0], 256, test_dev()) == nullptr,
              "row file: a k-banded row must not match K=256");
        CHECK(hit_for(&parsed[0], 1024, test_dev()) != nullptr,
              "row file: it must match K=1024");
        // The 13-field row is index 3 and gates on sms: the same shape it
        // names is served by it on a big-enough grid and skipped otherwise.
        // Its band is M>2048, N>1024, so 4096x4096 is 32x32 = 1024 CTAs.
        const TableRow* gated = &parsed[3];
        auto gated_hit = [&](int sms) {
            return plan_row_for(gated, 1, shape_query(4096, 4096, 4096, 0,
                                                      test_dev(sms)));
        };
        CHECK(gated_hit(128) != nullptr,
              "row file: gated row fires when the grid covers 2 CTAs/SM");
        CHECK(gated_hit(600) == nullptr,
              "row file: the same shape is skipped when 2 CTAs/SM outruns its grid");
        CHECK(gated_hit(0) == nullptr,
              "row file: gated row is skipped without device facts");
        // The 14-field row adds the wave form, which prices residency too: the
        // same 4096x4096 shape is 1024 CTAs against 1360 permille of
        // 128 * 2 = 348 on this device (fires) and of 600 * 2 = 1632 on a
        // 600-SM one (skipped, where the CTAs-per-SM gate still fired).
        const TableRow* waved = &parsed[4];
        auto waved_hit = [&](int sms) {
            return plan_row_for(waved, 1, shape_query(4096, 4096, 4096, 0,
                                                      test_dev(sms)));
        };
        CHECK(waved_hit(128) != nullptr, "row file: wave row fires at 1.36 waves");
        CHECK(waved_hit(600) == nullptr,
              "row file: wave row is skipped when its grid is under 1.36 waves");
        CHECK(parsed[4].min_wave_permille == 1360,
              "row file: wave permille parsed (%d)", parsed[4].min_wave_permille);
        CHECK(parsed[0].min_wave_permille == 0,
              "row file: an absent wave permille reads as no gate");
        // Residency is a FLOOR: the register term is the launch-bounds hint,
        // not the compiled count. On a part with 4x the smem and 2048 threads
        // per SM the 512-thread gated ring still prices at 2 — the register
        // file binds there, and the model cannot see that the compiler might
        // have used fewer than the hint's 64. A 256-thread ring would price at
        // the hint (2) while Ada's smem really fits 3, which is why no row
        // gates on one yet.
        const PlanQuery rich = shape_query(0, 0, 0, 0, test_dev(128, 400 * 1024));
        CHECK(plan_resident_ctas(TileClass::kBig128, 2, 32, rich) == 2,
              "residency: the register file caps a 512-thread tile at 2 on any 64K part");
        const PlanQuery own = shape_query(0, 0, 0);
        CHECK(plan_resident_ctas(TileClass::kBig128, 2, 32, own) == 2,
              "residency: the gated ring is 2 on this device, which is its measured figure");
        CHECK(plan_resident_ctas(TileClass::kBig128, 2, 64, own) == 1,
              "residency: the 96KB kK=64 ring is 1");
        // A ring past the per-block opt-in ceiling has no plan at all.
        CHECK(plan_resident_ctas(TileClass::kWide128x256, 2, 64, own) == 0,
              "residency: a ring over smem_max prices as unpriceable");
    }

    // 5. Row text parses identically to a row file (the runtime-injection
    //    binding accepts inline text; same parser, so same rows).
    {
        const char* rows_txt =
            "# parity probe\n"
            "1024 0 1024 0 2 0 3 2 0 64 512 0\n"
            "0 0 0 0 0 0 0 3 0 32\n"
            "0 0 0 0 99 0 0 3 0\n"
            "0 0 0 0 0 0 0 3 0 64 0 0 0 1360\n";
        std::vector<TableRow> from_text;
        const int kept = parse_plan_table_text(rows_txt, "parity-text", from_text);
        CHECK(kept == 3, "row text: %d rows survived (want 3)", kept);
        if (kept == 3) {
            CHECK(from_text[0].k_min == 512 && from_text[0].kk == 64,
                  "row text: K band parsed (k_min=%lld)", (long long)from_text[0].k_min);
            CHECK(from_text[1].kk == 32, "row text: 10-field form keeps kK=32");
            CHECK(from_text[2].min_wave_permille == 1360,
                  "row text: wave permille parsed (%d)",
                  from_text[2].min_wave_permille);
        }
    }

    // 6. Runtime-injected rows and the tier ranking: env file > injected >
    //    builtin, "-" kills all three, and a clear returns the shape to the
    //    degraded bands. All through the same lookup plan_gemm uses.
    {
        // A crosswise query (TT staging): every builtin row carries crosswise
        // 0, so this shape has no builtin row to hit — the tier probe.
        PlanQuery q = shape_query(1024, 1024, 4096, 0);
        q.crosswise = 1;
        RowTier tier = RowTier::kDegraded;
        std::optional<TableRow> row = table_row(q, &tier);
        CHECK(!row.has_value() && tier == RowTier::kDegraded,
              "injected: crosswise shape starts degraded (tier %d)", (int)tier);

        std::vector<TableRow> injected;
        parse_plan_table_text("0 0 0 0 0 1 1 2 0 64", "inject", injected);
        CHECK((int)injected.size() == 1, "injected: row text installed 1 row");
        set_plan_table_injected_rows(injected);
        tier = RowTier::kDegraded;
        row = table_row(q, &tier);
        CHECK(row.has_value() && tier == RowTier::kInjected &&
                  row->cta == TileClass::kNarrow128x64,
              "injected: row serves the shape (tier %d)", (int)tier);

        // The env file outranks the injected rows.
        const char* envpath = "/tmp/plan_classes_envfile.txt";
        if (FILE* f = std::fopen(envpath, "w")) {
            std::fprintf(f, "0 0 0 0 0 1 0 3 0 64\n");  // small64 s3 k64
            std::fclose(f);
        }
        setenv("ASTR_GEMM_TABLE", envpath, 1);
        tier = RowTier::kInjected;
        row = table_row(q, &tier);
        CHECK(row.has_value() && tier == RowTier::kOverride &&
                  row->cta == TileClass::kSmall64,
              "injected: env file outranks the injected rows (tier %d)", (int)tier);

        // While the env owns the source, every call re-reads it (sweeps
        // toggle the env per launch): same query, served by the env row.
        const PlanDecision d_env = plan_gemm_sourced(q);
        CHECK(d_env.tier == RowTier::kOverride &&
                  d_env.plan.cta == TileClass::kSmall64,
              "sourced: env-served plan comes from the env file (tier %d)",
              (int)d_env.tier);

        setenv("ASTR_GEMM_TABLE", "-", 1);
        tier = RowTier::kBuiltin;
        row = table_row(q, &tier);
        CHECK(!row.has_value() && tier == RowTier::kDegraded,
              "injected: '-' disables env, injected and builtin alike");

        unsetenv("ASTR_GEMM_TABLE");
        tier = RowTier::kDegraded;
        row = table_row(q, &tier);
        CHECK(row.has_value() && tier == RowTier::kInjected,
              "injected: unset env restores the injected rows (tier %d)", (int)tier);

        // An injected-row swap serves the next decision from the new rows
        // (no decision state outlives the call that made it).
        const PlanDecision d1 = plan_gemm_sourced(q);
        CHECK(d1.plan.cta == TileClass::kNarrow128x64, "sourced: first decision narrow");
        injected.clear();  // parse_plan_table_text appends; the swap replaces
        parse_plan_table_text("0 0 0 0 0 1 0 3 0 64", "swap", injected);
        set_plan_table_injected_rows(injected);
        const PlanDecision d2 = plan_gemm_sourced(q);
        CHECK(d2.plan.cta == TileClass::kSmall64,
              "sourced: row swap serves the next decision from the new rows");
        const PlanDecision d3 = plan_gemm_sourced(q);
        CHECK(d3.plan.cta == d2.plan.cta && d3.tier == d2.tier,
              "sourced: repeat query is stable");

        clear_plan_table_injected_rows();
        tier = RowTier::kInjected;
        row = table_row(q, &tier);
        CHECK(!row.has_value() && tier == RowTier::kDegraded,
              "injected: clear returns the shape to the degraded bands");
    }

    // 7. The planner probe: same decision the dispatch branch would make,
    //    tag selection mirrored branch-for-branch (NT crosswise 0 hits the
    //    builtin rows; TT is crosswise 1 and degraded; the symmetric NN
    //    rewrite swaps M and N exactly like gemm_dispatch).
    {
        const DeviceFacts dev = test_dev();
        const PlanProbe nt = plan_probe_for<__nv_bfloat16, __nv_bfloat16>(
            1024, 1024, 4096, 1, false, true, dev);
        CHECK(nt.perf_class == 0 && nt.crosswise == 0 &&
                  nt.tier == RowTier::kBuiltin && nt.plan.cta == TileClass::kSmall64 &&
                  nt.plan.stages == 3 && nt.plan.kk == 32,
              "probe: NT bf16xbf16 -> builtin small64 s3 k32 (tier %d cw %d)",
              (int)nt.tier, nt.crosswise);

        const PlanProbe tt = plan_probe_for<__nv_bfloat16, __nv_bfloat16>(
            1024, 1024, 4096, 1, true, true, dev);
        CHECK(tt.crosswise == 1 && tt.tier == RowTier::kDegraded &&
                  tt.plan.cta == TileClass::kNarrow128x64,
              "probe: TT has no builtin row -> degraded narrow (tier %d cw %d "
              "cta %d s%d k%d)",
              (int)tt.tier, tt.crosswise, (int)tt.plan.cta, tt.plan.stages,
              tt.plan.kk);

        // The n=0 regression: the m-only degraded query used to sit ON the
        // open n bound and skip every band, so every m took the small-CTA
        // fallback row and the narrow/big degraded bands were dead.
        CHECK(degraded_row_for(100).cta == TileClass::kSmall64 &&
                  degraded_row_for(1024).cta == TileClass::kNarrow128x64 &&
                  degraded_row_for(100000).cta == TileClass::kBig128,
              "probe: degraded bands pick by m (the n=0 skip regression)");

        const PlanProbe mixed = plan_probe_for<__nv_bfloat16, int8_t>(
            4096, 1024, 4096, 1, false, true, dev);
        CHECK(mixed.perf_class == 1 && mixed.crosswise == 0 &&
                  mixed.tier == RowTier::kBuiltin && mixed.plan.cta == TileClass::kBig128,
              "probe: NT bf16xint8 -> W8A16 builtin big128 (tier %d)", (int)mixed.tier);

        const PlanProbe nn = plan_probe_for<__nv_bfloat16, __nv_bfloat16>(
            2048, 512, 1024, 1, false, false, dev);
        const PlanProbe swapped_tt = plan_probe_for<__nv_bfloat16, __nv_bfloat16>(
            512, 2048, 1024, 1, true, true, dev);
        CHECK(nn.plan.cta == swapped_tt.plan.cta &&
                  nn.plan.stages == swapped_tt.plan.stages &&
                  nn.plan.kk == swapped_tt.plan.kk && nn.crosswise == swapped_tt.crosswise,
              "probe: symmetric NN rewrites to the transposed TT problem");
    }

    // 8. The recipe vocabulary: the (class, stages, kK) keys each ladder
    //    instantiates, deduped — the Python autotuner's candidate space.
    {
        const std::vector<GemmRecipe> two_byte = gemm_recipes_for(false, 2, 2);
        // Cross six (all kK 64) + the kK=32 twins the two-byte ladder adds
        // (small64, big128 at s2/s3, narrow s2), with the 16-warp twins
        // deduped onto their 32-warp keys.
        CHECK((int)two_byte.size() == 11, "vocab: two-byte ladder -> %d recipes (want 11)",
              (int)two_byte.size());
        auto has = [](const std::vector<GemmRecipe>& v, int cta, int s, int kk) {
            for (const GemmRecipe& r : v)
                if (r.cta == cta && r.stages == s && r.kk == kk) return true;
            return false;
        };
        CHECK(has(two_byte, (int)TileClass::kSmall64, 2, 32) &&
                  has(two_byte, (int)TileClass::kBig128, 3, 32),
              "vocab: two-byte ladder carries the kK=32 twins");
        CHECK(!has(two_byte, (int)TileClass::kWide128x256, 2, 64),
              "vocab: two-byte ladder has no wide CTA");

        const std::vector<GemmRecipe> byte = gemm_recipes_for(false, 1, 1);
        CHECK(has(byte, (int)TileClass::kWide128x256, 2, 64) &&
                  (int)byte.size() == 7,
              "vocab: byte ladder adds the wide CTA (%d recipes)", (int)byte.size());
        CHECK(!has(byte, (int)TileClass::kSmall64, 2, 32),
              "vocab: byte ladder has no kK=32 tile");

        const std::vector<GemmRecipe> cross = gemm_recipes_for(true, 2, 2);
        CHECK((int)cross.size() == 6, "vocab: crosswise ladder is the shared six (%d)",
              (int)cross.size());
        for (const GemmRecipe& r : cross)
            CHECK(r.kk == 64, "vocab: crosswise ladder is all kK=64 (got %d)", r.kk);

        // The mixed pair (2,1) runs the conservative ladder even congruent
        // (manifest_kind's fallback), so W8A16 candidates match the cross six.
        const std::vector<GemmRecipe> mixed = gemm_recipes_for(false, 2, 1);
        CHECK((int)mixed.size() == 6, "vocab: mixed widths fall back to the shared six (%d)",
              (int)mixed.size());
    }

    // 9. Planner cost (informational, one process): the row scan + wave
    //    pricing every launch runs. A 2026-09-13 exact-shape memo measured
    //    45 ns here against this ~110 ns plain path and was retired — the
    //    ~64 ns is invisible under the pybind marshalling around a launch.
    {
        const int kProbesSet = 256;
        const int kReps = 2000;
        PlanQuery qs[kProbesSet];
        for (int i = 0; i < kProbesSet; ++i) {
            qs[i] = shape_query(64 + 8 * i, 1024 + 16 * i, 768 + 32 * i,
                                i % 2 /* classes 0/1 */);
            qs[i].batch = 1;
            plan_gemm_sourced(qs[i]);
        }
        const auto t0 = std::chrono::steady_clock::now();
        volatile long long sink = 0;
        for (int r = 0; r < kReps; ++r)
            for (int i = 0; i < kProbesSet; ++i)
                sink += plan_gemm_sourced(qs[i]).plan.stages;
        const auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() /
                          (double)(kReps * kProbesSet);
        std::printf("planner bench: %.0f ns/query\n", ns);
    }

    std::printf(failures == 0 ? "\nall checks passed\n" : "\n%d FAILURES\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
