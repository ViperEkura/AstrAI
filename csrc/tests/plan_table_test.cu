// Compile + behavior check for the per-class plan tables and the int8 class
// fix. Standalone, like the other csrc tests:
//   nvcc -std=c++17 -arch=sm_89 -I csrc/kernels -o /tmp/plan_classes plan_classes_check.cu
// Prints the dtype class each operand pair keys, and which builtin row the
// planner matches for a few representative shapes, then exits non-zero if any
// expectation fails.
#include <cuda_runtime.h>

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
        {2048, 4096, 4096, 0, TileClass::kSmall64, 3, 32, "w16a16 kK=32, new row"},
        {4096, 1024, 4096, 0, TileClass::kSmall64, 3, 32, "w16a16 kK=32, new row"},
        // The v2 narrow-N rows survive the split.
        {2048, 512, 4096, 0, TileClass::kSmall64, 3, 32, "w16a16 v2 narrow-N kK=32"},
        // Bands the additions must not shadow: kK=64 floor.
        {64, 64, 64, 0, TileClass::kSmall64, 3, 64, "w16a16 tiny stays kK=64"},
        {100000, 100000, 4096, 0, TileClass::kBig128, 2, 64, "w16a16 big stays kK=64"},
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
        const TableRow* row =
            plan_row_for(table, count, t.m, t.n, t.perf_class, /*crosswise=*/0, t.k);
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

    // 3. Rows in a class table must be keyed for it (this is what the
    //    static_asserts in plan_table.h enforce at compile time).
    CHECK(builtin_rows_keyed_for<0>(kBuiltinPlanW16A16,
                                    (int)(sizeof(kBuiltinPlanW16A16) / sizeof(TableRow))),
          "W16A16 table keyed correctly");
    CHECK(builtin_rows_keyed_for<2>(kBuiltinPlanW8A8,
                                    (int)(sizeof(kBuiltinPlanW8A8) / sizeof(TableRow))),
          "W8A8 table keyed correctly");

    // 4. Row files: the 12-field form carries the K band, and the 10- and
    //    9-field forms keep the open band so existing sweeps keep parsing.
    const char* path = "/tmp/plan_classes_rowfile.txt";
    if (FILE* f = std::fopen(path, "w")) {
        std::fprintf(f, "# probe rows\n");
        std::fprintf(f, "1024 0 1024 0 2 0 3 2 0 64 512 0\n");  // k-banded wide
        std::fprintf(f, "0 0 0 0 0 0 0 3 0 32\n");              // 10-field
        std::fprintf(f, "0 0 0 0 0 0 0 3 0\n");                 // 9-field
        std::fprintf(f, "0 0 0 0 99 0 0 3 0\n");                // bad perf_class
        std::fprintf(f, "0 0 0 0 0 0 0 3 0 400 0\n");           // bad kK
        std::fclose(f);
    }
    std::vector<TableRow> parsed;
    CHECK(parse_plan_table_file(path, parsed), "row file parses");
    CHECK((int)parsed.size() == 3, "row file: %d rows survived (want 3; the 2 bad ones skip)",
          (int)parsed.size());
    if ((int)parsed.size() == 3) {
        CHECK(parsed[0].k_min == 512 && parsed[0].k_max == 0 && parsed[0].kk == 64,
              "row file: K band parsed (k_min=%lld k_max=%lld)", (long long)parsed[0].k_min,
              (long long)parsed[0].k_max);
        CHECK(parsed[0].cta == TileClass::kWide128x256, "row file: cta ordinal 3 -> wide CTA");
        CHECK(parsed[1].k_min == 0 && parsed[1].k_max == 0 && parsed[1].kk == 32,
              "row file: 10-field form keeps an open K band");
        CHECK(parsed[2].k_min == 0 && parsed[2].k_max == 0 && parsed[2].kk == 64,
              "row file: 9-field form keeps kK=64 and an open K band");
        const TableRow* hit = plan_row_for(parsed.data(), 1, 2048, 2048, 2, 0, 256);
        CHECK(hit == nullptr, "row file: a k-banded row must not match K=256");
        hit = plan_row_for(parsed.data(), 1, 2048, 2048, 2, 0, 1024);
        CHECK(hit != nullptr, "row file: it must match K=1024");
    }

    std::printf(failures == 0 ? "\nall checks passed\n" : "\n%d FAILURES\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
