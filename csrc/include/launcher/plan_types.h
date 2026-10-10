#pragma once
/* GEMM runtime config, planner query, and dispatch decision. */
#include <atomic>
#include <cstdint>

#include <utils/device.cuh>

namespace astrai {
namespace gemm {

struct GemmConfig {
    std::atomic<int> planner{-1};      // 0 table, 1 hybrid, 2 fitted model, 3 geometry heuristic
    std::atomic<int> log{-1};          // [gemm-plan] stderr log on/off
    std::atomic<int> tma_disabled{-1}; // cp.async staging forced everywhere
    std::atomic<int> mx_disabled{-1};  // sm_120a block-scale cell knocked out
    std::atomic<int> table_off{-1};    // 1 = "-" (no override, no injected, no builtin rows)
};

inline GemmConfig& gemm_config() {
    static GemmConfig cfg;
    return cfg;
}

void gemm_config_seed_once(); // planning.cpp: environment defaults

// Dtype-class ids the plan-table rows key on.
enum class GemmPerfClass : int { kW16A16 = 0, kW8A16, kW8A8, kF8A8 };

struct GemmRecipe {
    int cta;      // TileClass ordinal — the row-file serialization key
    int k_stages; // prefetched K tiles; ring has one extra slot
    int k_tile;   // K elements held in one pipeline stage
    int bm, bn;   // CTA geometry
    int wm, wn;   // warp tiling (the recipe name's W<x>x<y>)
    int threads;  // the manifest entry's warp tiling (first match wins)
    int smem;     // ring bytes at this staging pair's operand widths
};

struct KernelResources {
    GemmRecipe effective{}; // after warp widening and output-reclaim substitution
    int resident = 0;       // CUDA occupancy limit, not observed execution occupancy
    int registers = 0;
    int local_bytes = 0;
};

struct PlanQuery {
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    int64_t batch = 1;
    int perf_class = -1; // GemmPerfClass id; -1 matches any
    int crosswise = 0;   // direct-load operand count, see gemm_dispatch
    int ba = 2;          // operand element bytes
    int bb = 2;
    int out_elem_bytes = 2; // bf16 output by default
    int mma_k = 16;         // promoted MMA instruction K extent
    bool tma = true;        // effective staging selected for this launch
    DeviceFacts dev{};
    bool rank3a = false, rank3b = false;
    bool contiguous = false; // congruous rows and sector-aligned base pointers
    KernelResources (*resources)(const GemmRecipe&, const PlanQuery&) = nullptr;
};

struct PlanDecision {
    GemmRecipe recipe;
    int raster;
    const char* source;
};

struct LaunchPlan {
    PlanDecision decision;
    bool tma;
};

int plan_raster(const PlanQuery& q, int bm, int bn);
PlanDecision plan_dispatch(const PlanQuery& q);
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

} // namespace gemm
} // namespace astrai
