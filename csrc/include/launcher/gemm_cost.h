#pragma once
/* Shared geometry cost for GEMM and symmetric matrix operations. */
#include <launcher/plan_types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace astrai {
namespace gemm {

// Raster the longer grid dimension; keep M-side operand tiles inside the L2 budget.
inline int geometry_raster(const PlanQuery& q, int bm, int bn) {
    const int64_t m_tiles = (q.m + bm - 1) / bm;
    const int64_t n_tiles = (q.n + bn - 1) / bn;
    if (m_tiles < n_tiles)
        return -8;
    if (q.n * q.k * (int64_t)q.bb <= q.dev.l2_bytes * 7 / 10)
        return 1;
    const double reserve = 0.12 + 0.28 * (double)q.bb / (double)q.ba;
    const double budget = (1.0 - std::min(reserve, 0.5)) * (double)q.dev.l2_bytes;
    const int64_t ub = (int64_t)(budget / ((double)bm * (double)q.k * q.ba));
    const int64_t lb = (q.dev.sms + n_tiles - 1) / n_tiles;
    int64_t g = std::min(ub, m_tiles);
    if (ub >= lb)
        g = std::min(std::max(g, lb), m_tiles);
    return (int)std::max(g, (int64_t)1);
}

// Five work proxies ranked in log space, not an execution-time estimate.
// Callers supply the actual grid and epilogue traffic (e.g. triangular SYRK).
inline double geometry_cost(const GemmRecipe& r,
                            const PlanQuery& q,
                            int resident_ctas,
                            double blocks,
                            double output_bytes) {
    if (resident_ctas <= 0 || q.dev.sms <= 0 || q.k <= 0 || blocks <= 0 || output_bytes <= 0)
        return std::numeric_limits<double>::infinity();
    // Integer ceil without n+d-1 overflow; cast before multiplying grid axes.
    auto ceil_div = [](std::int64_t n, int d) { return n / d + (n % d != 0); };
    const double steps = (double)ceil_div(q.k, r.k_tile);
    const double resident = std::min((double)resident_ctas, std::ceil(blocks / q.dev.sms));
    const double waves = std::ceil(blocks / (q.dev.sms * resident));
    const double warps = r.threads / 32.0;
    const double copy = (double)r.k_tile * (r.bm * q.ba + r.bn * q.bb) / 512.0;
    const double mma = ((double)r.k_tile / q.mma_k) * (r.bm / 16.0) * (r.bn / 8.0) / warps;
    const double fragments = (double)r.k_tile * (r.wm * q.ba + r.wn * q.bb) / 512.0;
    const double conversions =
        (double)r.k_tile * (r.wm * (q.ba < q.bb) + r.wn * (q.bb < q.ba)) / 64.0;
    const double output = output_bytes / 512.0;
    const double issue = q.tma ? 2.0 : copy;

    const double lw = std::log(waves), lr = std::log(resident);
    const double ll = std::log(steps), lp = std::log(warps);
    // log(L*copy + output), without constructing L*copy. The exponential
    // argument is nonpositive; underflow only discards a negligible addend.
    const double a = ll + std::log(copy), b = std::log(output);
    const double log_memory = std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
    const std::array<double, 5> arms = {
        lw + (q.tma ? lr : 0.0) + log_memory,
        lw + lr + ll + lp + std::log(mma),
        lw + ll + std::log(mma + fragments + conversions),
        lw + lr + ll + lp + std::log(fragments + conversions),
        lw + lr + ll + std::log(2.0 * warps + issue),
    };
    // Per-arm normalization is common to all candidates, and the fifth root
    // is monotone: neither changes ranking. Never multiply arms or exp(score).
    double score = 0.0;
    for (double arm : arms)
        score += arm;
    return score;
}

} // namespace gemm
} // namespace astrai
