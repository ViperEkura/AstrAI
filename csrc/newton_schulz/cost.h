#pragma once
/* Wave-weighted work for the compiled Newton-Schulz matrix recipes. */

#include <launcher/gemm_cost.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace astrai::newton_schulz {

struct CostTerms {
    double geometry = std::numeric_limits<double>::infinity();
    double model = std::numeric_limits<double>::infinity();
    double waves = 0.0;
    double k_steps = 0.0;
    double load_bytes = 0.0;
    double mma_instructions = 0.0;
    double epilogue_bytes = 0.0;
    double local_traffic_bytes = 0.0;
};

// Use GEMM's wave-weighted work ordering with the actual compiled kernel
// occupancy. Register and shared-memory limits enter through resident CTAs.
// A local allocation is charged one read and one write per thread and CTA;
// it remains a cost rather than a reason to reject the recipe.
inline CostTerms cost_of(const gemm::GemmRecipe& recipe, const gemm::PlanQuery& query,
                         const gemm::KernelResources& resources, double blocks,
                         double epilogue_bytes) {
    CostTerms result;
    result.geometry = gemm::geometry_cost(recipe, query, resources.resident,
                                          blocks, epilogue_bytes);
    if (!std::isfinite(result.geometry))
        return result;
    result.k_steps = std::ceil(static_cast<double>(query.k) / recipe.k_tile);
    const double resident = std::min(static_cast<double>(resources.resident),
                                     std::ceil(blocks / query.dev.sms));
    result.waves = std::ceil(blocks / (query.dev.sms * resident));
    result.load_bytes = result.k_steps * recipe.k_tile *
                        (recipe.bm * query.ba + recipe.bn * query.bb);
    result.mma_instructions = result.k_steps * (recipe.k_tile / query.mma_k) *
                              (recipe.bm / 16.0) * (recipe.bn / 8.0);
    result.epilogue_bytes = epilogue_bytes;
    result.local_traffic_bytes = 2.0 * resources.local_bytes * recipe.threads;
    result.model = result.waves *
                   (result.load_bytes + result.epilogue_bytes + result.local_traffic_bytes);
    return result;
}

} // namespace astrai::newton_schulz
