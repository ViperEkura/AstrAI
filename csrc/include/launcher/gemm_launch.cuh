#pragma once
/* Typed CUDA launch and TMA descriptor setup. */
#include <cstdio>
#include <type_traits>

#include <kernel/gemm/kernel.cuh>
#include <launcher/plan_types.h>
#include <utils/device.cuh>
#include <utils/launch.cuh>

namespace astrai {
namespace gemm {

/*
 * Launchers — pure CUDA, usable from the binding and the C tests. The
 * runtime knobs live in GemmConfig (launcher/plan_types.h), owned at runtime by
 * astrai.extension.policy.gemm.plan.
 */

// Grid for one Policy's tile: N x M blocks, batch on z.
template <typename Traits> dim3 gemm_grid(const GemmParams& p) {
    return dim3((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
                (p.m + Traits::kBlockM - 1) / Traits::kBlockM, p.batch);
}

// One plan-log line per launch (" mx" marks the block_scale cell).
inline void log_gemm_plan(const GemmParams& p,
                          const dim3& grid,
                          int bm,
                          int bn,
                          int k_stages,
                          int smem,
                          bool tma,
                          bool mx = false) {
    if (!gemm_plan_log_enabled())
        return;
    std::fprintf(stderr,
                 "[gemm-plan] %lldx%lldx%lld b=%d -> tile %dx%d s%d%s%s "
                 "grid %dx%dx%d raster %d smem %d\n",
                 (long long)p.m, (long long)p.n, (long long)p.k, p.batch, bm, bn, k_stages,
                 tma ? " tma" : "", mx ? " mx" : "", grid.x, grid.y, grid.z, p.raster, smem);
}

/*
 * Launch with the smem budget: >48KB opt-ins once per instantiation via
 * cudaFuncSetAttribute. Templated on the kernel VALUE (auto NTTP) so
 * same-signature kernels never share the armed flag; a failed opt-in arms
 * nothing and the launch fails loudly.
 */
template <auto Kernel, typename... Args>
void launch_with_smem(int smem_bytes, dim3 grid, dim3 block, cudaStream_t stream, Args... args) {
    if (smem_bytes > 48 * 1024) {
        static bool armed = false; // per instantiation
        if (!armed) {
            const cudaError_t err = cudaFuncSetAttribute(
                Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            armed = (err == cudaSuccess);
        }
    }
    Kernel<<<grid, block, smem_bytes, stream>>>(args...);
    ASTRAI_LAUNCH_CHECK();
}

template <typename Policy> void launch_policy(GemmParams p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    dim3 grid = gemm_grid<Traits>(p);
    log_gemm_plan(p, grid, Traits::kBlockM, Traits::kBlockN, Traits::kStages, Policy::kSmemBytes,
                  /*tma=*/false, Traits::kMxCell);
    launch_with_smem<gemm_kernel<Policy>>(Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads),
                                          stream, p);
}

/*
 * TMA staging (sm_90+): descriptor build + the TMA twin of launch_policy;
 * descriptors are cached exact-match (tma.cuh), the encode is paid once.
 */

/*
 * Output-reclaim feasibility: a fat output (fp32) can outgrow the ring the
 * planner priced — the wide CTA's 128x256 of fp32 is 131072B against the
 * 73728B a 1-byte pair leaves. Exactly the launch twins' reclaim
 * static_assert, so a new CTA class cannot drift from it.
 */
template <typename Tile, typename ElemA, typename ElemB, typename OutT>
constexpr bool reclaim_fits() {
    return Tile::CtaShape::kM * Tile::CtaShape::kN * sizeof(OutT) <=
           ring_smem_bytes(Tile::CtaShape::kM, Tile::CtaShape::kN, Tile::CtaShape::kK,
                           Tile::kStages, (int)sizeof(ElemA), (int)sizeof(ElemB));
}

/*
 * Both operand descriptors for one TMA Policy. Dim/stride in bytes along
 * the contract dim; batch is a third dim only when it strides. Swizzle mode,
 * box extents and byte scaling derive from the staging layout (tma_spec in
 * memory/tma.cuh) — the same instances the fragment readers consume.
 */
template <typename Policy>
bool tma_maps_for(const GemmParams& p, CUtensorMap* ma, CUtensorMap* mb) {
    using Mainloop = GemmCollectiveMainloop<Policy>;
    const auto a = astrai::tma_map_cache().lookup(
        astrai::tma_spec<typename Mainloop::ElemA, typename Mainloop::SmemLayoutA,
                         Mainloop::kBlockM>(p.a_ptr, p.m, p.k, p.a_ld, p.batch, p.a_batch_stride));
    if (!a)
        return false;
    *ma = *a;
    const auto b = astrai::tma_map_cache().lookup(
        astrai::tma_spec<typename Mainloop::ElemB, typename Mainloop::SmemLayoutB,
                         Mainloop::kBlockN>(p.b_ptr, p.n, p.k, p.b_ld, p.batch, p.b_batch_stride));
    if (!b)
        return false;
    *mb = *b;
    return true;
}

/*
 * TMA launch for one Policy; false (nothing launched) on an undescribable
 * operand (misaligned base/ld) so the caller falls back to cp.async.
 */
template <typename Policy> bool launch_policy_tma(const GemmParams& p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    /*
     * The planner prices rings only; TMA's pad + barriers can tip past
     * the opt-in ceiling on the fattest pair — fall back, not fail.
     */
    if (Policy::kSmemBytes > astrai::device_facts().smem_max)
        return false;
    CUtensorMap ma{}, mb{};
    if (!tma_maps_for<Policy>(p, &ma, &mb))
        return false;
    dim3 grid = gemm_grid<Traits>(p);
    log_gemm_plan(p, grid, Traits::kBlockM, Traits::kBlockN, Traits::kStages, Policy::kSmemBytes,
                  /*tma=*/true, Traits::kMxCell);
    /*
     * Rank bits pick the instantiation: strided batch rides the 3D
     * emitters, broadcast keeps the shared 2D map; the per-stage pick
     * compiles away.
     */
    auto launch_rank = [&](auto rank3a, auto rank3b) {
        launch_with_smem<gemm_kernel_tma<Policy, decltype(rank3a)::value, decltype(rank3b)::value>>(
            Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads), stream, p, ma, mb);
    };
    if (p.batch > 1 && p.a_batch_stride > 0 && p.b_batch_stride > 0)
        launch_rank(std::true_type{}, std::true_type{});
    else if (p.batch > 1 && p.a_batch_stride > 0)
        launch_rank(std::true_type{}, std::false_type{});
    else if (p.batch > 1 && p.b_batch_stride > 0)
        launch_rank(std::false_type{}, std::true_type{});
    else
        launch_rank(std::false_type{}, std::false_type{});
    return true;
}

} // namespace gemm
} // namespace astrai
