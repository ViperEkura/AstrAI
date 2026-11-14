#pragma once
/*
 * Single-row GEMV route: capability gate and launch.
 *
 * The tile pipeline pads every problem up to a kBlockM-tall CTA tile, so a
 * decode step (m == 1) pays 63/64 or 127/128 of its MMA work for rows that do
 * not exist while still streaming the whole weight. The row GEMV keeps the
 * weight stream and drops the padding, which is the decode bottleneck:
 * m == 1 throughput is weight-bandwidth bound, not tensor-core bound.
 *
 * The route is a strict subset of the tile pipeline's contract:
 *   - m == 1 and batch == 1 (a batched decode keeps the tile path);
 *   - both operands contract-contiguous (crosswise_of == 0), so a row of B is
 *     b + n * b_ld and a row of A is a + 0 * a_ld;
 *   - row-major output (the swapped symmetric-NN lowering is excluded);
 *   - an element pair the row dot product has a kernel for.
 * Everything else falls through to the unchanged manifest dispatch.
 */
#include <type_traits>

#include <api/native/gemm.h>
#include <kernel/gemm/gemv.cuh>
#include <policy/gemm/manifest.h>
#include <utils/launch.cuh>

namespace astrai {
namespace gemm {

// Activation and weight dtypes the row dot product folds into fp32.
template <typename T>
inline constexpr bool gemv_activation_v = std::is_same_v<T, bf16> || std::is_same_v<T, fp16>;
template <typename T>
inline constexpr bool gemv_weight_v =
    std::is_same_v<T, bf16> || std::is_same_v<T, fp16> || std::is_same_v<T, int8_t>;

template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename LayoutOut,
          typename OutT>
inline constexpr bool gemv_eligible_v =
    gemv_activation_v<ElemA> && gemv_weight_v<ElemB> && std::is_same_v<LayoutOut, RowMajor> &&
    (std::is_same_v<OutT, bf16> || std::is_same_v<OutT, float>) &&
    crosswise_of<LayoutA, LayoutB>() == 0;

/* Eight rows per block: wide rows keep N/8 blocks in flight on one wave. */
constexpr int kGemvWarps = 8;

/*
 * True when the GEMV was launched; false means the caller keeps its path.
 * Row count rounds up: the last block's idle warps exit before any load.
 */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename LayoutOut,
          typename OutT>
bool try_launch_gemv([[maybe_unused]] const GemmParams& p,
                     [[maybe_unused]] cudaStream_t stream) {
    if constexpr (gemv_eligible_v<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT>) {
        if (p.m != 1 || p.batch != 1)
            return false;
        const int blocks = (p.n + kGemvWarps - 1) / kGemvWarps;
        gemv_kernel<ElemA, ElemB, OutT, kGemvWarps>
            <<<blocks, kGemvWarps * 32, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
        return true;
    } else {
        return false;
    }
}

} // namespace gemm
} // namespace astrai
