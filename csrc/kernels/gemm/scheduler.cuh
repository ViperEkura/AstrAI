#pragma once
// Tile scheduler: the linear CTA id maps to (block_m, block_n) in a
// runtime-selected raster order (GemmParams::raster):
//   raster > 0 — grouped raster: a group of `raster` M-tile rows sweeps
//     all N columns, M walked fastest inside the group, so consecutive
//     CTAs share one B column stripe (tall-ish outputs);
//   raster < 0 — mirrored: a group of -raster N-tile columns sweeps all
//     M rows, N walked fastest, consecutive CTAs share one A row stripe
//     (wide outputs, e.g. dW = g^T @ x);
//   raster == 0 — plain N-fastest raster (kRasterGroup=0's measured best
//     for dX's crosswise-B layouts where grouping was neutral).
// The order is a runtime value (chosen by plan_gemm's aspect heuristic)
// because making it a template parameter would multiply kernel
// instantiations; the scheduler runs once per CTA, so the branch is free.

namespace astrai {
namespace gemm {

struct GemmTileScheduler {
    __device__ static int2 tile(const uint3& block, const dim3& blocks,
                                int raster) {
        const int bid = int(block.y) * int(blocks.x) + int(block.x);
        if (raster > 0) {
            const int group_first_m = (bid / (raster * int(blocks.x))) * raster;
            const int group_rows =
                min(int(blocks.y) - group_first_m, raster);  // M-tail group is short
            return int2{group_first_m + bid % group_rows,
                        (bid % (raster * int(blocks.x))) / group_rows};
        } else if (raster < 0) {
            const int width = -raster;
            const int group_first_n = (bid / (width * int(blocks.y))) * width;
            const int group_cols =
                min(int(blocks.x) - group_first_n, width);  // N-tail group is short
            return int2{(bid % (width * int(blocks.y))) / group_cols,
                        group_first_n + bid % group_cols};
        } else {
            return int2{int(block.y), int(block.x)};
        }
    }
};

}  // namespace gemm
}  // namespace astrai
