#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <kernel/gemm/mainloop.cuh>
#include <epilogue/writer.cuh>
#include <scheduler.cuh>
#include <launcher/gemm_cost.h>
#include <launcher/kernel_resources.cuh>

#include <cmath>
#include <limits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>
#include <algorithm>

#include "entry.h"
#include "cost.h"

namespace {

// Reuse GEMM recipes rather than inventing a separate tile vocabulary.
using namespace astrai::gemm;
// Expose the existing GEMM small-tile widening with its actual warp name.
// Measured single-matrix SYMM recipe. Keep it out of unmeasured geometry planning.
using NarrowSymmTile = GemmTileConfig<Shape<32, 32, 32>, Shape<16, 16>, 2>;
using Tiles = tuple_cat_t<
    TileManifest, std::tuple<small_16w_t<Tile_64x64x64_W16x32_S2>, NarrowSymmTile>>;

template <typename Tile>
std::string tile_name() {
    return std::to_string(Tile::CtaShape::kM) + "x" +
           std::to_string(Tile::CtaShape::kN) + "x" +
           std::to_string(Tile::CtaShape::kK) + "_W" +
           std::to_string(Tile::WarpShape::kM) + "x" +
           std::to_string(Tile::WarpShape::kN) + "_S" +
           std::to_string(Tile::kStages);
}

template <typename Tile, bool RankK, bool ColumnInput = false, bool ColumnOutput = false>
using Policy = GemmPolicy<__nv_bfloat16, __nv_bfloat16,
    std::conditional_t<RankK != ColumnInput, RowMajor, ColMajor>,
    std::conditional_t<RankK && ColumnInput, RowMajor, ColMajor>, Tile,
    std::conditional_t<RankK || ColumnOutput, RowMajor, ColMajor>>;

// Rank-K CTAs own a triangular tile; SYMM uses the existing raster scheduler.
template <typename Tile, bool RankK, bool ColumnInput, bool ColumnOutput>
__global__ void __launch_bounds__(Policy<Tile, RankK, ColumnInput, ColumnOutput>::kCtaThreads,
                                 Policy<Tile, RankK, ColumnInput, ColumnOutput>::kMinCtas)
symmetric_kernel(GemmParams p, const __nv_bfloat16* addend, int64_t stride0,
                 int64_t stride1, float alpha, float beta) {
    using P = Policy<Tile, RankK, ColumnInput, ColumnOutput>;
    using Loop = GemmCollectiveMainloop<P>;
    using Traits = typename P::Traits;
    extern __shared__ __align__(16) char smem[];
    p.a_ptr = static_cast<const __nv_bfloat16*>(p.a_ptr) + blockIdx.z * p.a_batch_stride;
    p.b_ptr = static_cast<const __nv_bfloat16*>(p.b_ptr) + blockIdx.z * p.b_batch_stride;
    p.out_ptr = static_cast<__nv_bfloat16*>(p.out_ptr) + blockIdx.z * p.out_batch_stride;
    if (addend) addend += blockIdx.z * p.out_batch_stride;
    int2 block;
    if constexpr (RankK) {
        int row = static_cast<int>((sqrtf(8.0f * blockIdx.x + 1.0f) - 1.0f) * 0.5f);
        while (row * (row + 1) / 2 > blockIdx.x) --row;
        while ((row + 1) * (row + 2) / 2 <= blockIdx.x) ++row;
        block = int2{row, static_cast<int>(blockIdx.x) - row * (row + 1) / 2};
    } else {
        block = GemmTileScheduler::tile(blockIdx, gridDim, p.raster);
    }
    Loop loop(smem, static_cast<const __nv_bfloat16*>(p.a_ptr),
              static_cast<const __nv_bfloat16*>(p.b_ptr),
              p.m, p.n, p.k, p.a_ld, p.b_ld, threadIdx.x, block);
    typename Loop::AccTensor acc = {};
    loop.prologue();
    loop.accumulate(acc);
    astrai::PipelineSync<Loop::kStages>{}.drain();
    GemmCollectiveEpilogue<P> epilogue(smem, p, block.x, block.y, threadIdx.x);
    auto* output = static_cast<__nv_bfloat16*>(p.out_ptr);
    if constexpr (!RankK) {
        // Fuse the SYMM addend and BF16 rounding with output staging.
        // The rank-K path still mirrors its triangular tile after staging.
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        const int row0 = (warp / Traits::kWarpsN) * Traits::kWarpM + lane / 4;
        const int col0 = (warp % Traits::kWarpsN) * Traits::kWarpN + (lane % 4) * 2;
#pragma unroll
        for (int nt = 0; nt < Traits::kNt; ++nt) {
#pragma unroll
            for (int mt = 0; mt < Traits::kMt; ++mt) {
                const int r0 = row0 + mt * 16;
                const int c0 = col0 + nt * 8;
                const auto& cell = *acc(mt, nt);
#pragma unroll
                for (int element = 0; element < 4; ++element) {
                    const int r = r0 + (element / 2) * 8;
                    const int c = c0 + element % 2;
                    float value = alpha * cell[element];
                    if (addend && block.x * Traits::kBlockM + r < p.m &&
                        block.y * Traits::kBlockN + c < p.n) {
                        const int64_t global_r = block.x * Traits::kBlockM + r;
                        const int64_t global_c = block.y * Traits::kBlockN + c;
                        const int64_t offset = global_c * stride0 + global_r * stride1;
                        value = fmaf(beta, __bfloat162float(addend[offset]), value);
                    }
                    if constexpr (GemmCollectiveEpilogue<P>::t_out)
                        *epilogue.out_elem(c, r) = __float2bfloat16_rn(value);
                    else
                        *epilogue.out_elem(r, c) = __float2bfloat16_rn(value);
                }
            }
        }
        __syncthreads();
        epilogue.store(output);
    } else {
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        const int row0 = block.x * Traits::kBlockM +
                         (warp / Traits::kWarpsN) * Traits::kWarpM + lane / 4;
        const int col0 = block.y * Traits::kBlockN +
                         (warp % Traits::kWarpsN) * Traits::kWarpN + (lane % 4) * 2;
#pragma unroll
        for (int mt = 0; mt < Traits::kMt; ++mt) {
#pragma unroll
            for (int nt = 0; nt < Traits::kNt; ++nt) {
                const int row = row0 + mt * 16, col = col0 + nt * 8;
                auto& cell = *acc(mt, nt);
#pragma unroll
                for (int element = 0; element < 4; ++element) {
                    const int r = row + (element / 2) * 8, c = col + element % 2;
                    cell[element] *= alpha;
                    if (addend && r < p.m && c < p.n) {
                        const int64_t offset = static_cast<int64_t>(r) * stride0 + c * stride1;
                        cell[element] = fmaf(beta, __bfloat162float(addend[offset]), cell[element]);
                    }
                }
            }
        }
        static_assert(Traits::kBlockM == Traits::kBlockN, "triangular tiles must be square");
        epilogue.stage(acc);
        __syncthreads();
        constexpr int dim = Traits::kBlockM;
        // Mirror the rounded lower triangle; both halves are bitwise equal.
        for (int index = threadIdx.x; index < dim * dim; index += blockDim.x) {
            int r = index / dim, c = index % dim;
            const int gr = block.x * dim + r, gc = block.y * dim + c;
            if (gr >= p.m || gc >= p.n) continue;
            if (block.x == block.y && r < c) {
                const int tmp = r; r = c; c = tmp;
            }
            const auto value = *epilogue.out_elem(r, c);
            output[static_cast<int64_t>(gr) * p.out_ld + gc] = value;
        }
        if (block.x != block.y) {
            // Traverse the upper tile by output row; transpose the shared read
            // rather than issuing a strided global store in every warp lane.
            for (int index = threadIdx.x; index < dim * dim; index += blockDim.x) {
                const int r = index / dim, c = index % dim;
                const int gr = block.y * dim + r, gc = block.x * dim + c;
                if (gr < p.m && gc < p.n)
                    output[static_cast<int64_t>(gr) * p.out_ld + gc] =
                        *epilogue.out_elem(c, r);
            }
        }
    }
}

template <typename Tile, bool RankK, bool ColumnInput, bool ColumnOutput>
void launch_tile(GemmParams p, const __nv_bfloat16* addend, int64_t stride0,
                 int64_t stride1, float alpha, float beta, cudaStream_t stream) {
    using P = Policy<Tile, RankK, ColumnInput, ColumnOutput>;
    using T = typename P::Traits;
    TORCH_CHECK(P::kSmemBytes <= at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin,
                "tile exceeds device shared memory limit");
    if constexpr (P::kSmemBytes > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(symmetric_kernel<Tile, RankK, ColumnInput, ColumnOutput>,
                      cudaFuncAttributeMaxDynamicSharedMemorySize, P::kSmemBytes));
    }
    dim3 grid;
    if constexpr (RankK) {
        const int tiles = (p.m + T::kBlockM - 1) / T::kBlockM;
        grid = dim3(tiles * (tiles + 1) / 2, 1, p.batch);
    } else {
        grid = dim3((p.n + T::kBlockN - 1) / T::kBlockN,
                    (p.m + T::kBlockM - 1) / T::kBlockM, p.batch);
    }
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(grid.x <= properties->maxGridSize[0] && grid.y <= properties->maxGridSize[1],
                "tile exceeds device grid limit");
    symmetric_kernel<Tile, RankK, ColumnInput, ColumnOutput><<<grid, P::kCtaThreads, P::kSmemBytes, stream>>>(
        p, addend, stride0, stride1, alpha, beta);
}

template <bool RankK, bool ColumnInput, bool ColumnOutput, size_t I = 0>
bool dispatch_tile(const std::string& name, GemmParams p, const __nv_bfloat16* addend,
                   int64_t stride0, int64_t stride1, float alpha, float beta, cudaStream_t stream) {
    if constexpr (I < std::tuple_size_v<Tiles>) {
        using Tile = std::tuple_element_t<I, Tiles>;
        if constexpr ((!RankK || !std::is_same_v<Tile, NarrowSymmTile>) &&
                      (!RankK || Tile::CtaShape::kM == Tile::CtaShape::kN) &&
                      (!RankK || !ColumnInput || Tile::CtaShape::kK >= 64)) {
            if (name == tile_name<Tile>()) {
                launch_tile<Tile, RankK, ColumnInput, ColumnOutput>(p, addend, stride0, stride1, alpha, beta, stream);
                return true;
            }
        }
        return dispatch_tile<RankK, ColumnInput, ColumnOutput, I + 1>(name, p, addend, stride0, stride1, alpha, beta, stream);
    }
    return false;
}

template <bool RankK>
bool dispatch_layout(const std::string& tile, GemmParams p,
                     const c10::optional<torch::Tensor>& addend, float alpha, float beta,
                     bool column_input, bool column_output, cudaStream_t stream) {
    const auto* data = beta != 0.0f && addend.has_value()
        ? reinterpret_cast<const __nv_bfloat16*>(addend->data_ptr<at::BFloat16>()) : nullptr;
    const int64_t stride0 = data ? addend->stride(-2) : 0;
    const int64_t stride1 = data ? addend->stride(-1) : 0;
    if (column_input) {
        if constexpr (!RankK) {
            if (column_output)
                return dispatch_tile<RankK, true, true>(tile, p, data, stride0, stride1, alpha, beta, stream);
        }
        return dispatch_tile<RankK, true, false>(tile, p, data, stride0, stride1, alpha, beta, stream);
    }
    if constexpr (!RankK) {
        if (column_output)
            return dispatch_tile<RankK, false, true>(tile, p, data, stride0, stride1, alpha, beta, stream);
    }
    return dispatch_tile<RankK, false, false>(tile, p, data, stride0, stride1, alpha, beta, stream);
}

// Price the kernels actually instantiated here, including layout-specific
// register counts and shared-memory allocation. No timing or launch is used.
template <bool RankK, bool ColumnInput, bool ColumnOutput, size_t I = 0>
void append_plans(std::vector<std::pair<double, py::dict>>& rows,
                  const PlanQuery& q, bool addend, const std::string& mode) {
    if constexpr (I < std::tuple_size_v<Tiles>) {
        using Tile = std::tuple_element_t<I, Tiles>;
        if constexpr (!std::is_same_v<Tile, NarrowSymmTile> &&
                      (!RankK || Tile::CtaShape::kM == Tile::CtaShape::kN) &&
                      (!RankK || !ColumnInput || Tile::CtaShape::kK >= 64) &&
                      std::is_same_v<Tile, warp_widened_t<__nv_bfloat16, __nv_bfloat16, Tile>>) {
            using P = Policy<Tile, RankK, ColumnInput, ColumnOutput>;
            auto resource = kernel_resources<
                symmetric_kernel<Tile, RankK, ColumnInput, ColumnOutput>, P>(q);
            const GemmRecipe r{(int)tile_class<Tile>(), Tile::kStages, Tile::kTile,
                Tile::CtaShape::kM, Tile::CtaShape::kN,
                Tile::WarpShape::kM, Tile::WarpShape::kN,
                P::kCtaThreads, P::kSmemBytes};
            const double mt = std::ceil((double)q.m / r.bm);
            const double nt = std::ceil((double)q.n / r.bn);
            if (!RankK && mt > 65535)
                return append_plans<RankK, ColumnInput, ColumnOutput, I + 1>(rows, q, addend, mode);
            const double blocks = q.batch * (RankK ? mt * (mt + 1) / 2 : mt * nt);
            // Average valid epilogue traffic, including partial edge tiles.
            // Diagonal CTAs read a complete valid square before mirroring.
            const double grid = RankK ? mt * (mt + 1) / 2 : mt * nt;
            double cells = (double)q.m * q.n;
            if (addend) {
                if constexpr (RankK) {
                    const double tail = q.m - (mt - 1) * r.bm;
                    const double diagonal = (mt - 1) * r.bm * r.bm + tail * tail;
                    cells += ((double)q.m * q.n + diagonal) / 2;
                } else cells *= 2;
            }
            const auto cost = astrai::newton_schulz::cost_of(
                r, q, resource, blocks, (double)q.out_elem_bytes * cells / grid);
            const double score = mode == "model" ? cost.model : cost.geometry;
            if (std::isfinite(score)) {
                py::dict row;
                row["tile"] = tile_name<Tile>();
                const double operands = static_cast<double>(q.k) *
                    (static_cast<double>(q.m) * q.ba + static_cast<double>(q.n) * q.bb);
                const bool cache_resident = operands <= 0.7 * q.dev.l2_bytes;
                row["raster"] = RankK ? 1 : (mode == "model" && cache_resident ? 0 :
                    std::clamp(geometry_raster(q, r.bm, r.bn), -32, 32));
                row["score"] = score; row["blocks"] = blocks;
                row["source"] = mode;
                row["geometry_score"] = cost.geometry;
                row["model_score"] = cost.model;
                row["waves"] = cost.waves;
                row["k_steps"] = cost.k_steps;
                row["load_bytes"] = cost.load_bytes;
                row["mma_instructions"] = cost.mma_instructions;
                row["epilogue_bytes"] = cost.epilogue_bytes;
                row["local_traffic_bytes"] = cost.local_traffic_bytes;
                row["resident_ctas"] = resource.resident;
                row["registers"] = resource.registers;
                row["local_bytes"] = resource.local_bytes;
                row["shared_memory"] = r.smem;
                rows.emplace_back(score, std::move(row));
            }
        }
        append_plans<RankK, ColumnInput, ColumnOutput, I + 1>(rows, q, addend, mode);
    }
}

py::dict plan_impl(std::string operation, int64_t rows, int64_t cols, int64_t batch_size,
              std::string input_layout, std::string output_layout, bool addend, int device,
              std::string mode) {
    TORCH_CHECK(operation == "syrk" || operation == "symm", "operation must be syrk or symm");
    TORCH_CHECK((input_layout == "row" || input_layout == "column") &&
                (output_layout == "row" || output_layout == "column"), "invalid matrix layout");
    TORCH_CHECK(mode == "model" || mode == "geometry",
                "planner mode must be model or geometry");
    // Match the public BF16 domain, with overflow-safe product limits.
    const int64_t limit = std::numeric_limits<int>::max();
    if (rows < 64 || cols < 64 || rows % 64 || cols % 64 ||
        batch_size < 1 || batch_size > 65535 ||
        rows > limit / rows || cols > limit / rows ||
        batch_size > limit / (rows * cols) || batch_size > limit / (rows * rows))
        return py::dict();
    TORCH_CHECK(device >= 0 && device < c10::cuda::device_count(), "invalid CUDA device ordinal");
    const c10::cuda::CUDAGuard guard(device);
    const auto dev = astrai::device_facts();
    if (dev.cc < 80) return py::dict();
    PlanQuery q{};
    q.m = operation == "syrk" ? rows : cols;
    q.n = rows; q.k = operation == "syrk" ? cols : rows;
    q.batch = batch_size; q.dev = dev; q.tma = false;
    q.crosswise = operation == "syrk" ? (input_layout == "column" ? 2 : 0)
                                            : (input_layout == "column" ? 0 : 1);
    std::vector<std::pair<double, py::dict>> ranked;
    const bool column_input = input_layout == "column", column_output = output_layout == "column";
    if (operation == "syrk") {
        if (column_input) append_plans<true, true, false>(ranked, q, addend, mode);
        else append_plans<true, false, false>(ranked, q, addend, mode);
    } else if (column_input) {
        if (column_output) append_plans<false, true, true>(ranked, q, addend, mode);
        else append_plans<false, true, false>(ranked, q, addend, mode);
    } else {
        if (column_output) append_plans<false, false, true>(ranked, q, addend, mode);
        else append_plans<false, false, false>(ranked, q, addend, mode);
    }
    if (ranked.empty()) return py::dict();
    std::stable_sort(ranked.begin(), ranked.end(),
        [&mode](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first < b.first;
            // Equal byte work favors fewer K-loop iterations without a
            // tuned loop penalty. Geometry mode keeps its original tie order.
            return mode == "model" &&
                   py::cast<double>(a.second["k_steps"]) <
                   py::cast<double>(b.second["k_steps"]);
        });
    py::dict result = ranked.front().second.attr("copy")();
    py::list candidates;
    for (const auto& entry : ranked) candidates.append(entry.second);
    result["candidates"] = candidates;
    return result;
}

template <size_t I = 0>
void append_tiles(py::list& rows, bool rank_k) {
    if constexpr (I < std::tuple_size_v<Tiles>) {
        using Tile = std::tuple_element_t<I, Tiles>;
        using P = Policy<Tile, false>;
        if (!rank_k || (Tile::CtaShape::kM == Tile::CtaShape::kN &&
                        !std::is_same_v<Tile, NarrowSymmTile>)) {
            py::dict row;
            row["name"] = tile_name<Tile>();
            row["input_layouts"] = rank_k && Tile::CtaShape::kK < 64
                ? py::make_tuple("row") : py::make_tuple("row", "column");
            row["block_m"] = Tile::CtaShape::kM; row["block_n"] = Tile::CtaShape::kN;
            row["block_k"] = Tile::CtaShape::kK;
            row["warp_m"] = Tile::WarpShape::kM; row["warp_n"] = Tile::WarpShape::kN;
            row["stages"] = Tile::kStages; row["threads"] = P::kCtaThreads;
            row["shared_memory"] = P::kSmemBytes;
            rows.append(row);
        }
        append_tiles<I + 1>(rows, rank_k);
    }
}

py::list tiles_impl(std::string operation) {
    TORCH_CHECK(operation == "syrk" || operation == "symm", "operation must be syrk or symm");
    py::list rows;
    append_tiles(rows, operation == "syrk");
    return rows;
}

} // namespace

namespace astrai::newton_schulz::symmetric {

void launch_syrk(gemm::GemmParams p, const c10::optional<torch::Tensor>& addend,
                 float alpha, float beta, const std::string& tile,
                 bool column_input, cudaStream_t stream) {
    TORCH_CHECK(dispatch_layout<true>(tile, p, addend, alpha, beta,
                column_input, false, stream), "unknown SYRK tile: ", tile);
}

void launch_symm(gemm::GemmParams p, const c10::optional<torch::Tensor>& addend,
                 float alpha, float beta, const std::string& tile,
                 bool column_input, bool column_output, cudaStream_t stream) {
    TORCH_CHECK(dispatch_layout<false>(tile, p, addend, alpha, beta,
                column_input, column_output, stream), "unknown SYMM tile: ", tile);
}

py::dict plan(std::string operation, int64_t rows, int64_t cols, int64_t batch_size,
              std::string input_layout, std::string output_layout, bool addend, int device,
              std::string mode) {
    return plan_impl(operation, rows, cols, batch_size, input_layout, output_layout,
                     addend, device, mode);
}

py::list tiles(std::string operation) {
    return tiles_impl(operation);
}

} // namespace astrai::newton_schulz::symmetric
