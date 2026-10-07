#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <kernel/gemm/mainloop.cuh>
#include <epilogue/writer.cuh>
#include <scheduler.cuh>

#include <cmath>
#include <limits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

namespace astrai::symmetric::syrk {

// A 64x64 triangular CTA, with four warps computing one 32x32 quadrant
// each. Inspired by the tile dispatch in StarrickLiu/fused-muon (Apache-2.0),
// but implemented using CUDA WMMA and shared memory without CuTe/CUTLASS.
template <bool Addend>
__global__ void syrk64_kernel(const __nv_bfloat16* input,
                              const __nv_bfloat16* addend,
                              __nv_bfloat16* output,
                              int dim,
                              int reduction,
                              float alpha,
                              float beta) {
    using namespace nvcuda;
    input += static_cast<int64_t>(blockIdx.z) * dim * reduction;
    if (addend) addend += static_cast<int64_t>(blockIdx.z) * dim * dim;
    output += static_cast<int64_t>(blockIdx.z) * dim * dim;
    constexpr int kTile = 64;
    constexpr int kMma = 16;
    constexpr int kTileElements = kTile * kTile;
    constexpr int kPanelStride = kTile + 8;
    constexpr int kPanelElements = kTile * kPanelStride;
    constexpr int kOutputStride = kTile + 8;

    int tile_row = static_cast<int>((sqrtf(8.0f * blockIdx.x + 1.0f) - 1.0f) * 0.5f);
    while (tile_row * (tile_row + 1) / 2 > blockIdx.x) {
        --tile_row;
    }
    while ((tile_row + 1) * (tile_row + 2) / 2 <= blockIdx.x) {
        ++tile_row;
    }
    const int tile_col = blockIdx.x - tile_row * (tile_row + 1) / 2;
    const int row = tile_row * kTile;
    const int col = tile_col * kTile;

    __shared__ __align__(16) __nv_bfloat16 panel_a[kPanelElements];
    __shared__ __align__(16) __nv_bfloat16 panel_b[kPanelElements];
    __shared__ float accumulator[kTileElements];
    __shared__ __nv_bfloat16 rounded[kTile * kOutputStride];

    const int warp = threadIdx.x / warpSize;
    const int warp_row = (warp / 2) * 32;
    const int warp_col = (warp % 2) * 32;
    wmma::fragment<wmma::matrix_a, kMma, kMma, kMma, __nv_bfloat16, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, kMma, kMma, kMma, __nv_bfloat16, wmma::col_major> b0, b1;
    wmma::fragment<wmma::accumulator, kMma, kMma, kMma, float> p00, p01, p10, p11;
    wmma::fill_fragment(p00, 0.0f);
    wmma::fill_fragment(p01, 0.0f);
    wmma::fill_fragment(p10, 0.0f);
    wmma::fill_fragment(p11, 0.0f);

    for (int k_base = 0; k_base < reduction; k_base += kTile) {
        // Each cp.async transfers eight BF16 values along the contiguous K axis.
        constexpr int kSegments = kTile * (kTile / 8);
        for (int segment = threadIdx.x; segment < kSegments; segment += blockDim.x) {
            const int outer = segment / (kTile / 8);
            const int inner = segment % (kTile / 8) * 8;
            const int panel_index = outer * kPanelStride + inner;
            const auto* source_a = input + (row + outer) * reduction + k_base + inner;
            const auto* source_b = input + (col + outer) * reduction + k_base + inner;
            const uint32_t target_a = __cvta_generic_to_shared(panel_a + panel_index);
            const uint32_t target_b = __cvta_generic_to_shared(panel_b + panel_index);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(target_a),
                         "l"(source_a));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(target_b),
                         "l"(source_b));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        for (int k = 0; k < kTile; k += kMma) {
            const int a_offset = warp_row * kPanelStride + k;
            const int b_offset = warp_col * kPanelStride + k;
            wmma::load_matrix_sync(a0, panel_a + a_offset, kPanelStride);
            wmma::load_matrix_sync(a1, panel_a + a_offset + kMma * kPanelStride, kPanelStride);
            wmma::load_matrix_sync(b0, panel_b + b_offset, kPanelStride);
            wmma::load_matrix_sync(b1, panel_b + b_offset + kMma * kPanelStride, kPanelStride);
            wmma::mma_sync(p00, a0, b0, p00);
            wmma::mma_sync(p01, a0, b1, p01);
            wmma::mma_sync(p10, a1, b0, p10);
            wmma::mma_sync(p11, a1, b1, p11);
        }
        __syncthreads();
    }

    float* warp_output = accumulator + warp_row * kTile + warp_col;
    wmma::store_matrix_sync(warp_output, p00, kTile, wmma::mem_row_major);
    wmma::store_matrix_sync(warp_output + kMma, p01, kTile, wmma::mem_row_major);
    wmma::store_matrix_sync(warp_output + kMma * kTile, p10, kTile, wmma::mem_row_major);
    wmma::store_matrix_sync(warp_output + kMma * kTile + kMma, p11, kTile, wmma::mem_row_major);
    __syncthreads();

    for (int index = threadIdx.x; index < kTileElements; index += blockDim.x) {
        const int local_row = index / kTile;
        const int local_col = index % kTile;
        if (tile_row != tile_col || local_row >= local_col) {
            float value = alpha * accumulator[index];
            if constexpr (Addend) {
                const int offset = (row + local_row) * dim + col + local_col;
                value = fmaf(beta, __bfloat162float(addend[offset]), value);
            }
            const __nv_bfloat16 result = __float2bfloat16_rn(value);
            rounded[local_row * kOutputStride + local_col] = result;
            if (tile_row == tile_col && local_row != local_col) {
                rounded[local_col * kOutputStride + local_row] = result;
            }
        }
    }
    __syncthreads();

    for (int index = threadIdx.x; index < kTileElements; index += blockDim.x) {
        const int local_row = index / kTile;
        const int local_col = index % kTile;
        output[(row + local_row) * dim + col + local_col] =
            rounded[local_row * kOutputStride + local_col];
        if (tile_row != tile_col) {
            output[(col + local_row) * dim + row + local_col] =
                rounded[local_col * kOutputStride + local_row];
        }
    }
}

} // namespace astrai::symmetric::syrk

namespace {

bool dense_matrix(const torch::Tensor& x) {
    if (x.dim() != 2 && x.dim() != 3) return false;
    if (x.dim() == 3 && (x.size(0) < 1 || x.size(0) > 65535 ||
                        x.stride(0) != x.size(-2) * x.size(-1))) return false;
    return x.stride(-1) == 1 && x.stride(-2) == x.size(-1) ||
           x.stride(-2) == 1 && x.stride(-1) == x.size(-2);
}

void check_buffers(const torch::Tensor& x, const torch::Tensor& output) {
    TORCH_CHECK(x.is_cuda() && output.is_cuda(), "x and output must be CUDA tensors");
    TORCH_CHECK(x.device() == output.device(), "x and output must share a device");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
                "x and output must be bfloat16");
    TORCH_CHECK((x.dim() == 2 || x.dim() == 3) && output.dim() == x.dim(),
                "x and output must be matrices or matrix batches");
    TORCH_CHECK(x.dim() == 2 || x.size(0) == output.size(0), "batch size mismatch");
    const auto rows = x.size(-2);
    const auto reduction = x.size(-1);
    TORCH_CHECK(rows >= 64 && rows % 64 == 0 && reduction >= 64 && reduction % 64 == 0,
                "matrix dimensions must be positive multiples of 64");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "input must be 16-byte aligned");
    TORCH_CHECK(dense_matrix(x) && dense_matrix(output), "x and output must be dense row/column-major");
    TORCH_CHECK(output.size(-2) == rows && output.size(-1) == rows,
                "output must have matching square matrix shape");
    TORCH_CHECK(!x.is_alias_of(output), "input and output must not alias");
    TORCH_CHECK(rows * rows <= std::numeric_limits<int>::max() &&
                    reduction <= std::numeric_limits<int>::max(),
                "matrix dimensions exceed the kernel limit");
}


void check_addend(const torch::Tensor& output, const c10::optional<torch::Tensor>& addend,
                  float alpha, float beta) {
    TORCH_CHECK(std::isfinite(alpha) && std::isfinite(beta), "coefficients must be finite");
    TORCH_CHECK(beta == 0.0f || addend.has_value(), "nonzero beta requires an addend");
    if (!addend.has_value() || beta == 0.0f) return;
    TORCH_CHECK(addend->device() == output.device() &&
                addend->scalar_type() == output.scalar_type() &&
                addend->sizes() == output.sizes() && dense_matrix(*addend),
                "addend must match output device, dtype, shape and layout");
    TORCH_CHECK(!output.is_alias_of(*addend), "output must not alias addend");
}

// Reuse GEMM recipes rather than inventing a separate tile vocabulary.
using namespace astrai::gemm;
using Tiles = TileManifest;

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
                    const int64_t offset = RankK ? static_cast<int64_t>(r) * stride0 + c * stride1
                                                 : static_cast<int64_t>(c) * stride0 + r * stride1;
                    cell[element] = fmaf(beta, __bfloat162float(addend[offset]), cell[element]);
                }
            }
        }
    }
    GemmCollectiveEpilogue<P> epilogue(smem, p, block.x, block.y, threadIdx.x);
    auto* output = static_cast<__nv_bfloat16*>(p.out_ptr);
    if constexpr (!RankK) {
        epilogue.run(acc, output);
    } else {
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
    symmetric_kernel<Tile, RankK, ColumnInput, ColumnOutput><<<grid, P::kCtaThreads, P::kSmemBytes, stream>>>(
        p, addend, stride0, stride1, alpha, beta);
}

template <bool RankK, bool ColumnInput, bool ColumnOutput, size_t I = 0>
bool dispatch_tile(const std::string& name, GemmParams p, const __nv_bfloat16* addend,
                   int64_t stride0, int64_t stride1, float alpha, float beta, cudaStream_t stream) {
    if constexpr (I < std::tuple_size_v<Tiles>) {
        using Tile = std::tuple_element_t<I, Tiles>;
        if constexpr ((!RankK || Tile::CtaShape::kM == Tile::CtaShape::kN) &&
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

const __nv_bfloat16* addend_data(const c10::optional<torch::Tensor>& addend, float beta) {
    return beta != 0.0f && addend.has_value()
        ? reinterpret_cast<const __nv_bfloat16*>(addend->data_ptr<at::BFloat16>()) : nullptr;
}

void syrk_out(torch::Tensor x, torch::Tensor output,
              c10::optional<torch::Tensor> addend, float alpha, float beta,
              std::string tile) {
    check_buffers(x, output);
    check_addend(output, addend, alpha, beta);
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8, "BF16 tensor cores required");
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto* c_data = addend_data(addend, beta);
    const auto* input = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto* result = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    if (tile == "wmma64") {
        TORCH_CHECK(x.is_contiguous(), "wmma64 requires row-major input");
        const int tiles = x.size(-2) / 64;
        const dim3 grid(tiles * (tiles + 1) / 2, 1, x.dim() == 3 ? x.size(0) : 1);
        if (c_data)
            astrai::symmetric::syrk::syrk64_kernel<true><<<grid, 128, 0, stream>>>(
                input, c_data, result, x.size(-2), x.size(-1), alpha, beta);
        else
            astrai::symmetric::syrk::syrk64_kernel<false><<<grid, 128, 0, stream>>>(
                input, nullptr, result, x.size(-2), x.size(-1), alpha, beta);
    } else {
        GemmParams p{};
        p.a_ptr = input; p.b_ptr = input; p.out_ptr = result;
        p.m = x.size(-2); p.n = x.size(-2); p.k = x.size(-1);
        p.batch = x.dim() == 3 ? x.size(0) : 1;
        p.a_batch_stride = p.b_batch_stride = x.size(-2) * x.size(-1);
        p.out_batch_stride = x.size(-2) * x.size(-2);
        p.a_ld = x.stride(-1) == 1 ? x.size(-1) : x.size(-2);
        p.b_ld = p.a_ld; p.out_ld = x.size(-2);
        TORCH_CHECK(dispatch_layout<true>(tile, p, addend, alpha, beta, !x.is_contiguous(), false, stream),
                    "unknown SYRK tile: ", tile);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void symm_out(torch::Tensor symmetric, torch::Tensor x, torch::Tensor output,
              c10::optional<torch::Tensor> addend, float alpha, float beta,
              std::string tile, int raster) {
    TORCH_CHECK(symmetric.is_cuda() && x.is_cuda() && output.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(symmetric.device() == x.device() && x.device() == output.device(), "device mismatch");
    TORCH_CHECK(symmetric.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16 &&
                output.scalar_type() == at::kBFloat16, "all tensors must be bfloat16");
    TORCH_CHECK((x.dim() == 2 || x.dim() == 3) &&
                symmetric.dim() == x.dim() && output.dim() == x.dim(),
                "expected matrices or matrix batches");
    TORCH_CHECK(x.dim() == 2 || symmetric.size(0) == x.size(0), "batch size mismatch");
    const auto rows = x.size(-2), cols = x.size(-1);
    TORCH_CHECK(symmetric.size(-2) == rows && symmetric.size(-1) == rows &&
                output.sizes() == x.sizes(), "matrix shape mismatch");
    TORCH_CHECK(rows >= 64 && cols >= 64 && rows % 64 == 0 && cols % 64 == 0,
                "matrix dimensions must be positive multiples of 64");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(symmetric.data_ptr()) % 16 == 0,
                "inputs must be 16-byte aligned");
    TORCH_CHECK(dense_matrix(symmetric) && dense_matrix(x) && dense_matrix(output),
                "all tensors must be dense row/column-major");
    TORCH_CHECK(!output.is_alias_of(x) && !output.is_alias_of(symmetric),
                "output must not alias inputs");
    TORCH_CHECK(rows * cols <= std::numeric_limits<int>::max() &&
                rows * rows <= std::numeric_limits<int>::max(), "matrix exceeds kernel limit");
    TORCH_CHECK(raster >= -32 && raster <= 32, "raster must be in [-32, 32]");
    check_addend(output, addend, alpha, beta);
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8, "BF16 tensor cores required");
    GemmParams p{};
    // (S X)^T = X^T S: no materialized transpose, epilogue restores orientation.
    p.a_ptr = x.data_ptr(); p.b_ptr = symmetric.data_ptr(); p.out_ptr = output.data_ptr();
    p.m = cols; p.n = rows; p.k = rows;
    p.batch = x.dim() == 3 ? x.size(0) : 1;
    p.a_batch_stride = p.out_batch_stride = rows * cols;
    p.b_batch_stride = rows * rows;
    p.a_ld = x.is_contiguous() ? cols : rows;
    p.b_ld = rows; p.out_ld = output.is_contiguous() ? cols : rows; p.raster = raster;
    TORCH_CHECK(dispatch_layout<false>(tile, p, addend, alpha, beta,
                !x.is_contiguous(), !output.is_contiguous(), at::cuda::getCurrentCUDAStream().stream()), "unknown SYMM tile: ", tile);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <size_t I = 0>
void append_tiles(py::list& rows, bool rank_k) {
    if constexpr (I < std::tuple_size_v<Tiles>) {
        using Tile = std::tuple_element_t<I, Tiles>;
        using P = Policy<Tile, false>;
        if (!rank_k || Tile::CtaShape::kM == Tile::CtaShape::kN) {
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

py::list tiles(std::string operation) {
    TORCH_CHECK(operation == "syrk" || operation == "symm", "operation must be syrk or symm");
    py::list rows;
    if (operation == "syrk") {
        py::dict row;
        row["input_layouts"] = py::make_tuple("row");
        row["name"] = "wmma64"; row["block_m"] = 64; row["block_n"] = 64;
        row["block_k"] = 64; row["warp_m"] = 32; row["warp_n"] = 32;
        row["stages"] = 1; row["threads"] = 128; row["shared_memory"] = 44032;
        rows.append(row);
    }
    append_tiles(rows, operation == "syrk");
    return rows;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiles", &tiles, py::arg("operation"));
    m.def("symm_out", &symm_out, py::arg("symmetric"), py::arg("x"), py::arg("output"),
          py::arg("addend") = py::none(), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("tile") = "64x64x32_W16x32_S2", py::arg("raster") = 1);
    m.def("syrk_out", &syrk_out, py::arg("x"), py::arg("output"),
          py::arg("addend") = py::none(), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("tile") = "wmma64");
}
