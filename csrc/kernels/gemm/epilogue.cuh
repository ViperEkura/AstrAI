#pragma once
// Collective epilogue: fused bias, the bf16 scatter of the fp32
// accumulators through the reclaimed operand shared memory, and the
// coalesced copy-out. The staging swizzle is one instance of the unified
// family (common/swizzle.cuh) shared with the operand staging in load.cuh.

#include "common/swizzle.cuh"
#include "common/tensor.cuh"
#include "gemm/common.h"
#include "policy.cuh"

namespace astrai {
namespace gemm {

// Output-element packing facts for the staging tile: elements per 16B
// chunk, pair packing for the vectorized scatter, and the scalar
// conversion. bf16 is the fused-linear default; fp32 keeps the fp32
// accumulators unrounded (training dX/dW style outputs).
template <typename OutT>
struct OutElem;

template <>
struct OutElem<__nv_bfloat16> {
    using T2 = __nv_bfloat162;
    static constexpr int kChunkElems = 8;  // per 16B chunk
    static constexpr int kChunkShift = 3;
    static __device__ __forceinline__ __nv_bfloat162 pack2(float a, float b) {
        return __floats2bfloat162_rn(a, b);
    }
    static __device__ __forceinline__ __nv_bfloat16 cvt(float a) {
        return __float2bfloat16_rn(a);
    }
};

template <>
struct OutElem<float> {
    using T2 = float2;
    static constexpr int kChunkElems = 4;
    static constexpr int kChunkShift = 2;
    static __device__ __forceinline__ float2 pack2(float a, float b) {
        return make_float2(a, b);
    }
    static __device__ __forceinline__ float cvt(float a) { return a; }
};

template <typename Policy>
struct GemmCollectiveEpilogue {
    using Traits = typename Policy::Traits;
    using OutT = typename Policy::OutT;
    using OE = OutElem<OutT>;
    // The mainloop's accumulator element: fp32 for the float mma families,
    // int32 for the native s8 pair — the scale/bias folding below applies
    // after the int->float conversion, so both share one scatter path.
    using AccT = typename Traits::AccT;
    static constexpr bool kStreamOut = Policy::kStreamOut;
    static constexpr int kBlockM = Traits::kBlockM;
    static constexpr int kBlockN = Traits::kBlockN;
    static constexpr int kMt = Traits::kMt;
    static constexpr int kNt = Traits::kNt;
    using AccTensor = typename Traits::AccTensor;

    const float output_scale;
    const float* const a_scale;  // [a_scale_m] row factor or null
    const float* const b_scale;  // [b_scale_n] col factor or null
    const __nv_bfloat16* const bias;
    const int64_t m, n, out_ld;
    // Output orientation from the policy tag (CUTLASS LayoutC): the NN
    // swap computes E = B^T A^T, instantiated with LayoutOut = ColMajor.
    static constexpr bool t_out =
        !std::is_same_v<typename Policy::LayoutTagOut, RowMajor>;
    // Staged row width (rows and row length trade places under the swap),
    // its 16B-chunk count, and the staged-output layout — composition
    // (Swizzle, Layout<Shape, Stride>) over the chunk grid; a custom
    // instance (the row field XORs straight onto the chunk field).
    static constexpr int kRowElems = t_out ? kBlockM : kBlockN;
    static constexpr int kRowRows = t_out ? kBlockN : kBlockM;
    static constexpr int kRowChunks = kRowElems / OE::kChunkElems;
    static constexpr int kRowBits = log2_const<kRowChunks>::value;
    using OutLayout = decltype(
        composition(Swizzle<kRowBits, kRowBits>{},
                    Layout<Shape<kRowRows, kRowChunks>, Stride<kRowChunks, 1>>{}));
    // The staged output tile, typed by OutLayout (common/tensor.cuh):
    // operator()(row, elem) is the swizzled address.
    const Tensor<PtrEngine<OutT>, OutLayout> out_tile;
    const int row_elems, row_chunks;
    const int warp_m, warp_n, group, thread_in_group;
    const int64_t block_m, block_n;

    __device__ GemmCollectiveEpilogue(char* smem, const GemmParams& p,
                                     int64_t block_m, int64_t block_n, int tid)
        : out_tile{reinterpret_cast<OutT*>(smem)},
          output_scale((p.a_scale && p.a_scale_m == 0 ? *p.a_scale : 1.0f) *
                       (p.b_scale && p.b_scale_n == 0 ? *p.b_scale : 1.0f)),
          a_scale(p.a_scale_m > 0 ? p.a_scale : nullptr),
          b_scale(p.b_scale_n > 0 ? p.b_scale : nullptr),
          bias(reinterpret_cast<const __nv_bfloat16*>(p.bias_ptr)),
          m(p.m), n(p.n), out_ld(p.out_ld),
          row_elems(kRowElems),
          row_chunks(kRowChunks),
          warp_m((tid >> 5) / Traits::kWarpsN),
          warp_n((tid >> 5) % Traits::kWarpsN),
          group((tid & 31) >> 2),
          thread_in_group(tid & 3),
          block_m(block_m), block_n(block_n) {}

    // Swizzled address of one 16B chunk (row r, chunk c) of the staged
    // tile — the OutLayout instance riding out_tile. Plain orientation:
    // kBlockM rows of kBlockN elems; out-transposed (swap dispatch): rows
    // and row length trade places. Both row-chunk counts are powers of
    // two, keeping the XOR swizzle well-defined.
    __device__ __forceinline__ OutT* out_chunk(int r, int c) const {
        return out_tile(r, c << OE::kChunkShift);
    }
    __device__ __forceinline__ OutT* out_elem(int r, int v) const {
        return out_tile(r, v);
    }

    // Scatter the accumulators into the staging tile: the operand rings are
    // dead once the mainloop ends, so their space stages the bf16 output
    // tile. Threads scatter (STS.32 of bf16x2 pairs), a barrier makes the
    // tile coherent, then the whole CTA copies it out in fully-coalesced
    // 16B chunks. The 16B-chunk XOR swizzle keeps both the scatter and the
    // gather conflict-free.
    __device__ __forceinline__ void stage(AccTensor& acc) const {
        // Fused bias: added to the fp32 accumulator before the single bf16
        // rounding. The per-lane loads are L1 broadcasts; rows past the
        // edge skip the load (their smem slots never copy out). Under
        // the swapped orientation the bias indexes D-cols = the kernel's rows.
        const int local_col0 = warp_n * Traits::kWarpN + thread_in_group * 2;
        const int64_t bias_col0 = block_n * kBlockN;
        const int64_t bias_row0 = block_m * kBlockM;
        if (!t_out) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col = local_col0 + nt * 8;
                const int64_t gcol = bias_col0 + col;
                const float b0 = bias && gcol < n 
                    ? __bfloat162float(bias[gcol]) 
                    : 0.0f;
                const float b1 = bias && gcol + 1 < n 
                    ? __bfloat162float(bias[gcol + 1])
                    : 0.0f;
                const float c0 = b_scale && gcol < n ? b_scale[gcol] : 1.0f;
                const float c1 = b_scale && gcol + 1 < n ? b_scale[gcol + 1] : 1.0f;
#pragma unroll
                for (int mt = 0; mt < kMt; ++mt) {
                    const int r0 = warp_m * Traits::kWarpM + group + mt * 16;
                    // Per-row activation scale: D-row == kernel row here.
                    const float rfac = a_scale && bias_row0 + r0 < m
                            ? __ldcg(a_scale + bias_row0 + r0)
                            : 1.0f;
                    const float rfac8 = a_scale && bias_row0 + r0 + 8 < m
                        ? __ldcg(a_scale + bias_row0 + r0 + 8)
                        : 1.0f;
                    const auto& cell = *acc(mt, nt);
                    // Two bf16x2 stores per accumulator tile: rows g and
                    // g+8 of the m16n8 output, columns tig*2/tig*2+1 inside
                    // one 16B chunk.
                    const int off = col & (OE::kChunkElems - 1);  // in-chunk elems
                    *reinterpret_cast<typename OE::T2*>(
                        out_chunk(r0, col >> OE::kChunkShift) + off) =
                        OE::pack2((float)cell[0] * output_scale * rfac * c0 + b0,
                                  (float)cell[1] * output_scale * rfac * c1 + b1);
                    *reinterpret_cast<typename OE::T2*>(
                        out_chunk(r0 + 8, col >> OE::kChunkShift) + off) =
                        OE::pack2((float)cell[2] * output_scale * rfac8 * c0 + b0,
                                  (float)cell[3] * output_scale * rfac8 * c1 + b1);
                }
            }
        } else {
            // Transposed scatter: accumulator (kernel row r0, col) is
            // D[col0_global + col][row0_global + r0], staged at T[col][r0].
            // The acc pair spans two staged rows, so these are scalar
            // stores (the swap path is the rare NN layout). OOB elements
            // store dead lanes of the tile, never copied out.
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col = local_col0 + nt * 8;
#pragma unroll
                for (int mt = 0; mt < kMt; ++mt) {
                    const int r0 = warp_m * Traits::kWarpM + group + mt * 16;
                    const int64_t grow = bias_row0 + r0;
                    const int64_t grow8 = grow + 8;
                    const float b =
                        bias && grow < m ? __bfloat162float(bias[grow]) : 0.0f;
                    const float b8 =
                        bias && grow8 < m ? __bfloat162float(bias[grow8]) : 0.0f;
                    // Swapped orientation mirrors bias: D-cols come from
                    // kernel rows (the +8 acc half carries its own), D-rows
                    // from kernel cols.
                    const float c = b_scale && grow < m
                        ? b_scale[grow]
                        : 1.0f;
                    const float c8 = b_scale && grow8 < m
                        ? b_scale[grow8]
                        : 1.0f;
                    const float r0f = a_scale && bias_col0 + col < n
                        ? __ldcg(a_scale + bias_col0 + col)
                        : 1.0f;
                    const float r1f = a_scale && bias_col0 + col + 1 < n
                        ? __ldcg(a_scale + bias_col0 + col + 1)
                        : 1.0f;
                    const auto& cell = *acc(mt, nt);
                    *out_elem(col, r0) = OE::cvt((float)cell[0] * output_scale * r0f * c + b);
                    *out_elem(col + 1, r0) = OE::cvt((float)cell[1] * output_scale * r1f * c + b);
                    *out_elem(col, r0 + 8) =  OE::cvt((float)cell[2] * output_scale * r0f * c8 + b8);
                    *out_elem(col + 1, r0 + 8) = OE::cvt((float)cell[3] * output_scale * r1f * c8 + b8);
                }
            }
        }
    }

    // Coalesced copy-out: thread -> one 16B chunk; consecutive threads walk
    // a row so each global transaction covers a full 128B line. Under the
    // swap the staged rows are D-rows counted from block_n's stripe while
    // the row length is kernel m', so row/stride flip to the swapped dims.
    __device__ __forceinline__ void store(OutT* out) const {
        constexpr int kTotalChunks =
            kBlockM * (kBlockN / OE::kChunkElems);  // == kBlockN * (kBlockM/chunk)
        const int64_t row0_global = block_m * kBlockM;
        const int64_t col0_global = block_n * kBlockN;
        for (int idx = threadIdx.x; idx < kTotalChunks; idx += kCtaThreads) {
            const int r = idx / row_chunks;
            const int c = idx % row_chunks;
            const uint4 v = *reinterpret_cast<const uint4*>(out_chunk(r, c));
            const int64_t row = t_out 
                ? (int64_t)block_n * kBlockN + r
                : row0_global + r;
            const int64_t col = t_out 
                ? row0_global + (int64_t)c * OE::kChunkElems
                : col0_global + (int64_t)c * OE::kChunkElems;
            const int64_t rows_total = t_out ? n : m;
            const int64_t row_stride = out_ld;
            
            if (row >= rows_total) break;  // rows are consecutive: nothing left
            auto* dst = out + row * row_stride + col;
            if (col + OE::kChunkElems <= row_stride &&
                (reinterpret_cast<uintptr_t>(dst) & 15) == 0) {
                if constexpr (Policy::kStoreWriteThrough) {
                    // Streaming write-through: the output is read-once (no
                    // future reuse), so bypass the L2 write-back stage and
                    // preserve L2 for the reused weight/activation tiles.
                    __stwt(reinterpret_cast<uint4*>(dst), v);
                } else if constexpr (kStreamOut) {
                    // Evict-first streaming store knob: neutral on L20
                    // squares, -3..4% on rects; kept for other SKUs.
                    __stcs(reinterpret_cast<uint4*>(dst), v);
                } else {
                    *reinterpret_cast<uint4*>(dst) = v;
                }
            } else {
                // Row-edge chunk or an odd-stride row base: spill the
                // elements that survive the row edge.
                const OutT* elems = reinterpret_cast<const OutT*>(&v);
                for (int e = 0; e < OE::kChunkElems && col + e < row_stride; ++e)
                    dst[e] = elems[e];
            }
        }
    }

    __device__ __forceinline__ void run(AccTensor& acc, OutT* out) {
        stage(acc);
        __syncthreads();
        store(out);
    }

  private:
    static constexpr int kCtaThreads = Traits::kCtaThreads;
};

}  // namespace gemm
}  // namespace astrai
