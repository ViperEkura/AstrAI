// GDN preparation kernels: projection layout to chunk-kernel layout, in two launches.
//
// The chunked GDN kernels want head-major tensors ([B, H, T, D], head dim
// contiguous) with L2-normalized query/key rows and the gate pre-scanned into a
// chunk-local cumsum. What the layer's projections plus the local convolution
// hand over is [B, H, D, T] — head and head-dim outer, *token inner* — so going
// from one to the other is a real transpose, not a relabeling.
//
// Doing that with torch costs more than all four GDN kernels together: measured
// at T=2048 and T=8192 the transposes, dtype casts, L2 norms and the cumsum were
// 53% and 59% of the operator's wall clock, because each was its own launch over
// the whole tensor. Here the source tile is read with the token axis coalesced,
// staged in shared memory, and read back transposed so the head-dim axis of the
// store is coalesced too. The two operations collapse into two launches.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "common/launch.cuh"
#include "gated_deltanet/gated_deltanet.h"

namespace {

constexpr int kThreads = 256;
constexpr int kHeadDim = 128;  // D: the Qwen3.5 linear-attention head shape
constexpr int kTile = 64;      // tokens per block
constexpr int kQuarters = 4;   // norm reduction is split kQuarters ways
// Row pitch in shared memory. 66 bf16 is 33 words, so the transposed read (one
// thread per head dim) walks consecutive banks instead of colliding; a 16-byte
// multiple would be conflict-prone, which is why the vector stores below are
// scalar.
constexpr int kPitch = kTile + 2;

// One block per (b, h, token tile).
__global__ void gated_deltanet_fwd_qkv_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int64_t stride_qb,
    int64_t stride_kb,
    int64_t stride_vb,
    int seq,
    int heads,
    float eps
) {
    __shared__ __nv_bfloat16 q_s[kHeadDim * kPitch];
    __shared__ __nv_bfloat16 k_s[kHeadDim * kPitch];
    __shared__ __nv_bfloat16 v_s[kHeadDim * kPitch];
    __shared__ float partial_q[kQuarters][kTile];
    __shared__ float partial_k[kQuarters][kTile];
    __shared__ float scale_q[kTile];
    __shared__ float scale_k[kTile];

    const int tid = threadIdx.x;
    const int tiles = (seq + kTile - 1) / kTile;
    const int tile = blockIdx.x % tiles;
    const int bh = blockIdx.x / tiles;
    const int h = bh % heads;
    const int b = bh / heads;
    const int t0 = tile * kTile;

    // Source row (b, h, d, t) sits at b * stride_b + (h * D + d) * seq + t. The
    // batch stride is the caller's: q/k/v are slices of one concatenated
    // convolution buffer, so it spans all three of their channel counts and is
    // not H * D * T. It only looks that way when the batch size is 1.
    const int64_t q_row = static_cast<int64_t>(b) * stride_qb + h * kHeadDim * seq;
    const int64_t k_row = static_cast<int64_t>(b) * stride_kb + h * kHeadDim * seq;
    const int64_t v_row = static_cast<int64_t>(b) * stride_vb + h * kHeadDim * seq;
    // Two threads per head dim, each taking half the tile's tokens: 8 tokens per
    // vector load, four loads.
    const int load_d = tid >> 1;
    const int load_j = (tid & 1) * (kTile / 2);
#pragma unroll
    for (int i = 0; i < kTile / 2; i += 8) {
        const int j = load_j + i;
        const int64_t d_off = static_cast<int64_t>(load_d) * seq + j + t0;
        const uint4 qv = *reinterpret_cast<const uint4*>(q + q_row + d_off);
        const uint4 kv = *reinterpret_cast<const uint4*>(k + k_row + d_off);
        const uint4 vv = *reinterpret_cast<const uint4*>(v + v_row + d_off);
        const __nv_bfloat16* qh = reinterpret_cast<const __nv_bfloat16*>(&qv);
        const __nv_bfloat16* kh = reinterpret_cast<const __nv_bfloat16*>(&kv);
        const __nv_bfloat16* vh = reinterpret_cast<const __nv_bfloat16*>(&vv);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            q_s[load_d * kPitch + j + e] = qh[e];
            k_s[load_d * kPitch + j + e] = kh[e];
            v_s[load_d * kPitch + j + e] = vh[e];
        }
    }
    __syncthreads();

    // L2 norm down each token column: kQuarters threads fold part of the column
    // and the partials combine through shared memory, so the reduction never
    // needs the whole 128-long column in one thread.
    const int col = tid % kTile;
    const int quarter = tid / kTile;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    constexpr int kRowsPerQuarter = kHeadDim / kQuarters;
#pragma unroll
    for (int r = 0; r < kRowsPerQuarter; ++r) {
        const int row = quarter * kRowsPerQuarter + r;
        const float qv = __bfloat162float(q_s[row * kPitch + col]);
        const float kv = __bfloat162float(k_s[row * kPitch + col]);
        q_sum = fmaf(qv, qv, q_sum);
        k_sum = fmaf(kv, kv, k_sum);
    }
    partial_q[quarter][col] = q_sum;
    partial_k[quarter][col] = k_sum;
    __syncthreads();
    if (tid < kTile) {
        float qs = 0.0f;
        float ks = 0.0f;
#pragma unroll
        for (int r = 0; r < kQuarters; ++r) {
            qs += partial_q[r][tid];
            ks += partial_k[r][tid];
        }
        scale_q[tid] = rsqrtf(qs + eps);
        scale_k[tid] = rsqrtf(ks + eps);
    }
    __syncthreads();

    // Transposed store: consecutive threads write consecutive head dims.
    const int out_d = tid % kHeadDim;
    const int64_t out =
        (static_cast<int64_t>(b) * heads + h) * seq * kHeadDim + t0 * kHeadDim;
#pragma unroll
    for (int j = tid / kHeadDim; j < kTile; j += kThreads / kHeadDim) {
        const int64_t off = out + static_cast<int64_t>(j) * kHeadDim + out_d;
        q_out[off] = __float2bfloat16(
            __bfloat162float(q_s[out_d * kPitch + j]) * scale_q[j]
        );
        k_out[off] = __float2bfloat16(
            __bfloat162float(k_s[out_d * kPitch + j]) * scale_k[j]
        );
        v_out[off] = v_s[out_d * kPitch + j];
    }
}

// One block per (b, h, chunk), with the block size equal to the chunk. The
// source is [B, T, H] with H contiguous, so the gather is strided while the
// scatter is contiguous.
__global__ void gated_deltanet_fwd_gates_kernel(
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ g_out,
    float* __restrict__ beta_out,
    int seq,
    int heads,
    int chunk
) {
    extern __shared__ float scan[];
    const int i = threadIdx.x;
    const int chunks = (seq + chunk - 1) / chunk;
    const int ic = blockIdx.x % chunks;
    const int bh = blockIdx.x / chunks;
    const int h = bh % heads;
    const int b = bh / heads;
    const int start = ic * chunk;
    const int index = start + i;

    scan[i] = g[(b * seq + index) * heads + h];
    // The first step reads a neighbour, so the stores above must be visible
    // before the loop starts; without this barrier the scan races.
    __syncthreads();

    // Inclusive scan over the chunk (Hillis-Steele). The chunk size is the block
    // size, so each doubling needs one barrier pair and no cross-block combine.
    for (int step = 1; step < chunk; step <<= 1) {
        const float other = i >= step ? scan[i - step] : 0.0f;
        __syncthreads();
        scan[i] += other;
        __syncthreads();
    }

    const int out = (b * heads + h) * seq + index;
    g_out[out] = scan[i];
    beta_out[out] = beta[(b * seq + index) * heads + h];
}

}  // namespace

std::vector<torch::Tensor> gated_deltanet_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    double eps,
    int64_t chunk
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be on CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bf16");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bf16");
    TORCH_CHECK(g.scalar_type() == torch::kFloat32, "g must be fp32 (log decay)");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "beta must be fp32");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q/k/v must be [B, T, H, D]");
    TORCH_CHECK(g.dim() == 3 && beta.dim() == 3, "g/beta must be [B, T, H]");
    TORCH_CHECK(g.is_contiguous() && beta.is_contiguous(), "g/beta must be contiguous");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "q/k/v must share a shape");
    TORCH_CHECK(g.size(0) == q.size(0) && g.size(1) == q.size(1) &&
                    g.size(2) == q.size(2),
                "g/beta must be [B, T, H] matching q");

    const int batch = q.size(0);
    const int seq = q.size(1);
    const int heads = q.size(2);
    const int dim = q.size(3);
    TORCH_CHECK(dim == kHeadDim, "head dim must be ", kHeadDim, " for the prep kernel");
    TORCH_CHECK(chunk == kTile, "the scan kernel needs chunk == ", kTile);
    // Tiles must cover the sequence exactly: the vector loads below do not
    // mask, and the caller pads to a multiple of the chunk anyway.
    TORCH_CHECK(seq % kTile == 0, "seq must be a multiple of ", kTile);

    // The layer hands over [B, T, H, D] whose memory is laid out as [B, H, D, T],
    // so the token axis is the contiguous one. Accepting any other layout would
    // silently read the wrong elements, so it is checked rather than assumed.
    for (const auto* named : {&q, &k, &v}) {
        TORCH_CHECK(
            named->stride(1) == 1 && named->stride(3) == seq &&
                named->stride(2) == static_cast<int64_t>(seq) * dim,
            "q/k/v must carry the projection layout ([B, H, D, T] in memory)"
        );
    }

    auto head_major = [&](const torch::Tensor& t) {
        return torch::empty({batch, heads, seq, dim}, t.options());
    };
    auto q_out = head_major(q);
    auto k_out = head_major(k);
    auto v_out = head_major(v);
    auto g_out = torch::empty({batch, heads, seq}, g.options());
    auto beta_out = torch::empty({batch, heads, seq}, beta.options());

    const int tiles = (seq + kTile - 1) / kTile;
    gated_deltanet_fwd_qkv_kernel<<<batch * heads * tiles, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr()),
        q.stride(0), k.stride(0), v.stride(0),
        seq, heads, static_cast<float>(eps)
    );
    ASTRAI_LAUNCH_CHECK();

    const int chunks = (seq + chunk - 1) / chunk;
    gated_deltanet_fwd_gates_kernel<<<batch * heads * chunks, static_cast<int>(chunk),
                            chunk * sizeof(float), stream>>>(
        g.data_ptr<float>(),
        beta.data_ptr<float>(),
        g_out.data_ptr<float>(),
        beta_out.data_ptr<float>(),
        seq, heads, static_cast<int>(chunk)
    );
    ASTRAI_LAUNCH_CHECK();

    return {q_out, k_out, v_out, g_out, beta_out};
}
