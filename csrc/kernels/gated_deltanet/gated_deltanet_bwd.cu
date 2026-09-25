// Backward of the Gated DeltaNet output stage (FLA's `chunk_bwd_o`).
//
// Forward, per chunk of one (b, h), on head-major tensors:
//
//   gh    = chunk-local cumsum of the gate
//   decay = e^(gh_j - gh_l) for j >= l, zero above the diagonal
//   A     = (q k^T) * decay
//   qe    = q * e^gh
//   o     = scale * (qe @ h + A @ v_new)
//
// so the reverse pass over one chunk is
//
//   dO     = scale * do
//   dA     = dO @ v_new^T                      (reduces over V)
//   d_qe   = dO @ h^T                          (reduces over V)
//   dv_new = A^T @ dO
//   dh    += qe^T @ dO                         (accumulated across chunks)
//   dq     = d_qe * e^gh + (dA * decay) @ k
//   dk     = (dA * decay)^T @ q
//   dgh    = sum_K(d_qe * qe) + rowsum(dA * A) - colsum(dA * A)
//
// `dh` and `dv_new` are written by more than one stage of the reverse pass, so
// they are float32 accumulators that this kernel adds into with atomics; every
// other output is owned by exactly one block.
//
// This version uses plain FMA rather than tensor cores. The reverse pass is
// where a wrong index is expensive to find, so it is written to be checkable
// against autograd first; the tiling follows the 99 KB shared-memory ceiling of
// this part (one block per SM), which is also why `do` is read from global
// rather than staged — the accesses within a warp hit the same address and
// broadcast.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "common/launch.cuh"
#include "gated_deltanet.h"

namespace {

constexpr int kThreads = 256;
constexpr int kChunk = 64;
constexpr int kHeadDim = 128;

constexpr int kQElem = kChunk * kChunk;          // 4096
constexpr int kDElem = kChunk * kHeadDim;        // 8192 per (token, dim) tensor
constexpr int kHElem = kHeadDim * kHeadDim;      // 16384

// 64 x 64 outputs over 256 threads.
constexpr int kAPerThread = kQElem / kThreads;   // 16
// 64 x 128 outputs over 256 threads.
constexpr int kDPerThread = kDElem / kThreads;   // 32
// 128 x 128 outputs over 256 threads: dh is twice as large as the rest.
constexpr int kHPerThread = kHElem / kThreads;   // 64

__global__ void gated_deltanet_bwd_o_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v_new,
    const __nv_bfloat16* __restrict__ h,
    const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ do_ptr,
    __nv_bfloat16* __restrict__ dq,
    __nv_bfloat16* __restrict__ dk,
    float* __restrict__ dv_new,
    float* __restrict__ dh,
    float* __restrict__ dg,
    int seq,
    int heads,
    float scale
) {
    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* k_s = q_s + kDElem;
    __nv_bfloat16* vn_s = k_s + kDElem;
    __nv_bfloat16* h_s = vn_s + kDElem;
    float* a_s = reinterpret_cast<float*>(h_s + kHElem);
    float* gh_s = a_s + kQElem;
    float* dgh_s = gh_s + kChunk;
    // e^gh, precomputed: every decay below is a ratio of these rather than an
    // exp in the innermost loop, which is where the time went (the dh loop alone
    // was recomputing 4096 exponentials per thread). The ratio is safe because
    // decay only ever uses j >= l, where it is <= 1.
    float* egh_s = dgh_s + kChunk;

    const int tid = threadIdx.x;
    const int chunks = seq / kChunk;
    const int ic = blockIdx.x;
    const int bh = blockIdx.y;
    const int hh = bh % heads;
    const int b = bh / heads;
    const int t0 = ic * kChunk;

    const int64_t base = (static_cast<int64_t>(b) * heads + hh) * seq + t0;
    const __nv_bfloat16* q_base = q + base * kHeadDim;
    const __nv_bfloat16* k_base = k + base * kHeadDim;
    const __nv_bfloat16* vn_base = v_new + base * kHeadDim;
    const __nv_bfloat16* do_base = do_ptr + base * kHeadDim;
    const __nv_bfloat16* h_base = h + (static_cast<int64_t>(bh) * chunks + ic)
        * kHElem;
    __nv_bfloat16* dq_base = dq + base * kHeadDim;
    __nv_bfloat16* dk_base = dk + base * kHeadDim;
    float* dvn_base =
        dv_new + (static_cast<int64_t>(bh) * seq + t0) * kHeadDim;
    float* dh_base = dh + static_cast<int64_t>(bh) * kHElem;

    for (int i = tid; i < kDElem; i += kThreads) {
        q_s[i] = q_base[i];
        k_s[i] = k_base[i];
        vn_s[i] = vn_base[i];
    }
    for (int i = tid; i < kHElem; i += kThreads) {
        h_s[i] = h_base[i];
    }
    if (tid < kChunk) {
        gh_s[tid] = g[base + tid];
        dgh_s[tid] = 0.0f;
        egh_s[tid] = __expf(g[base + tid]);
    }
    __syncthreads();

    const float inv_scale = scale;

    // A = (q k^T) * decay on the lower triangle.
    for (int i = 0; i < kAPerThread; ++i) {
        const int idx = tid + i * kThreads;
        const int j = idx / kChunk;
        const int l = idx % kChunk;
        float acc = 0.0f;
        if (j >= l) {
            for (int d = 0; d < kHeadDim; ++d) {
                acc = fmaf(
                    __bfloat162float(q_s[j * kHeadDim + d]),
                    __bfloat162float(k_s[l * kHeadDim + d]),
                    acc
                );
            }
            acc *= egh_s[j] / egh_s[l];
        }
        a_s[idx] = acc;
    }
    __syncthreads();

    // dA and d_qe: each thread owns a fixed set of outputs and reduces over the
    // full V extent, so both live in registers across the whole pass.
    float da_reg[kAPerThread];
    float dqe_reg[kDPerThread];
#pragma unroll
    for (int i = 0; i < kAPerThread; ++i) {
        const int idx = tid + i * kThreads;
        const int j = idx / kChunk;
        const int l = idx % kChunk;
        float acc = 0.0f;
        for (int v = 0; v < kHeadDim; ++v) {
            acc = fmaf(
                __bfloat162float(do_base[j * kHeadDim + v]),
                __bfloat162float(vn_s[l * kHeadDim + v]),
                acc
            );
        }
        da_reg[i] = acc * inv_scale;
    }
#pragma unroll
    for (int i = 0; i < kDPerThread; ++i) {
        const int idx = tid + i * kThreads;
        const int j = idx / kHeadDim;
        const int d = idx % kHeadDim;
        // d_qe = dO @ h^T: the reduction runs over V, which is h's second axis,
        // and K is the free one. Summing over h's first axis instead is silently
        // correct whenever h happens to be symmetric, so this axis assignment is
        // the thing to re-check if dq ever drifts.
        float acc = 0.0f;
        for (int v = 0; v < kHeadDim; ++v) {
            acc = fmaf(
                __bfloat162float(do_base[j * kHeadDim + v]),
                __bfloat162float(h_s[d * kHeadDim + v]),
                acc
            );
        }
        dqe_reg[i] = acc * inv_scale;
    }

    // dv_new = A^T @ dO, accumulated (the state stage adds its own part later).
    for (int i = 0; i < kDPerThread; ++i) {
        const int idx = tid + i * kThreads;
        const int j = idx / kHeadDim;
        const int v = idx % kHeadDim;
        float acc = 0.0f;
        for (int l = 0; l < kChunk; ++l) {
            acc = fmaf(
                a_s[l * kChunk + j],
                __bfloat162float(do_base[l * kHeadDim + v]),
                acc
            );
        }
        if (acc != 0.0f) {
            atomicAdd(&dvn_base[j * kHeadDim + v], acc * inv_scale);
        }
    }

    // dh += qe^T @ dO. This one covers [K, V], not [chunk, K].
    for (int i = 0; i < kHPerThread; ++i) {
        const int idx = tid + i * kThreads;
        const int d = idx / kHeadDim;
        const int v = idx % kHeadDim;
        float acc = 0.0f;
        for (int j = 0; j < kChunk; ++j) {
            acc = fmaf(
                __bfloat162float(q_s[j * kHeadDim + d]) * egh_s[j],
                __bfloat162float(do_base[j * kHeadDim + v]),
                acc
            );
        }
        if (acc != 0.0f) {
            atomicAdd(&dh_base[d * kHeadDim + v], acc * inv_scale);
        }
    }
    __syncthreads();

    // Publish dA so the dq/dk/dgh products can read it, and fold the two dgh
    // terms that only need dA, A and the gate.
    for (int i = 0; i < kAPerThread; ++i) {
        const int idx = tid + i * kThreads;
        const int j = idx / kChunk;
        const int l = idx % kChunk;
        const float da = da_reg[i];
        const float prod = da * a_s[idx];
        if (prod != 0.0f) {
            atomicAdd(&dgh_s[j], prod);
            atomicAdd(&dgh_s[l], -prod);
        }
        a_s[idx] = da;
    }
    __syncthreads();

    // dq = d_qe * e^gh + (dA * decay) @ k, and the d_qe side of dgh.
    for (int i = 0; i < kDPerThread; ++i) {
        const int idx = tid + i * kThreads;
        const int j = idx / kHeadDim;
        const int d = idx % kHeadDim;
        float acc = dqe_reg[i] * egh_s[j];
        for (int l = 0; l < kChunk; ++l) {
            const float disc = j >= l ? egh_s[j] / egh_s[l] : 0.0f;
            acc = fmaf(
                a_s[j * kChunk + l] * disc,
                __bfloat162float(k_s[l * kHeadDim + d]),
                acc
            );
        }
        dq_base[j * kHeadDim + d] = __float2bfloat16(acc);
        const float qe = __bfloat162float(q_s[j * kHeadDim + d]) * egh_s[j];
        const float contrib = dqe_reg[i] * qe;
        if (contrib != 0.0f) {
            atomicAdd(&dgh_s[j], contrib);
        }
    }

    // dk = (dA * decay)^T @ q.
    for (int i = 0; i < kDPerThread; ++i) {
        const int idx = tid + i * kThreads;
        const int l = idx / kHeadDim;
        const int d = idx % kHeadDim;
        float acc = 0.0f;
        for (int j = 0; j < kChunk; ++j) {
            const float disc = j >= l ? egh_s[j] / egh_s[l] : 0.0f;
            acc = fmaf(
                a_s[j * kChunk + l] * disc,
                __bfloat162float(q_s[j * kHeadDim + d]),
                acc
            );
        }
        dk_base[l * kHeadDim + d] = __float2bfloat16(acc);
    }
    __syncthreads();

    if (tid < kChunk) {
        dg[base + tid] = dgh_s[tid];
    }
}

}  // namespace

std::vector<torch::Tensor> gated_deltanet_bwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v_new,
    torch::Tensor h,
    torch::Tensor g,
    torch::Tensor do_grad,
    double scale
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v_new.is_cuda() && h.is_cuda(),
                "q/k/v_new/h must be on CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bf16");
    TORCH_CHECK(v_new.scalar_type() == torch::kBFloat16, "v_new must be bf16");
    TORCH_CHECK(h.scalar_type() == torch::kBFloat16, "h must be bf16");
    TORCH_CHECK(do_grad.scalar_type() == torch::kBFloat16, "do must be bf16");
    TORCH_CHECK(g.scalar_type() == torch::kFloat32, "g must be fp32");
    TORCH_CHECK(q.dim() == 4 && q.size(3) == kHeadDim, "q must be [B, H, T, ", kHeadDim, "]");
    TORCH_CHECK(v_new.sizes() == q.sizes(), "v_new must match q");
    TORCH_CHECK(do_grad.sizes() == q.sizes(), "do must match q");
    TORCH_CHECK(g.size(2) == q.size(2), "g must be [B, H, T] matching q");
    TORCH_CHECK(h.dim() == 5, "h must be [B, H, chunks, K, V]");
    TORCH_CHECK(q.size(2) % kChunk == 0, "seq must be a multiple of ", kChunk);
    for (const auto* named : {&q, &k, &v_new, &do_grad}) {
        TORCH_CHECK(named->is_contiguous(), "q/k/v_new/do must be head-major contiguous");
    }
    TORCH_CHECK(h.is_contiguous() && g.is_contiguous(), "h/g must be contiguous");

    const int batch = q.size(0);
    const int heads = q.size(1);
    const int seq = q.size(2);
    const int chunks = seq / kChunk;

    auto dq = torch::empty_like(q);
    auto dk = torch::empty_like(k);
    auto dv_new = torch::zeros({batch, heads, seq, kHeadDim}, q.options().dtype(torch::kFloat32));
    auto dh = torch::zeros({batch, heads, kHeadDim, kHeadDim}, q.options().dtype(torch::kFloat32));
    auto dg = torch::empty({batch, heads, seq}, g.options());

    const size_t smem_bytes = kDElem * 3 * sizeof(__nv_bfloat16)
        + kHElem * sizeof(__nv_bfloat16) + kQElem * sizeof(float)
        + 3 * kChunk * sizeof(float);
    static bool configured = false;
    if (!configured) {
        ASTRAI_CUDA_CHECK(cudaFuncSetAttribute(
            gated_deltanet_bwd_o_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        ));
        configured = true;
    }

    gated_deltanet_bwd_o_kernel<<<dim3(chunks, batch * heads), kThreads,
                                          smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_new.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),
        g.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(do_grad.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()),
        dv_new.data_ptr<float>(),
        dh.data_ptr<float>(),
        dg.data_ptr<float>(),
        seq, heads, static_cast<float>(scale)
    );
    ASTRAI_LAUNCH_CHECK();

    return {dq, dk, dv_new, dh, dg};
}
