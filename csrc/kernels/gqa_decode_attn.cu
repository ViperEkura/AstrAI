// per-KV-head block, K shared in smem, each thread handles hd/32 elements
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <torch/extension.h>

using bf16 = __nv_bfloat16;

constexpr int CHUNK = 64;

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void gqa_decode_attn_kernel(
    const bf16* __restrict__ q_ptr,
    const bf16* __restrict__ k_ptr,
    const bf16* __restrict__ v_ptr,
    const bool*  __restrict__ mask_ptr,
    bf16* __restrict__ out_ptr,
    int B, int n_heads, int n_kv_heads, int seq_len, int hd
) {
    int batch = blockIdx.x / n_kv_heads;
    int kv_head = blockIdx.x % n_kv_heads;
    int group_size = blockDim.y;
    int q_head = kv_head * group_size + threadIdx.y;
    int lane = threadIdx.x;
    int hd_per_thread = hd / 32;

    float q_reg[8];
    int q_off = ((batch * n_heads + q_head) * 1) * hd + lane * hd_per_thread;
    #pragma unroll
    for (int i = 0; i < hd_per_thread; i++)
        q_reg[i] = __bfloat162float(q_ptr[q_off + i]);

    int kv_base = ((batch * n_kv_heads + kv_head) * seq_len) * hd;
    int mask_base = batch * seq_len;

    float m = -FLT_MAX, d = 0.0f, acc_reg[8] = {0.0f};
    float scale = rsqrtf((float)hd);

    extern __shared__ __align__(16) bf16 k_smem[];

    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK) {
        int this_chunk = min(CHUNK, seq_len - chunk_start);

        int total = this_chunk * hd;
        for (int i = threadIdx.y * 32 + lane; i < total; i += blockDim.x * blockDim.y)
            k_smem[i] = k_ptr[kv_base + chunk_start * hd + i];
        __syncthreads();

        for (int s = 0; s < this_chunk; s++) {
            float partial = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd_per_thread; i++)
                partial += q_reg[i] * __bfloat162float(k_smem[s * hd + lane * hd_per_thread + i]);
            partial = warp_reduce_sum(partial) * scale;

            if (!mask_ptr[mask_base + chunk_start + s]) partial = -FLT_MAX;

            float new_m = fmaxf(m, partial);
            float alpha = expf(m - new_m);
            float beta  = expf(partial - new_m);
            d = d * alpha + beta;

            int v_off = kv_base + (chunk_start + s) * hd + lane * hd_per_thread;
            #pragma unroll
            for (int i = 0; i < hd_per_thread; i++)
                acc_reg[i] = acc_reg[i] * alpha + __bfloat162float(v_ptr[v_off + i]) * beta;
            m = new_m;
        }
        __syncthreads();
    }

    int out_off = ((batch * n_heads + q_head) * 1) * hd + lane * hd_per_thread;
    #pragma unroll
    for (int i = 0; i < hd_per_thread; i++)
        out_ptr[out_off + i] = __float2bfloat16(acc_reg[i] / d);
}

torch::Tensor gqa_decode_attn(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && mask.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(k.dtype() == torch::kBFloat16);
    TORCH_CHECK(v.dtype() == torch::kBFloat16);
    TORCH_CHECK(mask.dtype() == torch::kBool);
    TORCH_CHECK(q.size(2) == 1, "Q seq_len must be 1");

    int B = q.size(0), n_heads = q.size(1), n_kv = k.size(1);
    int seq_len = k.size(2), hd = q.size(3);
    TORCH_CHECK(hd % 32 == 0, "head_dim must be multiple of 32");
    int group_size = n_heads / n_kv;
    auto out = torch::empty_like(q);

    size_t smem = CHUNK * hd * sizeof(bf16); // K chunk
    dim3 block(32, group_size);
    dim3 grid(B * n_kv);

    gqa_decode_attn_kernel<<<grid, block, smem>>>(
        reinterpret_cast<const bf16*>(q.data_ptr()),
        reinterpret_cast<const bf16*>(k.data_ptr()),
        reinterpret_cast<const bf16*>(v.data_ptr()),
        mask.data_ptr<bool>(),
        reinterpret_cast<bf16*>(out.data_ptr()),
        B, n_heads, n_kv, seq_len, hd
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gqa_decode_attn", &gqa_decode_attn, "GQA decode v2 (per-KV-head, shared K)");
}
