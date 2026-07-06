#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <torch/extension.h>

using bf16 = __nv_bfloat16;

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void gqa_prefill_attn_kernel(
    const bf16* __restrict__ q_ptr,
    const bf16* __restrict__ k_ptr,
    const bf16* __restrict__ v_ptr,
    const bool*  __restrict__ mask_ptr,
    bf16* __restrict__ out_ptr,
    int B, int n_heads, int n_kv_heads, int q_len, int kv_len, int hd,
    int use_mask, int is_causal, int causal_offset
) {
    int flat_id = blockIdx.x;
    int pos = flat_id % q_len;
    flat_id /= q_len;
    int q_head = flat_id % n_heads;
    int batch = flat_id / n_heads;
    int kv_head = q_head / (n_heads / n_kv_heads);
    int lane = threadIdx.x;
    int hd_per_thread = hd / 32;

    // each thread handles hd/32 elements of Q
    float q_reg[8];
    int q_off = ((batch * n_heads + q_head) * q_len + pos) * hd + lane * hd_per_thread;
    #pragma unroll
    for (int i = 0; i < hd_per_thread; i++)
        q_reg[i] = __bfloat162float(q_ptr[q_off + i]);

    int kv_base = ((batch * n_kv_heads + kv_head) * kv_len) * hd;
    int limit = is_causal ? min(pos + causal_offset + 1, kv_len) : kv_len;

    float m = -FLT_MAX, d = 0.0f, acc_reg[8] = {0.0f};
    float scale = rsqrtf((float)hd);

    int mask_stride = q_len * kv_len;
    int mask_off = batch * mask_stride + pos * kv_len;

    for (int s = 0; s < limit; s++) {
        float partial = 0.0f;
        int k_off = kv_base + s * hd + lane * hd_per_thread;
        #pragma unroll
        for (int i = 0; i < hd_per_thread; i++)
            partial += q_reg[i] * __bfloat162float(k_ptr[k_off + i]);
        partial = warp_reduce_sum(partial) * scale;

        if (use_mask && !mask_ptr[mask_off + s]) partial = -FLT_MAX;

        float new_m = fmaxf(m, partial);
        float alpha = expf(m - new_m);
        float beta  = expf(partial - new_m);
        d = d * alpha + beta;

        int v_off = kv_base + s * hd + lane * hd_per_thread;
        #pragma unroll
        for (int i = 0; i < hd_per_thread; i++)
            acc_reg[i] = acc_reg[i] * alpha + __bfloat162float(v_ptr[v_off + i]) * beta;
        m = new_m;
    }

    int out_off = ((batch * n_heads + q_head) * q_len + pos) * hd + lane * hd_per_thread;
    #pragma unroll
    for (int i = 0; i < hd_per_thread; i++)
        out_ptr[out_off + i] = __float2bfloat16(acc_reg[i] / d);
}

torch::Tensor gqa_prefill_attn(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    bool is_causal = false, int64_t causal_offset = 0,
    c10::optional<double> scale = c10::nullopt
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(k.dtype() == torch::kBFloat16);
    TORCH_CHECK(v.dtype() == torch::kBFloat16);

    int B = q.size(0), n_heads = q.size(1), q_len = q.size(2), hd = q.size(3);
    int n_kv = k.size(1), kv_len = k.size(2);
    TORCH_CHECK(hd % 32 == 0, "head_dim must be multiple of 32");

    bool use_mask = mask.has_value();
    const bool* mask_ptr = nullptr;
    if (use_mask) {
        TORCH_CHECK(mask.value().dtype() == torch::kBool);
        mask_ptr = mask.value().data_ptr<bool>();
    }

    auto out = torch::empty_like(q);

    dim3 block(32);
    dim3 grid(B * n_heads * q_len);

    gqa_prefill_attn_kernel<<<grid, block>>>(
        reinterpret_cast<const bf16*>(q.data_ptr()),
        reinterpret_cast<const bf16*>(k.data_ptr()),
        reinterpret_cast<const bf16*>(v.data_ptr()),
        mask_ptr,
        reinterpret_cast<bf16*>(out.data_ptr()),
        B, n_heads, n_kv, q_len, kv_len, hd,
        (int)use_mask, (int)is_causal, (int)causal_offset
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gqa_prefill_attn", &gqa_prefill_attn, "GQA prefill attention (naive)");
}
