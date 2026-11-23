#include "entry.h"
#include <api/torch/attention.h>
#include <launcher/attention.cuh>

namespace astrai {
namespace attention {

// Contiguous-KV parameter packing
static void pack_dense(torch::Tensor q,
                       torch::Tensor k,
                       torch::Tensor v,
                       c10::optional<torch::Tensor> mask,
                       c10::optional<double> scale,
                       AttentionParams& p,
                       bool is_causal) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    check_qkv_dtype(q, k, v);
    TORCH_CHECK(k.sizes() == v.sizes(), "K and V must have identical shapes");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "Q/K/V must be 4D");
    q = q.transpose(1, 2);
    check_int32_offset(q, "Q", 4);
    p.batch = static_cast<int>(q.size(0));
    p.q_head = static_cast<int>(q.size(1));
    p.q_len = static_cast<int>(q.size(2));
    p.head_dim = static_cast<int>(q.size(3));
    p.q_b_stride = static_cast<int>(q.stride(0));
    p.q_h_stride = static_cast<int>(q.stride(1));
    p.q_l_stride = static_cast<int>(q.stride(2));

    k = k.transpose(1, 2);
    v = v.transpose(1, 2);

    TORCH_CHECK(k.size(0) == p.batch, "K/V batch must match Q");
    check_int32_offset(k, "K/V", 4);
    p.kv_head = static_cast<int>(k.size(1));
    p.kv_len = static_cast<int>(k.size(2));
    TORCH_CHECK(p.kv_len > 0 && p.kv_len <= std::numeric_limits<int>::max() - 16,
                "K/V sequence length must fit attention tile arithmetic");

    TORCH_CHECK(p.kv_head > 0 && p.q_head > 0 && p.q_head % p.kv_head == 0,
                "q_head must be divisible by kv_head");
    TORCH_CHECK(k.size(3) == p.head_dim, "K/V head_dim must match Q");
    TORCH_CHECK(static_cast<int64_t>(p.q_len) * (p.q_head / p.kv_head) <=
                    std::numeric_limits<int>::max() - 1024,
                "packed query rows exceed int32 range");

    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
                "Q/K/V head_dim must be contiguous");

    p.kv_b_stride = static_cast<int>(k.stride(0));
    p.kv_h_stride = static_cast<int>(k.stride(1));
    p.kv_l_stride = static_cast<int>(k.stride(2));

    p.is_causal = is_causal;

    p.q_ptr = q.data_ptr();
    p.k_ptr = k.data_ptr();
    p.v_ptr = v.data_ptr();
    p.new_k_ptr = nullptr;
    p.new_v_ptr = nullptr;
    finish_pack(scale, p);

    pack_mask(mask, p, q.device());
}

torch::Tensor attention(torch::Tensor q,
                        torch::Tensor k,
                        torch::Tensor v,
                        c10::optional<torch::Tensor> mask,
                        c10::optional<double> scale,
                        bool is_causal,
                        int64_t kv_split_size) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    AttentionParams p{};
    match_kv_layout(k, v);
    pack_dense(q, k, v, mask, scale, p, is_causal);

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    p.o_ptr = O.data_ptr();
    set_kv_split_size(p, kv_split_size);

    if (q.scalar_type() == at::kBFloat16)
        launch_attention<bf16>(p, stream);
    else
        launch_attention<fp16>(p, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

} // namespace attention
} // namespace astrai
