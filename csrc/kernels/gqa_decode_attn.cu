#include "gqa_decode_attn.cuh"
#include <torch/extension.h>

#ifndef ASTRAI_NO_MMA
#include "gqa_decode_attn_mma.cuh"
#endif

// Scalar fallback: one warp per query head, per (batch, kv_head) block.
static void launch_scalar_decode(const GQAParams& p) {
    int group_size = p.q_head / p.kv_head;
    size_t smem = DC_CHUNK * p.head_dim * sizeof(bf16);
    gqa_decode_attn_kernel<<<p.batch * p.kv_head, dim3(32, group_size), smem>>>(p);
}

#ifndef ASTRAI_NO_MMA
// Tensor-core head-packing requires 1 < G <= 16 (the MMA M dim) and no mask.
static bool decode_use_mma(const GQAParams& p) {
    int G = p.q_head / p.kv_head;
    return !p.use_mask && G > 1 && G <= 16;
}

// Decode has only batch*kv_head independent tasks; without split-K the grid is
// tiny (e.g. 16 blocks) and leaves most SMs idle. Pick the smallest split count
// that fills the device (~2 blocks/SM), capped by the tile count and 32.
static int decode_num_splits(const GQAParams& p, int tiles_total) {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int base_blocks = p.kv_head * p.batch;
    int desired = 2 * (sm_count > 0 ? sm_count : 64);
    int n = (desired + base_blocks - 1) / base_blocks;
    return std::max(1, std::min(n, std::min(tiles_total, 32)));
}

template <int HEAD_DIM, int BC>
static void launch_mma_decode(GQAParams& p) {
    constexpr int BR = 16, LD = HEAD_DIM;  // XOR swizzle → no padding
    int smem = (2 * BC * LD + BR * LD) * (int)sizeof(bf16);
    int tiles_total = (p.kv_len + BC - 1) / BC;
    int num_splits = decode_num_splits(p, tiles_total);

    // Enough (batch, kv_head) work to fill the SMs → single pass, direct write.
    if (num_splits <= 1) {
        cudaFuncSetAttribute(gqa_decode_attn_mma_kernel<HEAD_DIM, BC>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        gqa_decode_attn_mma_kernel<HEAD_DIM, BC>
            <<<dim3(p.kv_head, p.batch), 32, smem>>>(p);
        return;
    }

    // Split-K (FlashDecoding): partition kv across blocks, then reduce.
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto o_part = torch::empty({p.batch, p.q_head, num_splits, p.head_dim}, fopt);
    auto ml_part = torch::empty({p.batch, p.q_head, num_splits, 2}, fopt);

    cudaFuncSetAttribute(gqa_decode_attn_mma_splitk_kernel<HEAD_DIM, BC>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    gqa_decode_attn_mma_splitk_kernel<HEAD_DIM, BC>
        <<<dim3(p.kv_head, p.batch, num_splits), 32, smem>>>(
            p, o_part.data_ptr<float>(), ml_part.data_ptr<float>(), num_splits);
    gqa_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(
        o_part.data_ptr<float>(), ml_part.data_ptr<float>(), p.o,
        num_splits, p.head_dim);
}
#endif

template <int HEAD_DIM>
static void dispatch_decode(GQAParams& p) {
#ifndef ASTRAI_NO_MMA
    if (decode_use_mma(p)) {
        launch_mma_decode<HEAD_DIM, 32>(p);
        return;
    }
#endif
    launch_scalar_decode(p);
}

torch::Tensor gqa_decode_attn(
    torch::Tensor q,
    torch::Tensor k, 
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    bool is_causal = false, 
    int64_t causal_offset = 0,
    c10::optional<double> scale = c10::nullopt
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(k.dtype() == torch::kBFloat16);
    TORCH_CHECK(v.dtype() == torch::kBFloat16);
    TORCH_CHECK(q.size(2) == 1, "Q seq_len must be 1");

    GQAParams p;
    p.batch = q.size(0);
    p.q_head = q.size(1);
    p.kv_head = k.size(1);
    p.q_len = 1;
    p.kv_len = k.size(2);
    p.head_dim = q.size(3);
    TORCH_CHECK(p.head_dim % 32 == 0, "head_dim must be multiple of 32");
    p.use_mask = mask.has_value();
    p.is_causal = (int)is_causal;
    p.causal_offset = (int)causal_offset;
    p.scale = scale.has_value() ? (float)scale.value() : 1.0f / sqrtf((float)p.head_dim);
    p.q = (const bf16*)q.data_ptr();
    p.k = (const bf16*)k.data_ptr();
    p.v = (const bf16*)v.data_ptr();
    if (p.use_mask) {
        TORCH_CHECK(mask.value().dtype() == torch::kBool);
        TORCH_CHECK(mask.value().dim() == 2);
        TORCH_CHECK(mask.value().size(0) == p.batch);
        TORCH_CHECK(mask.value().size(1) == p.kv_len);
        p.mask = mask.value().data_ptr<bool>();
    } else {
        p.mask = nullptr;
    }

    auto O = torch::empty_like(q);
    p.o = (bf16*)O.data_ptr();

    switch (p.head_dim) {
        case 32:
            dispatch_decode<32>(p);
            break;
        case 64:
            dispatch_decode<64>(p);
            break;
        case 128:
            dispatch_decode<128>(p);
            break;
        case 256:
            dispatch_decode<256>(p);
            break;
        default:
            TORCH_CHECK(false, "decode: unsupported head_dim ", p.head_dim,
                        " (supported: 32, 64, 128, 256)");
    }
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gqa_decode_attn", &gqa_decode_attn,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("is_causal") = false,
        py::arg("causal_offset") = 0,
        py::arg("scale") = py::none(),
        "GQA decode (tensor-core head-packing on sm_80+, scalar fallback)");
}
