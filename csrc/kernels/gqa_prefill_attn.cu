#include "gqa_prefill_attn.cuh"
#include <torch/extension.h>

#ifndef ASTRAI_NO_MMA
#include "gqa_prefill_attn_mma.cuh"
#endif

template <int HEAD_DIM>
static void dispatch_prefill(GQAParams& p) {
#ifndef ASTRAI_NO_MMA
    constexpr int WARPS = 4, BC = 16, BR = 16;
    // Double-buffered K/V doubles smem, so BC is halved to 16 to keep 3+
    // blocks/SM. Register-hint MIN_BLOCKS tuned per HEAD_DIM's smem footprint.
    constexpr int MIN_BLOCKS = (HEAD_DIM <= 64) ? 6 : (HEAD_DIM <= 128) ? 3 : 2;
    dim3 grid((p.q_len + BR * WARPS - 1) / (BR * WARPS), p.q_head, p.batch);
    dim3 block(WARPS * 32, 1, 1);
    // Static shared memory — no dynamic smem or cudaFuncSetAttribute needed.
    // sK[BC*LD] + sV[BC*LD] + sQ[BR*LD], all sized by template params.
    gqa_prefill_attn_mma_kernel<HEAD_DIM, WARPS, BC, MIN_BLOCKS><<<grid, block>>>(p);
#else
    constexpr int G = 8, ROWS = 32, P_BC = 32;
    dim3 grid((p.q_len + ROWS - 1) / ROWS, p.q_head, p.batch);
    dim3 block(G, ROWS, 1);
    gqa_prefill_attn_kernel_t<HEAD_DIM, G, ROWS, P_BC><<<grid, block>>>(p);
#endif
}

torch::Tensor gqa_prefill_attn(
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

    GQAParams p;
    p.batch = q.size(0);
    p.q_head = q.size(1);
    p.kv_head = k.size(1);
    p.q_len = q.size(2);
    p.kv_len = k.size(2);
    p.head_dim = q.size(3);
    TORCH_CHECK(p.head_dim % 16 == 0, "head_dim must be multiple of 16");
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
            dispatch_prefill<32>(p);
            break;
        case 64:
            dispatch_prefill<64>(p);
            break;
        case 128:
            dispatch_prefill<128>(p);
            break;
        case 256:
            dispatch_prefill<256>(p);
            break;
        default:
            TORCH_CHECK(false, "prefill: unsupported head_dim ", p.head_dim,
                        " (supported: 32,64,128,256)");
    }
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gqa_prefill_attn", &gqa_prefill_attn,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("is_causal") = false,
        py::arg("causal_offset") = 0,
        py::arg("scale") = py::none(),
        "GQA prefill (tensor-core mma on sm_80+, scalar fallback)");
}
