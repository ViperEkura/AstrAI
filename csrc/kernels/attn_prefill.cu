#include "attn_prefill_split_q.cuh"
#include "attn_entry_utils.cuh"

#ifndef ASTRAI_NO_MMA
#include "attn_prefill_split_q_mma.cuh"

template <int HEAD_DIM, bool IsCausal, bool HasMask>
static void launch_mma_prefill(AttentionParams<bf16>& p) {
    constexpr int WARPS = 4;
    constexpr int BC = (HEAD_DIM <= 128) ? 32 : 16;
    using Traits = KernelTraits<HEAD_DIM, BC, WARPS, 2>;
    dim3 grid((p.q_len + Traits::BR * WARPS - 1) / (Traits::BR * WARPS),
              p.q_head, p.batch);
    dim3 block(Traits::NUM_THREADS, 1, 1);
    attn_prefill_split_q_mma_kernel<Traits, IsCausal, HasMask><<<grid, block>>>(p);
}
#endif

template <int HEAD_DIM, bool IsCausal, bool HasMask>
static void launch_scalar_prefill(AttentionParams<bf16>& p) {
    constexpr int G = 8, ROWS = 32, P_BC = 32;
    dim3 grid((p.q_len + ROWS - 1) / ROWS, p.q_head, p.batch);
    dim3 block(G, ROWS, 1);
    attn_prefill_split_q_kernel_t<HEAD_DIM, G, ROWS, P_BC, IsCausal, HasMask><<<grid, block>>>(p);
}

template <int HEAD_DIM>
static void dispatch_prefill(AttentionParams<bf16>& p) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);

#ifndef ASTRAI_NO_MMA
    if (is_causal) {
        if (has_mask)      launch_mma_prefill<HEAD_DIM, true, true>(p);
        else               launch_mma_prefill<HEAD_DIM, true, false>(p);
    } else {
        if (has_mask)      launch_mma_prefill<HEAD_DIM, false, true>(p);
        else               launch_mma_prefill<HEAD_DIM, false, false>(p);
    }
#else
    if (is_causal) {
        if (has_mask)      launch_scalar_prefill<HEAD_DIM, true, true>(p);
        else               launch_scalar_prefill<HEAD_DIM, true, false>(p);
    } else {
        if (has_mask)      launch_scalar_prefill<HEAD_DIM, false, true>(p);
        else               launch_scalar_prefill<HEAD_DIM, false, false>(p);
    }
#endif
}

torch::Tensor attn_prefill(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    int64_t layout
) {
    AttentionParams<bf16> p;
    attn_pack_params(q, k, v, mask, causal_offset, scale, layout, p);
    TORCH_CHECK(p.head_dim % 16 == 0, "head_dim must be multiple of 16");

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == 1) ? O.transpose(1, 2) : O;
    p.o = (bf16*)O_view.data_ptr();

    DISPATCH_HEAD_DIM(p.head_dim, dispatch_prefill, p);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_prefill", &attn_prefill,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        py::arg("layout") = 0,
        "GQA prefill (tensor-core mma on sm_80+, scalar fallback)");
}
