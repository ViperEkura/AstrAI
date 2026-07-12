#include "attn_prefill_split_q.cuh"
#include "attn_entry_utils.cuh"

#ifndef ASTRAI_NO_MMA
#include "attn_prefill_split_q_mma.cuh"
#endif

template <int HEAD_DIM>
static void dispatch_prefill(AttentionParams<bf16>& p) {
#ifndef ASTRAI_NO_MMA
    constexpr int WARPS = 4, BR = 16;
    // KV tile: bigger tiles amortize the per-tile cp.async wait + barrier +
    // loop overhead over more tensor-core work (this kernel is latency-bound,
    // not compute/bandwidth-bound), so BC=32 wins ~6-8% over BC=16 for
    // D<=128. D=256 stays at 16: BC=32 double-buffered would need 64KB smem,
    // over the 48KB static cap. Both keep 3 blocks/SM (2 for D=256).
    constexpr int BC = (HEAD_DIM <= 128) ? 32 : 16;
    dim3 grid((p.q_len + BR * WARPS - 1) / (BR * WARPS), p.q_head, p.batch);
    dim3 block(WARPS * 32, 1, 1);
    // Static shared memory — no dynamic smem or cudaFuncSetAttribute needed.
    // sK[BC*LD] + sV[BC*LD] + sQ[BR*LD], all sized by template params.
    attn_prefill_split_q_mma_kernel<HEAD_DIM, WARPS, BC><<<grid, block>>>(p);
#else
    constexpr int G = 8, ROWS = 32, P_BC = 32;
    dim3 grid((p.q_len + ROWS - 1) / ROWS, p.q_head, p.batch);
    dim3 block(G, ROWS, 1);
    attn_prefill_split_q_kernel_t<HEAD_DIM, G, ROWS, P_BC><<<grid, block>>>(p);
#endif
}

torch::Tensor attn_prefill(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    bool is_causal = false,
    int64_t causal_offset = 0,
    c10::optional<double> scale = c10::nullopt
) {
    AttentionParams<bf16> p;
    attn_pack_params(q, k, v, mask, is_causal, causal_offset, scale, p);
    TORCH_CHECK(p.head_dim % 16 == 0, "head_dim must be multiple of 16");

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
    m.def("attn_prefill", &attn_prefill,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("is_causal") = false,
        py::arg("causal_offset") = 0,
        py::arg("scale") = py::none(),
        "GQA prefill (tensor-core mma on sm_80+, scalar fallback)");
}
