#include "attn_decode_split_kv.cuh"
#include "attn_entry_utils.cuh"

#ifndef ASTRAI_NO_MMA
#include "attn_decode_split_kv_mma.cuh"

template <int HEAD_DIM, int BC, int STAGES, bool IsCausal, bool HasMask>
static void launch_mma_decode_impl(AttentionParams<bf16>& p) {
    using Traits = KernelTraits<HEAD_DIM, BC, 1, STAGES>;
    int tiles_total = (p.kv_len + BC - 1) / BC;
    p.num_splits = compute_num_splits(p.batch * p.kv_head, tiles_total);
    alloc_split_partials(p);

    attn_decode_split_kv_mma_kernel<Traits, IsCausal, HasMask><<<dim3(p.kv_head, p.batch, p.num_splits), 32>>>(p);
    attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}

template <int HEAD_DIM, int BC, bool IsCausal, bool HasMask>
static void launch_mma_decode(AttentionParams<bf16>& p) {
    constexpr int STAGES = (HEAD_DIM <= 128) ? 2 : 1;
    launch_mma_decode_impl<HEAD_DIM, BC, STAGES, IsCausal, HasMask>(p);
}
#endif

template <int HEAD_DIM, bool IsCausal, bool HasMask>
static void launch_scalar_decode(AttentionParams<bf16>& p) {
    int group_size = p.q_head / p.kv_head;
    int chunks_total = (p.kv_len + DC_CHUNK - 1) / DC_CHUNK;
    p.num_splits = compute_num_splits(p.batch * p.kv_head, chunks_total);
    alloc_split_partials(p);

    size_t smem = DC_CHUNK * p.head_dim * sizeof(bf16);
    
    dim3 grid(p.batch * p.kv_head, 1, p.num_splits); 
    dim3 block(32, group_size); 
    attn_decode_split_kv_kernel<HEAD_DIM, IsCausal, HasMask><<<grid, block, smem>>>(p);
    attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}

template <int HEAD_DIM>
static void dispatch_decode(AttentionParams<bf16>& p) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);

#ifndef ASTRAI_NO_MMA
    int G = p.q_head / p.kv_head;
    if (G >= 1 && G <= 16) {
        if (is_causal) {
            if (has_mask)      launch_mma_decode<HEAD_DIM, 32, true, true>(p);
            else               launch_mma_decode<HEAD_DIM, 32, true, false>(p);
        } else {
            if (has_mask)      launch_mma_decode<HEAD_DIM, 32, false, true>(p);
            else               launch_mma_decode<HEAD_DIM, 32, false, false>(p);
        }
        return;
    }
#endif
    if (is_causal) {
        if (has_mask)      launch_scalar_decode<HEAD_DIM, true, true>(p);
        else               launch_scalar_decode<HEAD_DIM, true, false>(p);
    } else {
        if (has_mask)      launch_scalar_decode<HEAD_DIM, false, true>(p);
        else               launch_scalar_decode<HEAD_DIM, false, false>(p);
    }
}

torch::Tensor attn_decode(
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
    TORCH_CHECK(p.q_len == 1, "Q seq_len must be 1");
    TORCH_CHECK(p.head_dim % 32 == 0, "head_dim must be multiple of 32");

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == 1) ? O.transpose(1, 2) : O;
    p.o = (bf16*)O_view.data_ptr();

    DISPATCH_HEAD_DIM(p.head_dim, dispatch_decode, p);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_decode", &attn_decode,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        py::arg("layout") = 0,
        "GQA decode (tensor-core head-packing on sm_80+, scalar fallback)");
}
