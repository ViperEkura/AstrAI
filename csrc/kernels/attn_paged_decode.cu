#include "attn_paged_decode_split_kv.cuh"
#ifndef ASTRAI_NO_MMA
#include "attn_paged_decode_split_kv_mma.cuh"
#endif

#include "attn_entry_utils.cuh"

static void launch_paged_scalar_decode(PagedAttentionParams<bf16>& p) {
    int group_size = p.q_head / p.kv_head;
    int chunks_total = (p.kv_len + PDC_CHUNK - 1) / PDC_CHUNK;
    p.num_splits = compute_num_splits(p.batch * p.kv_head, chunks_total);
    alloc_split_partials(p);

    size_t smem = PDC_CHUNK * p.head_dim * sizeof(bf16);
    dim3 grid = dim3(p.batch * p.kv_head, 1, p.num_splits);
    dim3 block = dim3(32, group_size);
    paged_attn_decode_split_kv_kernel<<<grid, block, smem>>>(p);
    paged_attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}

#ifndef ASTRAI_NO_MMA
template <int HEAD_DIM, int BC, int STAGES = (HEAD_DIM <= 128) ? 2 : 1>
static void launch_paged_mma_decode(PagedAttentionParams<bf16>& p) {
    int tiles_total = (p.kv_len + BC - 1) / BC;
    p.num_splits = compute_num_splits(p.batch * p.kv_head, tiles_total);
    alloc_split_partials(p);

    paged_attn_decode_split_kv_mma_kernel<HEAD_DIM, BC, STAGES><<<dim3(p.kv_head, p.batch, p.num_splits), 32>>>(p);
    paged_attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}
#endif

template <int HEAD_DIM>
static void dispatch_paged_decode(PagedAttentionParams<bf16>& p) {
#ifndef ASTRAI_NO_MMA
    int G = p.q_head / p.kv_head;
    if (G >= 1 && G <= 16 && p.page_size >= 32) {
        launch_paged_mma_decode<HEAD_DIM, 32>(p);
        return;
    }
#endif
    launch_paged_scalar_decode(p);
}

torch::Tensor attn_paged_decode(
    torch::Tensor q,
    torch::Tensor page_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t page_size,
    int64_t kv_len,
    c10::optional<torch::Tensor> mask,
    int64_t causal_offset,
    double scale,
    int64_t layout
) {
    PagedAttentionParams<bf16> p;
    attn_pack_paged_params(q, page_table, k_cache, v_cache,
                           page_size, kv_len, mask, causal_offset, scale, layout, p);

    auto O = torch::empty_strided(q.sizes(), q.strides(), q.options());
    auto O_view = (layout == 1) ? O.transpose(1, 2) : O;
    p.o = (bf16*)O_view.data_ptr();

    DISPATCH_HEAD_DIM(p.head_dim, dispatch_paged_decode, p);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_paged_decode", &attn_paged_decode,
        py::arg("q"),
        py::arg("page_table"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("page_size"),
        py::arg("kv_len"),
        py::arg("mask") = py::none(),
        py::arg("causal_offset") = -1,
        py::arg("scale") = 0.0,
        py::arg("layout") = 0,
        "Paged GQA decode — split-KV with direct page-table access.");
}
