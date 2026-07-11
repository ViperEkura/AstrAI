#include "attn_paged_decode_split_kv.cuh"
#ifndef ASTRAI_NO_MMA
#include "attn_paged_decode_split_kv_mma.cuh"
#endif

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

static int paged_decode_num_splits(int base_blocks, int tiles_total) {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int n = (2 * sm_count + base_blocks - 1) / base_blocks;
    return std::max(1, std::min(n, std::min(tiles_total, 32)));
}

static void launch_paged_scalar_decode(PagedAttentionParams<bf16>& p) {
    int group_size = p.q_head / p.kv_head;
    int chunks_total = (p.kv_len + PDC_CHUNK - 1) / PDC_CHUNK;
    p.num_splits = paged_decode_num_splits(p.batch * p.kv_head, chunks_total);

    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto o_part = torch::empty({p.batch, p.q_head, p.num_splits, p.head_dim}, fopt);
    auto ml_part = torch::empty({p.batch, p.q_head, p.num_splits, 2}, fopt);
    p.o_part = o_part.data_ptr<float>();
    p.ml_part = ml_part.data_ptr<float>();

    size_t smem = PDC_CHUNK * p.head_dim * sizeof(bf16);
    paged_attn_decode_split_kv_kernel<<<
        dim3(p.batch * p.kv_head, 1, p.num_splits),
        dim3(32, group_size),
        smem>>>(p);
    paged_attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}

#ifndef ASTRAI_NO_MMA
template <int HEAD_DIM, int BC>
static void launch_paged_mma_decode(PagedAttentionParams<bf16>& p) {
    int tiles_total = (p.kv_len + BC - 1) / BC;
    p.num_splits = paged_decode_num_splits(p.batch * p.kv_head, tiles_total);

    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto o_part = torch::empty({p.batch, p.q_head, p.num_splits, p.head_dim}, fopt);
    auto ml_part = torch::empty({p.batch, p.q_head, p.num_splits, 2}, fopt);
    p.o_part = o_part.data_ptr<float>();
    p.ml_part = ml_part.data_ptr<float>();

    paged_attn_decode_split_kv_mma_kernel<HEAD_DIM, BC>
        <<<dim3(p.kv_head, p.batch, p.num_splits), 32>>>(p);
    paged_attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}
#endif

template <int HEAD_DIM>
static void dispatch_paged_decode(PagedAttentionParams<bf16>& p) {
#ifndef ASTRAI_NO_MMA
    int G = p.q_head / p.kv_head;
    if (!p.use_mask && G >= 1 && G <= 16 && p.page_size >= 32) {
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
    bool is_causal = false,
    int64_t causal_offset = 0,
    c10::optional<double> scale = c10::nullopt
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    int batch = q.size(0);
    int q_head = q.size(1);
    int head_dim = q.size(3);
    int kv_head = k_cache.size(2);
    int max_pages = page_table.size(1);

    TORCH_CHECK(q.is_cuda() && page_table.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k_cache.dtype() == torch::kBFloat16, "k_cache must be bf16");
    TORCH_CHECK(v_cache.dtype() == torch::kBFloat16, "v_cache must be bf16");
    TORCH_CHECK(page_table.dtype() == torch::kLong, "page_table must be int64");
    TORCH_CHECK(q.size(2) == 1, "Q seq_len must be 1 (decode)");
    TORCH_CHECK(head_dim % 32 == 0, "head_dim must be multiple of 32");
    TORCH_CHECK(k_cache.size(1) == page_size,
                "k_cache dim 1 must equal page_size, got ",
                k_cache.size(1), " vs ", page_size);
    TORCH_CHECK(k_cache.size(0) >= 0, "k_cache must have at least 0 pages");

    float scale_val = scale.has_value()
        ? static_cast<float>(scale.value())
        : 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto O = torch::empty_like(q);

    PagedAttentionParams<bf16, float> p;
    p.batch = batch;
    p.q_head = q_head;
    p.kv_head = kv_head;
    p.q_len = 1;
    p.kv_len = static_cast<int>(kv_len);
    p.head_dim = head_dim;
    p.use_mask = (mask.has_value() && mask.value().defined()) ? 1 : 0;
    p.is_causal = is_causal ? 1 : 0;
    p.causal_offset = static_cast<int>(causal_offset);
    p.num_splits = 1;
    p.scale = scale_val;
    p.page_size = static_cast<int>(page_size);
    p.max_pages = max_pages;
    p.page_table = page_table.data_ptr<int64_t>();
    p.k_cache = reinterpret_cast<const bf16*>(k_cache.data_ptr());
    p.v_cache = reinterpret_cast<const bf16*>(v_cache.data_ptr());
    p.q = reinterpret_cast<const bf16*>(q.data_ptr());
    p.mask = p.use_mask ? mask.value().data_ptr<bool>() : nullptr;
    p.o = reinterpret_cast<bf16*>(O.data_ptr());
    p.o_part = nullptr;
    p.ml_part = nullptr;

    switch (p.head_dim) {
        case 32:  dispatch_paged_decode<32>(p);  break;
        case 64:  dispatch_paged_decode<64>(p);  break;
        case 128: dispatch_paged_decode<128>(p); break;
        case 256: dispatch_paged_decode<256>(p); break;
        default:
            TORCH_CHECK(false, "paged_decode: unsupported head_dim ", p.head_dim,
                        " (supported: 32, 64, 128, 256)");
    }

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
        py::arg("is_causal") = false,
        py::arg("causal_offset") = 0,
        py::arg("scale") = py::none(),
        "Paged GQA decode — split-KV with direct page-table access.");
}
