#include "attn_decode_split_kv.cuh"
#include "attn_entry_utils.cuh"

#ifndef ASTRAI_NO_MMA
#include "attn_decode_split_kv_mma.cuh"
#endif

static int decode_num_splits(int base_blocks, int tiles_total) {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int n = (2 * sm_count + base_blocks - 1) / base_blocks;
    return std::max(1, std::min(n, std::min(tiles_total, 32)));
}

// Scalar fallback: one warp per query head, split-KV across grid.z.
static void launch_scalar_decode(AttentionParams<bf16>& p) {
    int group_size = p.q_head / p.kv_head;
    int chunks_total = (p.kv_len + DC_CHUNK - 1) / DC_CHUNK;
    p.num_splits = decode_num_splits(p.batch * p.kv_head, chunks_total);

    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto o_part = torch::empty({p.batch, p.q_head, p.num_splits, p.head_dim}, fopt);
    auto ml_part = torch::empty({p.batch, p.q_head, p.num_splits, 2}, fopt);
    p.o_part = o_part.data_ptr<float>();
    p.ml_part = ml_part.data_ptr<float>();

    size_t smem = DC_CHUNK * p.head_dim * sizeof(bf16);
    attn_decode_split_kv_kernel<<<dim3(p.batch * p.kv_head, 1, p.num_splits), dim3(32, group_size), smem>>>(p);
    attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}

#ifndef ASTRAI_NO_MMA
// MMA head-packing requires G <= 16 (BR=16 rows). sm_80+ tensor-core
// + cp.async wins even at G=1 (decode is memory-bound, not compute-bound).
// STAGES=2 (double-buffer) for D<=128 (smem 16 KB); STAGES=1 for D=256
// (double-buffer would be 32 KB, near the 48 KB static cap — keep single
// to preserve occupancy).
template <int HEAD_DIM, int BC, int STAGES = (HEAD_DIM <= 128) ? 2 : 1>
static void launch_mma_decode(AttentionParams<bf16>& p) {
    int tiles_total = (p.kv_len + BC - 1) / BC;
    p.num_splits = decode_num_splits(p.batch * p.kv_head, tiles_total);

    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto o_part = torch::empty({p.batch, p.q_head, p.num_splits, p.head_dim}, fopt);
    auto ml_part = torch::empty({p.batch, p.q_head, p.num_splits, 2}, fopt);
    p.o_part = o_part.data_ptr<float>();
    p.ml_part = ml_part.data_ptr<float>();

    attn_decode_split_kv_mma_kernel<HEAD_DIM, BC, STAGES><<<dim3(p.kv_head, p.batch, p.num_splits), 32>>>(p);
    attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}
#endif

template <int HEAD_DIM>
static void dispatch_decode(AttentionParams<bf16>& p) {
#ifndef ASTRAI_NO_MMA
    int G = p.q_head / p.kv_head;
    if (!p.use_mask && G >= 1 && G <= 16) {
        launch_mma_decode<HEAD_DIM, 32>(p);
        return;
    }
#endif
    launch_scalar_decode(p);
}

torch::Tensor attn_decode(
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
    TORCH_CHECK(p.q_len == 1, "Q seq_len must be 1");
    TORCH_CHECK(p.head_dim % 32 == 0, "head_dim must be multiple of 32");

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
    m.def("attn_decode", &attn_decode,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("is_causal") = false,
        py::arg("causal_offset") = 0,
        py::arg("scale") = py::none(),
        "GQA decode (tensor-core head-packing on sm_80+, scalar fallback)");
}
