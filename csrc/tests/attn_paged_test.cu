// Compile:
//   nvcc -I csrc/kernels -arch=sm_89 -O3 --use_fast_math --ptxas-options=-O3 \
//        --extra-device-vectorization -Xcompiler -fopenmp \
//        csrc/tests/attn_paged_test.cu \
//        -o /tmp/test_paged && /tmp/test_paged

#include <cstring>
#include <vector>
#include "test_utils.cuh"
#include "attention/dispatchers.cuh"

using namespace astrai::attention;

struct PagedDecodeDispatch { AttentionParams<bf16>& p; template<int H> void operator()() { dispatch_paged_decode<H>(p, 0); } };
struct PagedPrefillDispatch { AttentionParams<bf16>& p; template<int H> void operator()() { dispatch_paged_prefill<H>(p, 0); } };

static int make_q_tile_mapping(const std::vector<int>& q_lens,
                               int** d_batch, int** d_tile) {
    constexpr int ROWS = 64;
    std::vector<int> h_batch;
    std::vector<int> h_tile;
    for (int b = 0; b < (int)q_lens.size(); ++b) {
        int n_tiles = (q_lens[b] + ROWS - 1) / ROWS;
        for (int tile = 0; tile < n_tiles; ++tile) {
            h_batch.push_back(b);
            h_tile.push_back(tile);
        }
    }
    size_t bytes = h_batch.size() * sizeof(int);
    cudaMalloc(d_batch, bytes);
    cudaMalloc(d_tile, bytes);
    cudaMemcpy(*d_batch, h_batch.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_tile, h_tile.data(), bytes, cudaMemcpyHostToDevice);
    return (int)h_batch.size();
}

// ---- CPU reference: paged decode with variable seq_lens ----
// Q: [B, Hq, D], K/V pool: [pool_size, Hkv, D]
// req_to_token: [num_reqs, max_ctx_len], req_pool_indices: [B]
// kv_indptr: [B+1].  mask: [B, max_seq_len] bool (True=keep) or NULL.
static void cpu_paged_decode_ref(
    const float* Q, const float* K_pool, const float* V_pool,
    const int* req_to_token, const int* req_pool_indices,
    const int* kv_indptr, const bool* mask, int mask_b_stride,
    int B, int Hq, int Hkv, int D, int max_ctx_len,
    float* O)
{
    float scale = 1.0f / sqrtf((float)D);
    int n_rep = Hq / Hkv;
    for (int b = 0; b < B; b++) {
        int seq_len = kv_indptr[b + 1] - kv_indptr[b];
        int req_idx = req_pool_indices[b];
        #pragma omp parallel for schedule(dynamic)
        for (int h = 0; h < Hq; h++) {
            int kv_h = h / n_rep;
            float mv = -INFINITY, sv = 0.0f;
            float accum[256] = {0.0f};
            for (int kj = 0; kj < seq_len; kj++) {
                if (mask && !mask[b * mask_b_stride + kj]) continue;
                int slot = req_to_token[req_idx * max_ctx_len + kj];
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += Q[(b * Hq + h) * D + d] *
                           K_pool[slot * Hkv * D + kv_h * D + d];
                dot *= scale;
                float nm = fmaxf(mv, dot);
                float a = expf(mv - nm);
                float be = expf(dot - nm);
                sv = sv * a + be;
                for (int d = 0; d < D; d++)
                    accum[d] = accum[d] * a +
                               V_pool[slot * Hkv * D + kv_h * D + d] * be;
                mv = nm;
            }
            float inv = 1.0f / sv;
            for (int d = 0; d < D; d++)
                O[(b * Hq + h) * D + d] = accum[d] * inv;
        }
    }
}

// ---- CPU reference: paged prefill with ragged batch ----
// Q: [total_q, Hq, D], K/V pool: [pool_size, Hkv, D]
// req_to_token: [num_reqs, max_ctx_len], req_pool_indices: [B]
// kv_indptr: [B+1], qo_indptr: [B+1].
// mask: [B, max_q_len, max_seq_len] bool (True=keep, q-local + kv-local
//   positions) or NULL.  Used only when causal==0 to apply an arbitrary
//   attention mask on top of the (unused) causal logic.
static void cpu_paged_prefill_ref(
    const float* Q, const float* K_pool, const float* V_pool,
    const int* req_to_token, const int* req_pool_indices,
    const int* kv_indptr, const int* qo_indptr,
    const bool* mask, int mask_l_stride, int mask_kv_stride,
    int B, int Hq, int Hkv, int D, int max_ctx_len, int causal,
    float* O)
{
    float scale = 1.0f / sqrtf((float)D);
    int n_rep = Hq / Hkv;
    for (int b = 0; b < B; b++) {
        int seq_len = kv_indptr[b + 1] - kv_indptr[b];
        int q_len = qo_indptr[b + 1] - qo_indptr[b];
        int causal_off = seq_len - q_len;
        int req_idx = req_pool_indices[b];
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int h = 0; h < Hq; h++) {
            for (int qi = 0; qi < q_len; qi++) {
                int kv_h = h / n_rep;
                float mv = -INFINITY, sv = 0.0f;
                float accum[256] = {0.0f};
                int lim = causal ? min(seq_len, causal_off + qi + 1) : seq_len;
                for (int kj = 0; kj < lim; kj++) {
                    if (mask && !mask[b * mask_l_stride * mask_kv_stride
                                     + qi * mask_kv_stride + kj]) continue;
                    int slot = req_to_token[req_idx * max_ctx_len + kj];
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++)
                        dot += Q[(qo_indptr[b] + qi) * Hq * D + h * D + d] *
                               K_pool[slot * Hkv * D + kv_h * D + d];
                    dot *= scale;
                    float nm = fmaxf(mv, dot);
                    float a = expf(mv - nm);
                    float be = expf(dot - nm);
                    sv = sv * a + be;
                    for (int d = 0; d < D; d++)
                        accum[d] = accum[d] * a +
                                   V_pool[slot * Hkv * D + kv_h * D + d] * be;
                    mv = nm;
                }
                float inv = 1.0f / sv;
                for (int d = 0; d < D; d++)
                    O[(qo_indptr[b] + qi) * Hq * D + h * D + d] = accum[d] * inv;
            }
        }
    }
}

// ---- paged validation table (kernel vs CPU ref, abs error only) ----
inline void print_paged_header() {
    printf("%-58s | %11s | %6s\n",
           "config", "max_err", "result");
    printf("----------------------------------------------------------------"
           "--------------------------------\n");
}

inline void print_paged_row(const char* cfg, float max_err, bool pass) {
    printf("%-58s | %11.3e | %s\n",
           cfg, max_err, pass ? "PASS" : "FAIL");
}

// float mirror of a bf16 buffer, for the CPU references.
static float* to_floats(const bf16* src, size_t n) {
    float* dst = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) dst[i] = bf2f(src[i]);
    return dst;
}

// ======================================================================
// Shared paged test rig: owns the flat KV pool, request table, index
// buffers and (optionally) the split partials / mask, in host mirrors +
// device buffers; fills and uploads them, and frees everything on
// destruction.  Q is [total_q, Hq, D] — decode passes one row per request
// (total_q == B), ragged prefill the packed per-request rows.
// ======================================================================
struct PagedRig {
    int B, Hq, Hkv, D;
    int total_q = 0, max_sl = 0, max_ctx = 0, pool_size = 0, num_reqs = 0;
    std::vector<int> q_lens, kv_lens;
    size_t q_elems = 0, kv_elems = 0, mask_elems = 0;

    bf16 *h_q = nullptr, *h_k = nullptr, *h_v = nullptr;
    bool *h_mask = nullptr;
    std::vector<int> h_rtt, h_rpi, h_kvi, h_qoi;

    bf16 *d_q, *d_o, *d_k, *d_v;
    int *d_rtt, *d_rpi, *d_kvi, *d_qoi = nullptr;
    bool *d_mask = nullptr;
    float *d_op = nullptr, *d_ml = nullptr;

    // Zero-value overrides pick the test defaults: ctx = max_sl + 16,
    // pool = B * ctx, reqs = B + 4.  ragged allocates qo_indptr;
    // split_partials the decode o_part/ml_part buffers.
    PagedRig(int B_, int Hq_, int Hkv_, int D_, std::vector<int> ql,
             std::vector<int> kl, bool ragged, bool split_partials,
             int ctx_capacity = 0, int pool_override = 0, int reqs = 0)
        : B(B_), Hq(Hq_), Hkv(Hkv_), D(D_),
          q_lens(std::move(ql)), kv_lens(std::move(kl)) {
        for (int b = 0; b < B; b++) {
            total_q += q_lens[b];
            max_sl = max(max_sl, kv_lens[b]);
        }
        max_ctx = ctx_capacity ? ctx_capacity : max_sl + 16;
        pool_size = pool_override ? pool_override : B * max_ctx;
        num_reqs = reqs ? reqs : B + 4;

        q_elems = (size_t)total_q * Hq * D;
        kv_elems = (size_t)pool_size * Hkv * D;
        h_q = (bf16*)malloc(q_elems * sizeof(bf16));
        h_k = (bf16*)malloc(kv_elems * sizeof(bf16));
        h_v = (bf16*)malloc(kv_elems * sizeof(bf16));
        CUDA_CHECK(cudaMalloc(&d_q, q_elems * sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&d_o, q_elems * sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&d_k, kv_elems * sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&d_v, kv_elems * sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&d_rtt, (size_t)num_reqs * max_ctx * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_rpi, (size_t)B * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_kvi, (size_t)(B + 1) * sizeof(int)));
        if (ragged)
            CUDA_CHECK(cudaMalloc(&d_qoi, (B + 1) * sizeof(int)));
        if (split_partials) {
            CUDA_CHECK(cudaMalloc(&d_op, (size_t)B * Hq * MAX_SPLITS * D * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_ml, (size_t)B * Hq * MAX_SPLITS * 2 * sizeof(float)));
        }
    }

    ~PagedRig() {
        free(h_q); free(h_k); free(h_v); free(h_mask);
        cudaFree(d_q); cudaFree(d_o); cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_rtt); cudaFree(d_rpi); cudaFree(d_kvi);
        if (d_qoi) cudaFree(d_qoi);
        if (d_mask) cudaFree(d_mask);
        if (d_op) cudaFree(d_op);
        if (d_ml) cudaFree(d_ml);
    }

    PagedRig(const PagedRig&) = delete;
    PagedRig& operator=(const PagedRig&) = delete;

    // Random q/k/v fill + upload (kernel and CPU ref share the data).
    void fill_data() {
        auto rnd = [] { return (rand() / (float)RAND_MAX) * 2.0f - 1.0f; };
        for (size_t i = 0; i < q_elems; i++) h_q[i] = f2bf(rnd());
        for (size_t i = 0; i < kv_elems; i++) {
            h_k[i] = f2bf(rnd());
            h_v[i] = f2bf(rnd());
        }
        cudaMemcpy(d_q, h_q, q_elems * sizeof(bf16), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, h_k, kv_elems * sizeof(bf16), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_v, kv_elems * sizeof(bf16), cudaMemcpyHostToDevice);
    }

    // Scattered request table (unique slots wrapping the pool), identity
    // pool indices, kv prefix sums (+ qo prefix sums when ragged).
    void fill_indices() {
        h_rtt.resize((size_t)num_reqs * max_ctx);
        int next_slot = 0;
        for (int r = 0; r < num_reqs; r++)
            for (int p = 0; p < max_ctx; p++)
                h_rtt[r * max_ctx + p] = next_slot++ % pool_size;
        h_rpi.resize(B);
        for (int b = 0; b < B; b++) h_rpi[b] = b;
        h_kvi.assign(B + 1, 0);
        for (int b = 0; b < B; b++) h_kvi[b + 1] = h_kvi[b] + kv_lens[b];
        cudaMemcpy(d_rtt, h_rtt.data(), h_rtt.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rpi, h_rpi.data(), h_rpi.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kvi, h_kvi.data(), h_kvi.size() * sizeof(int), cudaMemcpyHostToDevice);
        if (d_qoi) {
            h_qoi.assign(B + 1, 0);
            for (int b = 0; b < B; b++) h_qoi[b + 1] = h_qoi[b] + q_lens[b];
            cudaMemcpy(d_qoi, h_qoi.data(), h_qoi.size() * sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    // Mask storage [rows × cols]; the test fills h_mask then uploads.
    void alloc_mask(int rows, int cols) {
        mask_elems = (size_t)rows * cols;
        h_mask = (bool*)malloc(mask_elems);
        CUDA_CHECK(cudaMalloc(&d_mask, mask_elems));
    }
    void upload_mask() {
        cudaMemcpy(d_mask, h_mask, mask_elems, cudaMemcpyHostToDevice);
    }

    // Common launch parameters; callers tweak causal/mask/q-tile fields.
    AttentionParams<bf16> base_params() {
        AttentionParams<bf16> p = {};
        p.batch = B; p.q_head = Hq; p.kv_head = Hkv;
        p.head_dim = D; p.q_len = total_q;
        p.q_l_stride = Hq * D; p.q_h_stride = D; p.q_d_stride = 1;
        p.max_context_len = max_ctx;
        p.scale = 1.0f / sqrtf((float)D);
        p.q_ptr = d_q; p.k_ptr = d_k; p.v_ptr = d_v;
        p.req_to_token = d_rtt; p.req_pool_indices = d_rpi;
        p.kv_indptr = d_kvi; p.qo_indptr = d_qoi;
        p.o_ptr = d_o; p.o_part = d_op; p.ml_part = d_ml;
        return p;
    }

    // Download O, compare against the CPU ref, print one table row.
    int check(const char* cfg, const float* ref) {
        bf16* h_o = (bf16*)malloc(q_elems * sizeof(bf16));
        cudaMemcpy(h_o, d_o, q_elems * sizeof(bf16), cudaMemcpyDeviceToHost);
        const float atol = 0.01f, rtol = 0.01f;
        bool pass = true;
        float max_err = 0.0f;
        for (size_t i = 0; i < q_elems; i++) {
            float e = fabsf(bf2f(h_o[i]) - ref[i]);
            if (e > max_err) max_err = e;
            if (e > atol + rtol * fabsf(ref[i])) { pass = false; break; }
        }
        print_paged_row(cfg, max_err, pass);
        free(h_o);
        return pass ? 0 : 1;
    }
};

// ======================================================================
// DECODE TEST
// ======================================================================
template <int HEAD_DIM>
static int run_decode_test(int B, int Hq, int Hkv, int max_seq,
                            int causal, int seed, int context_capacity = 0,
                            int fixed_seq_len = 0) {
    srand(seed);
    std::vector<int> seq_lens(B);
    for (int b = 0; b < B; b++)
        seq_lens[b] = fixed_seq_len ? fixed_seq_len : 8 + rand() % (max_seq - 8);

    PagedRig rig(B, Hq, Hkv, HEAD_DIM, std::vector<int>(B, 1), seq_lens,
                 /*ragged=*/false, /*split_partials=*/true, context_capacity);
    rig.fill_data();
    rig.fill_indices();

    char cfg[80];
    snprintf(cfg, sizeof(cfg), "DECODE B=%d Hq=%d Hkv=%d D=%d max_sl=%d causal=%d",
             B, Hq, Hkv, HEAD_DIM, rig.max_sl, causal);

    float* qf = to_floats(rig.h_q, rig.q_elems);
    float* kf = to_floats(rig.h_k, rig.kv_elems);
    float* vf = to_floats(rig.h_v, rig.kv_elems);
    float* ref = (float*)calloc(rig.q_elems, sizeof(float));
    cpu_paged_decode_ref(qf, kf, vf, rig.h_rtt.data(), rig.h_rpi.data(),
                         rig.h_kvi.data(), nullptr, 0,
                         B, Hq, Hkv, HEAD_DIM, rig.max_ctx, ref);

    AttentionParams<bf16> p = rig.base_params();
    p.causal_offset = causal ? 0 : -1;
    dispatch_by_head_dim(HEAD_DIM, PagedDecodeDispatch{p});
    cudaDeviceSynchronize();

    int fail = rig.check(cfg, ref);
    free(qf); free(kf); free(vf); free(ref);
    return fail;
}

// ======================================================================
// DECODE WITH MASK TEST (regression: 2D mask on mixed seq_lens)
// ======================================================================
template <int HEAD_DIM>
static int run_decode_mask_test(int B, int Hq, int Hkv, int max_seq,
                                int seed) {
    srand(seed);
    std::vector<int> seq_lens(B);
    for (int b = 0; b < B; b++)
        seq_lens[b] = 8 + rand() % (max_seq - 8);

    PagedRig rig(B, Hq, Hkv, HEAD_DIM, std::vector<int>(B, 1), seq_lens,
                 /*ragged=*/false, /*split_partials=*/true);
    rig.fill_data();
    rig.fill_indices();
    rig.alloc_mask(B, rig.max_sl);
    // Keep the even positions of each request's kv range, drop the rest —
    // exercises the HasMask path with per-request seq_len.
    for (int b = 0; b < B; b++)
        for (int k = 0; k < rig.max_sl; k++)
            rig.h_mask[b * rig.max_sl + k] = (k < seq_lens[b]) && (k % 2 == 0);
    rig.upload_mask();

    char cfg[80];
    snprintf(cfg, sizeof(cfg), "DECODE-MASK B=%d Hq=%d Hkv=%d D=%d max_sl=%d",
             B, Hq, Hkv, HEAD_DIM, rig.max_sl);

    float* qf = to_floats(rig.h_q, rig.q_elems);
    float* kf = to_floats(rig.h_k, rig.kv_elems);
    float* vf = to_floats(rig.h_v, rig.kv_elems);
    float* ref = (float*)calloc(rig.q_elems, sizeof(float));
    cpu_paged_decode_ref(qf, kf, vf, rig.h_rtt.data(), rig.h_rpi.data(),
                         rig.h_kvi.data(), rig.h_mask, rig.max_sl,
                         B, Hq, Hkv, HEAD_DIM, rig.max_ctx, ref);

    AttentionParams<bf16> p = rig.base_params();
    p.causal_offset = -1; p.use_mask = 1;
    p.mask = rig.d_mask; p.mask_b_stride = rig.max_sl;
    dispatch_by_head_dim(HEAD_DIM, PagedDecodeDispatch{p});
    cudaDeviceSynchronize();

    int fail = rig.check(cfg, ref);
    free(qf); free(kf); free(vf); free(ref);
    return fail;
}

// ======================================================================
// PREFILL TEST
// ======================================================================
template <int HEAD_DIM>
static int run_prefill_test(int B, int Hq, int Hkv,
                             std::vector<int>& q_lens,
                             std::vector<int>& kv_lens,
                             int causal, int seed) {
    PagedRig rig(B, Hq, Hkv, HEAD_DIM, q_lens, kv_lens,
                 /*ragged=*/true, /*split_partials=*/false);
    srand(seed);
    rig.fill_data();
    rig.fill_indices();

    char cfg[80];
    snprintf(cfg, sizeof(cfg), "PREFILL B=%d Hq=%d Hkv=%d D=%d max_sl=%d causal=%d",
             B, Hq, Hkv, HEAD_DIM, rig.max_sl, causal);

    float* qf = to_floats(rig.h_q, rig.q_elems);
    float* kf = to_floats(rig.h_k, rig.kv_elems);
    float* vf = to_floats(rig.h_v, rig.kv_elems);
    float* ref = (float*)calloc(rig.q_elems, sizeof(float));
    cpu_paged_prefill_ref(qf, kf, vf, rig.h_rtt.data(), rig.h_rpi.data(),
                          rig.h_kvi.data(), rig.h_qoi.data(),
                          nullptr, 0, 0,
                          B, Hq, Hkv, HEAD_DIM, rig.max_ctx, causal, ref);

    int *d_qtb, *d_qti;
    int num_q_tiles = make_q_tile_mapping(q_lens, &d_qtb, &d_qti);

    AttentionParams<bf16> p = rig.base_params();
    p.causal_offset = causal ? 0 : -1;
    p.q_tile_to_batch = d_qtb; p.q_tile_to_index = d_qti;
    p.num_q_tiles = num_q_tiles;
    dispatch_by_head_dim(HEAD_DIM, PagedPrefillDispatch{p});
    cudaDeviceSynchronize();
    cudaFree(d_qtb); cudaFree(d_qti);

    int fail = rig.check(cfg, ref);
    free(qf); free(kf); free(vf); free(ref);
    return fail;
}

// ======================================================================
// PREFILL WITH MASK TEST (regression: 4D causal mask on single request)
// ======================================================================
template <int HEAD_DIM>
static int run_prefill_mask_test(int Hq, int Hkv, int q_len, int seed) {
    srand(seed);
    std::vector<int> ql = {q_len}, kl = {q_len};
    PagedRig rig(1, Hq, Hkv, HEAD_DIM, ql, kl,
                 /*ragged=*/true, /*split_partials=*/false);
    rig.fill_data();
    rig.fill_indices();
    rig.alloc_mask(q_len, q_len);
    // 4D causal mask [B, 1, q_len, q_len], True=keep.
    for (int qi = 0; qi < q_len; qi++)
        for (int kj = 0; kj < q_len; kj++)
            rig.h_mask[qi * q_len + kj] = (kj <= qi);
    rig.upload_mask();

    char cfg[80];
    snprintf(cfg, sizeof(cfg), "PREFILL-MASK Hq=%d Hkv=%d D=%d q_len=%d",
             Hq, Hkv, HEAD_DIM, q_len);
    fflush(stdout);

    float* qf = to_floats(rig.h_q, rig.q_elems);
    float* kf = to_floats(rig.h_k, rig.kv_elems);
    float* vf = to_floats(rig.h_v, rig.kv_elems);
    float* ref = (float*)calloc(rig.q_elems, sizeof(float));
    // CPU ref with causal=0 so it consults the mask (not the causal flag).
    cpu_paged_prefill_ref(qf, kf, vf, rig.h_rtt.data(), rig.h_rpi.data(),
                          rig.h_kvi.data(), rig.h_qoi.data(),
                          rig.h_mask, q_len, q_len,
                          1, Hq, Hkv, HEAD_DIM, rig.max_ctx, 0, ref);

    int *d_qtb, *d_qti;
    int num_q_tiles = make_q_tile_mapping(ql, &d_qtb, &d_qti);

    AttentionParams<bf16> p = rig.base_params();
    p.causal_offset = -1; p.use_mask = 1;
    p.mask = rig.d_mask; p.mask_b_stride = q_len * q_len;
    p.mask_l_stride = q_len;
    p.q_tile_to_batch = d_qtb; p.q_tile_to_index = d_qti;
    p.num_q_tiles = num_q_tiles;
    dispatch_by_head_dim(HEAD_DIM, PagedPrefillDispatch{p});
    cudaDeviceSynchronize();
    cudaFree(d_qtb); cudaFree(d_qti);

    int fail = rig.check(cfg, ref);
    free(qf); free(kf); free(vf); free(ref);
    return fail;
}

// ======================================================================
// BENCH
// ======================================================================
template <int HEAD_DIM>
static void bench_decode(int B, int Hq, int Hkv, int seq_len) {
    PagedRig rig(B, Hq, Hkv, HEAD_DIM, std::vector<int>(B, 1),
                 std::vector<int>(B, seq_len), /*ragged=*/false,
                 /*split_partials=*/true,
                 /*ctx=*/max(16384, seq_len + 16),
                 /*pool=*/B * (seq_len + 16), /*reqs=*/B);
    rig.fill_data();
    rig.fill_indices();

    AttentionParams<bf16> p = rig.base_params();
    p.causal_offset = 0;
    auto launch = [&]() {
        dispatch_by_head_dim(HEAD_DIM, PagedDecodeDispatch{p});
    };
    // Decode: q_len=1, query is the last token → attends to all [0, seq_len).
    // FLOPs = 2 * (QK^T + PV) = 4 * B * Hq * seq_len * D.
    double flops = 4.0 * B * Hq * (double)seq_len * HEAD_DIM;
    BenchResult r = bench_kernel(launch, 3, 10, flops);

    char cfg[64];
    snprintf(cfg, sizeof(cfg), "DEC B=%2d Hq=%2d Hk=%d kv=%4d D=%3d",
             B, Hq, Hkv, seq_len, HEAD_DIM);
    print_bench_row(cfg, r);
}

template <int HEAD_DIM>
static void bench_prefill(int B, int Hq, int Hkv, int q_len, int kv_len, int causal) {
    PagedRig rig(B, Hq, Hkv, HEAD_DIM, std::vector<int>(B, q_len),
                 std::vector<int>(B, kv_len), /*ragged=*/true,
                 /*split_partials=*/false, /*ctx=*/0, /*pool=*/0, /*reqs=*/B);
    rig.fill_data();
    rig.fill_indices();

    std::vector<int> q_lens(B, q_len);
    int *d_qtb, *d_qti;
    int num_q_tiles = make_q_tile_mapping(q_lens, &d_qtb, &d_qti);

    AttentionParams<bf16> p = rig.base_params();
    p.causal_offset = causal ? 0 : -1;
    p.q_tile_to_batch = d_qtb; p.q_tile_to_index = d_qti;
    p.num_q_tiles = num_q_tiles;

    auto launch = [&]() {
        dispatch_by_head_dim(HEAD_DIM, PagedPrefillDispatch{p});
    };
    // FLOPs = 2 * (QK^T + PV) = 4 * effective_qk_pairs * Hq * D.
    // Non-causal: effective = q_len * kv_len.
    // Causal: Q row qi attends to [0, causal_off + qi + 1) where
    //   causal_off = kv_len - q_len.  Total KV accesses per request:
    //   sum_{qi=0}^{q_len-1} (kv_len - q_len + qi + 1)
    //   = q_len * (kv_len - q_len) + q_len * (q_len + 1) / 2.
    double eff_kv;
    if (causal) {
        eff_kv = (double)q_len * (kv_len - q_len)
               + (double)q_len * (q_len + 1) / 2.0;
    } else {
        eff_kv = (double)q_len * kv_len;
    }
    double flops = 4.0 * B * Hq * eff_kv * HEAD_DIM;
    BenchResult r = bench_kernel(launch, 3, 10, flops);

    char cfg[80];
    snprintf(cfg, sizeof(cfg), "PRE B=%d Hq=%2d Hk=%d q=%4d kv=%4d D=%3d c=%d",
             B, Hq, Hkv, q_len, kv_len, HEAD_DIM, causal);
    print_bench_row(cfg, r);
    cudaFree(d_qtb); cudaFree(d_qti);
}

int main() {
    int fail = 0;

    // ===== DECODE TESTS =====
    printf("=== Paged Decode Tests ===\n");
    print_paged_header();
    fail += run_decode_test<128>(1, 32, 4, 512, 0, 1);
    fail += run_decode_test<128>(1, 32, 4, 1024, 0, 2);
    fail += run_decode_test<128>(4, 32, 4, 512, 0, 3);
    fail += run_decode_test<128>(8, 32, 4, 1024, 0, 4);
    fail += run_decode_test<128>(4, 32, 8, 2048, 0, 5);
    fail += run_decode_test<128>(1, 16, 1, 256, 0, 6);
    fail += run_decode_test<128>(2, 8, 2, 512, 1, 7);
    fail += run_decode_test<64>(1, 4, 2, 256, 0, 8);
    fail += run_decode_test<256>(1, 2, 1, 256, 0, 9);
    fail += run_decode_test<128>(16, 32, 4, 2048, 0, 10);
    fail += run_decode_test<128>(32, 32, 4, 1024, 0, 11);
    // Production keeps a fixed 32768-wide request table.  This forces 32
    // splits, so seq_len > 512 gives each split multiple cp.async tiles.
    fail += run_decode_test<64>(1, 24, 4, 1100, 0, 12, 32768, 1100);

    // Decode with 2D mask (regression: mixed seq_lens + HasMask)
    fail += run_decode_mask_test<128>(2, 8, 2, 256, 30);
    fail += run_decode_mask_test<128>(4, 32, 4, 512, 31);
    fail += run_decode_mask_test<64>(2, 4, 2, 128, 32);

    if (fail) { printf("\nFAILED decode tests\n"); return fail; }

    // ===== PREFILL TESTS =====
    printf("\n=== Paged Prefill Tests ===\n");
    print_paged_header();
    // Single request, pure prefill (q_len == kv_len)
    {
        std::vector<int> ql = {512};
        std::vector<int> kl = {512};
        fail += run_prefill_test<128>(1, 32, 4, ql, kl, 1, 20);
    }
    {
        std::vector<int> ql = {1024};
        std::vector<int> kl = {1024};
        fail += run_prefill_test<128>(1, 32, 4, ql, kl, 1, 21);
    }
    {
        std::vector<int> ql = {2048};
        std::vector<int> kl = {2048};
        fail += run_prefill_test<128>(1, 32, 4, ql, kl, 1, 22);
    }
    // Ragged batch: different q_lens and kv_lens
    {
        std::vector<int> ql = {128, 256, 64};
        std::vector<int> kl = {128, 256, 64};
        fail += run_prefill_test<128>(3, 32, 4, ql, kl, 1, 23);
    }
    {
        std::vector<int> ql = {64, 128, 256, 32};
        std::vector<int> kl = {64, 128, 256, 32};
        fail += run_prefill_test<128>(4, 32, 4, ql, kl, 1, 24);
    }
    // Extend: kv_len > q_len (append to existing cache)
    {
        std::vector<int> ql = {64, 128};
        std::vector<int> kl = {256, 512};
        fail += run_prefill_test<128>(2, 32, 4, ql, kl, 1, 25);
    }
    // Non-causal
    {
        std::vector<int> ql = {256, 128};
        std::vector<int> kl = {256, 128};
        fail += run_prefill_test<128>(2, 32, 4, ql, kl, 0, 26);
    }
    // Single token (q_len=1 per request, like decode but via prefill path)
    {
        std::vector<int> ql = {1, 1, 1, 1};
        std::vector<int> kl = {128, 256, 64, 512};
        fail += run_prefill_test<128>(4, 32, 4, ql, kl, 1, 27);
    }
    // D=64
    {
        std::vector<int> ql = {128, 64};
        std::vector<int> kl = {128, 64};
        fail += run_prefill_test<64>(2, 4, 2, ql, kl, 1, 28);
    }
    // D=256
    {
        std::vector<int> ql = {128, 64};
        std::vector<int> kl = {128, 64};
        fail += run_prefill_test<256>(2, 2, 1, ql, kl, 1, 29);
    }

    // Prefill with 4D causal mask (regression: single-request mask path)
    fail += run_prefill_mask_test<128>(32, 4, 512, 40);
    fail += run_prefill_mask_test<128>(32, 4, 1024, 41);
    fail += run_prefill_mask_test<64>(4, 2, 256, 42);

    if (fail) { printf("\nFAILED prefill tests\n"); return fail; }
    printf("\nAll tests passed!\n");

    // ===== BENCH =====
    printf("\n===== PAGED DECODE BENCH =====\n");
    print_bench_header();
    bench_decode<128>(1, 32, 4, 512);
    bench_decode<128>(1, 32, 4, 1024);
    bench_decode<128>(1, 32, 4, 2048);
    bench_decode<128>(1, 32, 4, 4096);
    bench_decode<128>(1, 32, 4, 16384);
    bench_decode<128>(4, 32, 4, 2048);
    bench_decode<128>(16, 32, 4, 2048);

    printf("\n===== PAGED PREFILL BENCH =====\n");
    print_bench_header();
    bench_prefill<128>(1, 32, 4, 512, 512, 0);
    bench_prefill<128>(1, 32, 4, 1024, 1024, 0);
    bench_prefill<128>(1, 32, 4, 2048, 2048, 0);
    bench_prefill<128>(1, 32, 4, 2048, 2048, 1);
    bench_prefill<128>(4, 32, 4, 2048, 2048, 1);
    bench_prefill<128>(1, 32, 4, 4096, 4096, 1);

    return 0;
}
