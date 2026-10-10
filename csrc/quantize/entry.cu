/*
 * Quantize's entry implementation: the composed one-call pass, the ring
 * binding and the dtype dispatch, compiled into every module that needs the
 * quantize chain — the quantize module itself and the fp8-linear composition
 * in the gemm module (which shares it instead of re-deriving it, so the
 * implementation is a plain .cu listed in both modules' CMake source lists).
 * The declaration surface is api/quantize.h; this TU is the body.
 *
 * The instantiation list lives HERE, not in the header: every expansion site
 * (the refusal message, the dispatch switch, the supported-input check) is
 * in this file, the ASTRAI_GEMM_PAIRS-in-gemm.cu shape — a list that only
 * one TU expands does not need to live where other TUs can see it.
 */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <torch/extension.h>

#include <api/dtype.h>
#include <api/fp8_checks.h>
#include <api/quantize.h>
#include <api/quantize_common.h>
#include <datatype/element.cuh>
#include <kernel/quantize/kernel.cuh>

namespace astrai {
namespace quant {

/*
 * Supported input element types; ScalarTypeOf<T> supplies the PyTorch
 * boundary value. The refusal and dispatch use these same rows.
 */
#define ASTRAI_QUANT_IN_DTYPES(X)                                                                  \
    X(bf16)                                                                                        \
    X(fp16)                                                                                        \
    X(float)

namespace {

/*
 * A scalar type quantize has no kernel for: say which ones it does have,
 * read off the list above (the attention_dtypes.h pattern).
 */
std::string unsupported_quant_input_message(at::ScalarType st) {
    std::string instantiated;
#define ASTRAI_QUANT_NAME_ROW(T)                                                                   \
    instantiated += std::string(instantiated.empty() ? "" : ", ") + toString(scalar_type_v<T>);
    ASTRAI_QUANT_IN_DTYPES(ASTRAI_QUANT_NAME_ROW)
#undef ASTRAI_QUANT_NAME_ROW
    return std::string("quantize has no kernel for ") + toString(st) +
           " (instantiated: " + instantiated + ")";
}

/*
 * Dtype dispatch over the merged quantize launcher: one case per row above
 * for a (possibly mixed) fp8 output pair. Unsupported inputs throw.
 */
template <typename Fp8TA, typename Fp8TB>
void launch_for_dtype(const torch::Tensor& x, const QuantParams& p, cudaStream_t stream) {
    switch (x.scalar_type()) {
#define ASTRAI_QUANT_CASE(T)                                                                       \
    case scalar_type_v<T>:                                                                         \
        return launch_fp8_quantize<Fp8TA, T, Fp8TB>(p, stream);
        ASTRAI_QUANT_IN_DTYPES(ASTRAI_QUANT_CASE)
#undef ASTRAI_QUANT_CASE
    }
    throw std::runtime_error(unsupported_quant_input_message(x.scalar_type()));
}

/*
 * Row-major format picks the A side, transposed format the B side (both the
 * same in every non-hybrid use).
 */
void launch_quantize_for(
    const torch::Tensor& x, const QuantParams& p, bool a_e5m2, bool b_e5m2, cudaStream_t stream) {
    if (a_e5m2)
        b_e5m2 ? launch_for_dtype<fp8_e5m2, fp8_e5m2>(x, p, stream)
               : launch_for_dtype<fp8_e5m2, fp8_e4m3>(x, p, stream);
    else
        b_e5m2 ? launch_for_dtype<fp8_e4m3, fp8_e5m2>(x, p, stream)
               : launch_for_dtype<fp8_e4m3, fp8_e4m3>(x, p, stream);
}

} // namespace

/*
 * The delayed-scaling ring as raw device pointers. Offsets are RingLayout's;
 * ``hist_len`` is required (numel cannot recover it — the composed ring's
 * trailing pair overshoots). Pair 1 is that double buffer's, so it is not
 * bound here: its publisher passes its own slots.
 */
struct RingView {
    float* hist = nullptr;
    float* scale_out = nullptr;
    float* scale_recip_out = nullptr;
    float* scratch = nullptr;
    unsigned int* done = nullptr;
    int len = 0;
    torch::Tensor amax; // the fold's raw-domain amax sink (RingLayout::amax)
    bool bound = false;
};

static RingView ring_view(const torch::Tensor& st, int64_t hist_idx, int64_t hist_len) {
    RingView r;
    TORCH_CHECK(st.is_cuda() && st.dim() == 1 && st.scalar_type() == torch::kFloat32,
                "ring state must be a 1D float32 CUDA tensor");
    const RingLayout layout{hist_len};
    const int64_t n = hist_len;
    TORCH_CHECK(n > 0 && hist_idx >= 0 && hist_idx < n,
                "ring state too small or hist_idx out of range");
    TORCH_CHECK(st.numel() >= layout.size(/*pairs=*/1), "ring state holds ", st.numel(),
                " floats: a history of ", n, " needs at least ", layout.size(1));
    float* base = st.data_ptr<float>();
    r.hist = base;
    r.scale_out = base + layout.scale(0);
    r.scale_recip_out = base + layout.recip(0);
    r.done = reinterpret_cast<unsigned int*>(base + layout.done());
    r.scratch = base + layout.scratch();
    r.len = static_cast<int>(n);
    r.amax = st.narrow(0, layout.amax(), 1);
    r.bound = true;
    return r;
}

static void
bind_ring(QuantParams& p, const RingView& r, int64_t hist_idx, double fp8_max, double pow2_margin) {
    p.fold_ring = true;
    p.hist = r.hist;
    p.scale_out = r.scale_out;
    p.scale_recip_out = r.scale_recip_out;
    p.amax_scratch = r.scratch;
    p.done = r.done;
    p.hist_len = r.len;
    p.hist_idx = static_cast<int>(hist_idx);
    p.fp8_max = static_cast<float>(fp8_max);
    p.pow2_margin = static_cast<float>(pow2_margin);
}

/*
 * Validates and launches one pass; layout selects row-major/transposed outputs.
 * transposed_dtype defaults to dtype_a; differing formats convert from one
 * read and match two single-format passes. A ring enables delayed-scaling fold
 * and requires hist_len; publication overrides can target its next scale/
 * reciprocal pair to avoid a snapshot. Without a ring, only scale and cast run.
 */
QuantizeOutputs run_quantize(torch::Tensor x,
                             torch::Tensor scale,
                             QuantLayout layout,
                             at::ScalarType dtype_a,
                             c10::optional<at::ScalarType> dtype_b,
                             c10::optional<torch::Tensor> ring,
                             int64_t hist_idx,
                             double fp8_max,
                             double pow2_margin,
                             c10::optional<torch::Tensor> pub_scale,
                             c10::optional<torch::Tensor> pub_recip,
                             c10::optional<int64_t> hist_len) {
    TORCH_CHECK(x.is_cuda(), "CUDA tensors required");
    {
        bool supported = false;
#define ASTRAI_QUANT_SUPPORTED_ROW(T) supported = supported || x.scalar_type() == scalar_type_v<T>;
        ASTRAI_QUANT_IN_DTYPES(ASTRAI_QUANT_SUPPORTED_ROW)
#undef ASTRAI_QUANT_SUPPORTED_ROW
        if (!supported)
            throw std::runtime_error(unsupported_quant_input_message(x.scalar_type()));
    }
    const at::ScalarType out_dtype = dtype_a;
    const at::ScalarType t_dtype = dtype_b.has_value() ? *dtype_b : dtype_a;
    for (const at::ScalarType dt : {out_dtype, t_dtype})
        TORCH_CHECK(dt == torch::kFloat8_e4m3fn || dt == torch::kFloat8_e5m2,
                    "unsupported quantize output dtype: expected "
                    "float8_e4m3fn or float8_e5m2");
    TORCH_CHECK(layout == QuantLayout::RowMajor || x.dim() >= 2,
                "transposed quantize layouts need a 2D+ tensor");
    TORCH_CHECK(scale.is_cuda() && scale.device() == x.device() &&
                    scale.scalar_type() == torch::kFloat32 && scale.numel() == 1,
                "scale must be a CUDA float32 scalar on the input device");
    TORCH_CHECK(at::cuda::getDeviceProperties(x.device().index())->major >= 8,
                "quantize requires compute capability 8.0+");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto input = x.contiguous();
    auto out_opts = input.options().dtype(out_dtype);
    auto out_opts_t = input.options().dtype(t_dtype);

    QuantParams p;
    p.input_ptr = input.data_ptr();
    p.scale = scale.data_ptr<float>();
    QuantizeOutputs outs;
    if (ring.has_value() && ring->defined()) {
        /*
         * A wrong window is silent (scale slots read as history), so a ring
         * without its hist_len is an error, not a guess.
         */
        TORCH_CHECK(hist_len.has_value(), "quantize: ring_state needs hist_len (the history window "
                                          "length) — the buffer's trailing slots make it "
                                          "unrecoverable from numel");
        const RingView r = ring_view(*ring, hist_idx, *hist_len);
        outs.amax = r.amax;
        p.amax = r.amax.data_ptr<float>();
        bind_ring(p, r, hist_idx, fp8_max, pow2_margin);
        if (pub_scale.has_value() && pub_recip.has_value()) {
            TORCH_CHECK(pub_scale->is_cuda() && pub_recip->is_cuda() &&
                            pub_scale->scalar_type() == torch::kFloat32 &&
                            pub_recip->scalar_type() == torch::kFloat32 &&
                            pub_scale->numel() == 1 && pub_recip->numel() == 1,
                        "publish override slots must be CUDA float32 scalars");
            p.scale_out = pub_scale->data_ptr<float>();
            p.scale_recip_out = pub_recip->data_ptr<float>();
        }
    }
    /*
     * The merged kernel views the whole buffer as one flat [rows][cols]
     * tile grid: leading dims fold into rows so 1D and 3D inputs are fully
     * covered. An empty tensor folds to rows=0 with a 1-wide cols axis —
     * the launcher still fires one block so the ring fold publishes.
     */
    const int64_t numel = input.numel();
    const int64_t cols = numel == 0 ? 1 : input.size(-1);
    const int64_t rows = numel / cols;
    TORCH_CHECK(cols <= INT32_MAX && rows <= INT32_MAX,
                "quantize tensor too large for the tiled grid");
    p.rows = static_cast<int>(rows);
    p.cols = static_cast<int>(cols);
    /*
     * The merged kernel takes placement as data: the stride pair for the
     * row-major side ((cols, 1)); the transposed side derives its canonical
     * (1, rows) contract in-kernel.
     */
    p.out_row_stride = p.cols;
    p.out_col_stride = 1;
    if (layout != QuantLayout::Transposed) {
        /*
         * Direct-to-allocator empty (no dispatcher round trip): the outputs
         * are fresh kernel destinations, never autograd-visible on their own.
         */
        outs.out = torch::Tensor(
            at::detail::empty_cuda(input.sizes(), out_dtype, input.device(), std::nullopt));
        p.output_ptr = outs.out.data_ptr();
    }
    if (layout != QuantLayout::RowMajor) {
        outs.out_t = torch::Tensor(
            at::detail::empty_cuda({cols, rows}, t_dtype, input.device(), std::nullopt));
        p.output_transposed_ptr = outs.out_t.data_ptr();
    }
    launch_quantize_for(input, p, out_dtype == torch::kFloat8_e5m2, t_dtype == torch::kFloat8_e5m2,
                        stream.stream());
    C10_CUDA_CHECK(cudaGetLastError());
    return outs;
}

} // namespace quant
} // namespace astrai
