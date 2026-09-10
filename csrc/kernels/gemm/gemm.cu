// GEMM family binding (module `gemm`): the single quantized-GEMM entry
// ``quant_gemm`` — every dtype pairing (bf16 / int8 / fp8 operands, per-
// operand scales) dispatches over one dtype-generic kernel family. The
// policy instantiation space compiles one explicit instantiation per dtype
// pair (one per .cu below), so the heavy template work runs as parallel
// nvcc jobs. This TU keeps the dtype-pair switch + pybind; the C tests
// instantiate from the headers instead.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <torch/extension.h>

#include "common/device.cuh"
#include "gemm.cuh"
#include "quantize/checks.h"
#include "quantize/common.h"

using namespace astrai;
using namespace astrai::quant;

namespace astrai {
namespace gemm {

// The per-pair specializations are explicitly instantiated in their own TUs
// (gemm_bf16_bf16.cu etc.), one nvcc job per dtype pair. These extern
// template declarations keep the bindings below from re-instantiating: the
// address-of forms are references to the externally defined symbols only.
// (They must sit here, outside the anonymous namespace — nvcc rejects
// extern template declarations in an anonymous namespace.)
extern template void gemm_dispatch<__nv_bfloat16, __nv_bfloat16>(
    GemmParams, cudaStream_t, bool, bool);
extern template void gemm_dispatch<__nv_bfloat16, int8_t>(
    GemmParams, cudaStream_t, bool, bool);
extern template void gemm_dispatch<int8_t, int8_t>(
    GemmParams, cudaStream_t, bool, bool);
extern template void gemm_dispatch<__nv_bfloat16, __nv_fp8_e4m3>(
    GemmParams, cudaStream_t, bool, bool);
extern template void gemm_dispatch<__nv_bfloat16, __nv_fp8_e5m2>(
    GemmParams, cudaStream_t, bool, bool);
extern template void gemm_dispatch<__nv_fp8_e4m3, __nv_fp8_e4m3>(
    GemmParams, cudaStream_t, bool, bool);
extern template void gemm_dispatch<__nv_fp8_e5m2, __nv_fp8_e5m2>(
    GemmParams, cudaStream_t, bool, bool);

namespace {

// Inner-layout resolution for one GEMM operand. The user flag names the
// math (0 = last two dims are [rows][contract], 1 = transposed); the
// storage may independently be a col-major view (.t() of a contiguous
// buffer), which folds into the returned dispatch flag at zero copy — the
// kernel's LayoutA/LayoutB tags cover both storages. m/n/k derive from the
// user flag only. Tensors whose inner dims are neither natural layout fall
// back to .contiguous().
bool resolve_operand(const torch::Tensor& t_in, bool flag, int64_t& ld,
                     int64_t& batch_stride, torch::Tensor& storage) {
    torch::Tensor t = t_in;
    bool col_major = false;
    if (t.stride(-1) != 1) {
        if (t.stride(-2) == 1) {
            col_major = true;
        } else {
            t = t.contiguous();
        }
    }
    storage = t;
    ld = col_major ? t.stride(-1) : t.stride(-2);
    batch_stride = t.dim() == 3 ? t.stride(0) : 0;
    return flag ^ col_major;
}


}  // namespace

// ---------------------------------------------------------------------------
// quant_gemm: the single quantized-GEMM entry over the dtype-generic
// dispatch. Scale semantics follow GemmParams: one float32 element is the
// per-tensor device scalar; a 1-D float32 tensor of the operand's extent
// is per-row (activations) / per-channel (weights), applied in the epilogue.
// ---------------------------------------------------------------------------

namespace {

// A resolved dequant scale: n == 0 marks the per-tensor device scalar.
struct QuantScale {
    const float* ptr = nullptr;
    int n = 0;
};

QuantScale resolve_quant_scale(const torch::Tensor& s, int64_t extent,
                               const char* name) {
    TORCH_CHECK(s.defined(), name, " is required");
    TORCH_CHECK(s.is_cuda() && s.scalar_type() == torch::kFloat32 &&
                    s.is_contiguous(),
                name, " must be a contiguous CUDA float32 tensor");
    TORCH_CHECK(s.numel() == 1 || s.numel() == extent,
                name, " must hold 1 element (per-tensor) or ", extent,
                " (per-row/per-channel)");
    return {s.data_ptr<float>(), s.numel() == 1 ? 0 : (int)extent};
}

// py::object -> torch::Tensor with a uniform error message; a none object
// stays undefined (callers gate on is_none()).
torch::Tensor cast_tensor_arg(const py::object& o, const char* name) {
    try {
        return o.cast<torch::Tensor>();
    } catch (const py::cast_error&) {
        TORCH_CHECK(false, name, " must be a torch.Tensor or None");
        return {};
    }
}

// The dtype-pair dispatch switch below replaces a hand-maintained if/else
// chain: each supported (activation, weight) pair selects its
// gemm_dispatch specialization exactly once, and an unsupported pair raises
// with the actual operand dtypes in the message instead of a hardcoded
// list that can drift out of sync. The specializations are explicitly
// instantiated in their own TUs (one nvcc job per dtype pair), so the
// switch is nothing but a jump table over resolved function pointers —
// the extern template declarations above keep any re-instantiation out of
// this TU. pack_dtypes is constexpr, so every case label is a compile-time
// constant and the switch lowers to one indexed branch, no
// runtime-initialized state. The function is not constexpr only because
// the default arm throws.
using GemmDispatchFn = void (*)(GemmParams, cudaStream_t, bool, bool);

constexpr uint16_t pack_dtypes(c10::ScalarType a, c10::ScalarType b) {
    return static_cast<uint16_t>(static_cast<uint8_t>(a)) << 8 |
           static_cast<uint8_t>(b);
}

GemmDispatchFn find_gemm_dispatch(c10::ScalarType a, c10::ScalarType b) {
    switch (pack_dtypes(a, b)) {
        case pack_dtypes(torch::kBFloat16, torch::kBFloat16):
            return &gemm_dispatch<__nv_bfloat16, __nv_bfloat16>;
        case pack_dtypes(torch::kBFloat16, torch::kChar):
            return &gemm_dispatch<__nv_bfloat16, int8_t>;
        case pack_dtypes(torch::kChar, torch::kChar):
            return &gemm_dispatch<int8_t, int8_t>;
        case pack_dtypes(torch::kBFloat16, torch::kFloat8_e4m3fn):
            return &gemm_dispatch<__nv_bfloat16, __nv_fp8_e4m3>;
        case pack_dtypes(torch::kBFloat16, torch::kFloat8_e5m2):
            return &gemm_dispatch<__nv_bfloat16, __nv_fp8_e5m2>;
        case pack_dtypes(torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn):
            return &gemm_dispatch<__nv_fp8_e4m3, __nv_fp8_e4m3>;
        case pack_dtypes(torch::kFloat8_e5m2, torch::kFloat8_e5m2):
            return &gemm_dispatch<__nv_fp8_e5m2, __nv_fp8_e5m2>;
        default:
            TORCH_CHECK(false,
                        "unsupported operand dtype pair ", toString(a), " x ", toString(b),
                        ": expected bf16 x int8 (W8A16), int8 x int8 (W8A8), "
                        "bf16 x bf16 (W16A16), bf16 x fp8 (W-F8A16), or "
                        "matching fp8 x fp8");
    }
}

}  // namespace

// The single quantized-GEMM entry (one kernel for every cell, the only
// export). The dtype pair picks the mma mode:
//   bf16 x bf16 (W16A16)        — no scales
//   bf16 x int8 (W8A16)         — b_scale required
//   int8 x int8 (W8A8)          — both scales required
//   bf16 x fp8 (W-F8A16)        — b_scale optional (in-register hardware
//                                  widen to the bf16 mma)
//   fp8 x fp8, matching formats — both scales optional
// Scale arity is validated per side: int8 requires its dequant scale, fp8
// takes one optionally (per-tensor scalar or the operand's extent), bf16
// rejects one (nothing to dequant). The body packs GemmParams (batch
// broadcast rules, zero-copy transposed views, fused bf16 bias) and hands
// it to the dtype-pair dispatch.
torch::Tensor quant_gemm(torch::Tensor a, torch::Tensor b, py::object a_scale,
                         py::object b_scale, bool trans_a, bool trans_b,
                         py::object bias) {
    const auto dt_a = a.scalar_type(), dt_b = b.scalar_type();
    const bool i8a = dt_a == torch::kChar, i8b = dt_b == torch::kChar;
    const bool f8a = dt_a == torch::kFloat8_e4m3fn || dt_a == torch::kFloat8_e5m2;
    const bool f8b = dt_b == torch::kFloat8_e4m3fn || dt_b == torch::kFloat8_e5m2;
    const bool b16a = dt_a == torch::kBFloat16, b16b = dt_b == torch::kBFloat16;
    // The supported-pair set is validated once, by the dispatch switch's
    // default arm (find_gemm_dispatch), with the operand dtypes in the
    // message.
    if (f8a || f8b) {
        astrai::quant::check_fp8_device(a.device().index());
    }
    const int64_t m = trans_a ? a.size(-1) : a.size(-2);
    const int64_t n = trans_b ? b.size(-2) : b.size(-1);
    auto opt_scale = [&](py::object s, int64_t extent, const char* name,
                         bool i8_side, bool bf16_side) -> QuantScale {
        if (s.is_none()) {
            TORCH_CHECK(!i8_side, "quant_gemm: ", name,
                        " is required for an int8 operand");
            return {nullptr, 0};
        }
        torch::Tensor t = cast_tensor_arg(s, name);
        TORCH_CHECK(!bf16_side, "quant_gemm: ", name,
                    " given for a bf16 operand (nothing to dequant)");
        return resolve_quant_scale(t, extent, name);
    };
    const QuantScale sa = opt_scale(a_scale, m, "a_scale", i8a, b16a);
    const QuantScale sb = opt_scale(b_scale, n, "b_scale", i8b, b16b);

    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK((a.dim() == 2 || a.dim() == 3) &&
                    (b.dim() == 2 || b.dim() == 3),
                "a and b must be 2D or 3D (batched)");
    TORCH_CHECK(a.device() == b.device(), "a and b must share device");
    torch::Tensor bias_t;
    if (!bias.is_none()) bias_t = cast_tensor_arg(bias, "bias");
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t batch_a = a.dim() == 3 ? a.size(0) : 1;
    const int64_t batch_b = b.dim() == 3 ? b.size(0) : 1;
    TORCH_CHECK(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                "batch dim mismatch (got ", batch_a, " and ", batch_b, ")");
    const int64_t batch = std::max(batch_a, batch_b);
    TORCH_CHECK(batch <= 65535, "batch dim exceeds the grid.z launch limit");

    torch::Tensor a_st, b_st;
    int64_t a_ld, b_ld, a_bstride, b_bstride;
    const bool tag_a = resolve_operand(a, trans_a, a_ld, a_bstride, a_st);
    const bool tag_b = resolve_operand(b, trans_b, b_ld, b_bstride, b_st);
    const int64_t k = trans_a ? a.size(-2) : a.size(-1);
    TORCH_CHECK(k == (trans_b ? b.size(-1) : b.size(-2)), "inner dim mismatch");
    TORCH_CHECK(sa.ptr == nullptr || sa.n == 0 || sa.n == m,
                "a_scale extent must match m");
    TORCH_CHECK(sb.ptr == nullptr || sb.n == 0 || sb.n == n,
                "w_scale extent must match n");

    const bool batched_out = a.dim() == 3 || b.dim() == 3;
    torch::Tensor output =
        batched_out
            ? torch::empty({batch, m, n}, a.options().dtype(torch::kBFloat16))
            : torch::empty({m, n}, a.options().dtype(torch::kBFloat16));
    GemmParams p;
    p.a_ptr = a_st.data_ptr();
    p.b_ptr = b_st.data_ptr();
    p.out_ptr = output.data_ptr();
    p.a_scale = sa.ptr;
    p.a_scale_m = sa.n;
    p.b_scale = sb.ptr;
    p.b_scale_n = sb.n;
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.a_ld = static_cast<int>(a_ld);
    p.b_ld = static_cast<int>(b_ld);
    if (bias_t.defined() && bias_t.numel() > 0) {
        TORCH_CHECK(bias_t.is_cuda() && bias_t.scalar_type() == torch::kBFloat16,
                    "quantized gemm bias must be a CUDA bf16 tensor");
        TORCH_CHECK(bias_t.dim() == 1 && bias_t.size(0) == n,
                    "quantized gemm bias must be 1D of length n=", n);
        TORCH_CHECK(bias_t.is_contiguous(), "bias must be contiguous");
        p.bias_ptr = bias_t.data_ptr();
    }
    p.batch = static_cast<int>(batch);
    p.a_batch_stride = (batch_a == 1 && batch > 1) ? 0 : a_bstride;
    p.b_batch_stride = (batch_b == 1 && batch > 1) ? 0 : b_bstride;
    p.out_batch_stride = m * n;
    p.out_ld = static_cast<int>(n);

    find_gemm_dispatch(dt_a, dt_b)(p, stream.stream(), tag_a, tag_b);
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

}  // namespace gemm
}  // namespace astrai

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_gemm", &astrai::gemm::quant_gemm, py::arg("a"), py::arg("b"),
          py::arg("a_scale") = py::none(), py::arg("b_scale") = py::none(),
          py::arg("trans_a") = false, py::arg("trans_b") = true,
          py::arg("bias") = py::none());
}
