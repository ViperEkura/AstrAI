#pragma once
/*
 * quant_gemm's op entry ladder, in one place: classify the dtype pair, gate
 * the device, resolve the dequant scales, resolve layouts/leading dims,
 * validate the geometry, allocate, fill GemmParams, dispatch. Declared in
 * api/gemm.h; the pybind spelling is in bindings.cu; gemm.cu's
 * quant_gemm_impl is the thin wrapper that instantiates the ladder with the
 * family's dtype-pair lookup. The family's TU-local impl header (the
 * attention/entry.h shape): it lives beside its consumers, not in
 * include/launcher/, which keeps declaration surfaces only.
 *
 * The lookup is a template parameter on purpose: the pair table and the
 * switch stamped from it are gemm.cu's (anonymous namespace, TU-internal by
 * design), so passing the lookup in keeps this header free of any
 * include-order contract — it includes at the top of its TU like any other
 * header.
 *
 * Scale contract — the one spelling the docs and the Python adapters point
 * at instead of restating: a scale is a contiguous CUDA float32 tensor of
 * numel 1 (per-tensor device scalar) or the operand's extent (per-row
 * activations a_scale[m], per-channel weights b_scale[n]). Both fold
 * multiplicatively into the epilogue, so both must be indexable by output
 * coordinates; a scale indexed along K cannot be represented here — it would
 * have to apply inside the mainloop accumulation (GemmParams says the same;
 * docs/developer/kernels/gemm.md, "Scales", is the prose home). Per side:
 * int8 requires its scale, fp8 takes one optionally, bf16 rejects one.
 */

#include <algorithm>

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <api/gemm_common.h>

#include <api/fp8_checks.h>
#include <api/gemm.h>
#include <utils/device.cuh>

namespace astrai {
namespace gemm {

namespace {

/*
 * Inner-layout resolution for one GEMM operand. The user flag names the math
 * (0 = last two dims are [rows][contract], 1 = transposed); the storage may
 * independently be a col-major view (.t() of a contiguous buffer), which
 * folds into the returned dispatch flag at zero copy — the kernel's
 * LayoutA/LayoutB tags cover both storages. m/n/k derive from the user flag
 * only. Tensors whose inner dims are neither natural layout fall back to
 * .contiguous().
 */
bool resolve_operand(const torch::Tensor& t_in,
                     bool flag,
                     int64_t& ld,
                     int64_t& batch_stride,
                     torch::Tensor& storage) {
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

// A resolved dequant scale: n == 0 marks the per-tensor device scalar.
struct QuantScale {
    const float* ptr = nullptr;
    int n = 0;
};

QuantScale resolve_quant_scale(const torch::Tensor& s,
                               int64_t extent,
                               const char* name,
                               const c10::Device& device) {
    TORCH_CHECK(s.defined(), name, " is required");
    TORCH_CHECK(s.is_cuda() && s.scalar_type() == torch::kFloat32 && s.is_contiguous(), name,
                " must be a contiguous CUDA float32 tensor");
    TORCH_CHECK(s.device() == device, name, " must be on the same device as operands");
    TORCH_CHECK(s.numel() == 1 || s.numel() == extent, name,
                " must hold 1 element (per-tensor) or ", extent, " (per-row/per-channel)");
    return {s.data_ptr<float>(), s.numel() == 1 ? 0 : (int)extent};
}

} // namespace

/*
 * The single quantized-GEMM entry body (one kernel for every cell, the only
 * kernel-facing export). The dtype pair picks the mma mode:
 *   bf16 x bf16 (W16A16)        — no scales
 *   bf16 x int8 (W8A16)         — b_scale required
 *   int8 x int8 (W8A8)          — both scales required
 *   fp8 x fp8, matching formats — both scales optional
 * Scale arity is validated per side: int8 requires its dequant scale, fp8
 * takes one optionally (per-tensor scalar or the operand's extent), bf16
 * rejects one (nothing to dequant). The body packs GemmParams (batch
 * broadcast rules, zero-copy transposed views, fused bf16 bias) and hands
 * it to the dtype-pair dispatch `Lookup` selects.
 */
template <auto Lookup>
torch::Tensor quant_gemm_ladder(torch::Tensor a,
                                torch::Tensor b,
                                c10::optional<torch::Tensor> a_scale,
                                c10::optional<torch::Tensor> b_scale,
                                bool trans_a,
                                bool trans_b,
                                c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK((a.dim() == 2 || a.dim() == 3) && (b.dim() == 2 || b.dim() == 3),
                "a and b must be 2D or 3D (batched)");
    TORCH_CHECK(a.device() == b.device(), "a and b must share device");
    const auto dt_a = a.scalar_type(), dt_b = b.scalar_type();
    const bool i8a = dt_a == torch::kChar, i8b = dt_b == torch::kChar;
    const bool f8a = dt_a == torch::kFloat8_e4m3fn || dt_a == torch::kFloat8_e5m2;
    const bool f8b = dt_b == torch::kFloat8_e4m3fn || dt_b == torch::kFloat8_e5m2;
    const bool b16a = dt_a == torch::kBFloat16, b16b = dt_b == torch::kBFloat16;
    /*
     * The supported-pair set is validated once, by the dispatch switch's
     * default arm (the Lookup), with the operand dtypes in the message.
     */
    if (f8a || f8b) {
        astrai::quant::check_fp8_device(a.device().index());
    }
    const int64_t m = trans_a ? a.size(-1) : a.size(-2);
    const int64_t n = trans_b ? b.size(-2) : b.size(-1);
    const int64_t k = trans_a ? a.size(-2) : a.size(-1);
    TORCH_CHECK(m > 0 && n > 0 && k > 0, "quant_gemm: M, N and K must be greater than zero (got ",
                m, ", ", n, ", ", k, ")");
    auto opt_scale = [&](const c10::optional<torch::Tensor>& s, int64_t extent, const char* name,
                         bool i8_side, bool bf16_side) -> QuantScale {
        if (!s.has_value()) {
            TORCH_CHECK(!i8_side, "quant_gemm: ", name, " is required for an int8 operand");
            return {nullptr, 0};
        }
        TORCH_CHECK(!bf16_side, "quant_gemm: ", name,
                    " given for a bf16 operand (nothing to dequant)");
        return resolve_quant_scale(*s, extent, name, a.device());
    };
    const QuantScale sa = opt_scale(a_scale, m, "a_scale", i8a, b16a);
    const QuantScale sb = opt_scale(b_scale, n, "b_scale", i8b, b16b);

    torch::Tensor bias_t;
    if (bias.has_value())
        bias_t = *bias;
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t batch_a = a.dim() == 3 ? a.size(0) : 1;
    const int64_t batch_b = b.dim() == 3 ? b.size(0) : 1;
    TORCH_CHECK(batch_a == batch_b || batch_a == 1 || batch_b == 1, "batch dim mismatch (got ",
                batch_a, " and ", batch_b, ")");
    const int64_t batch = std::max(batch_a, batch_b);
    TORCH_CHECK(batch <= 65535, "batch dim exceeds the grid.z launch limit");

    torch::Tensor a_st, b_st;
    int64_t a_ld, b_ld, a_bstride, b_bstride;
    const bool tag_a = resolve_operand(a, trans_a, a_ld, a_bstride, a_st);
    const bool tag_b = resolve_operand(b, trans_b, b_ld, b_bstride, b_st);
    TORCH_CHECK(k == (trans_b ? b.size(-1) : b.size(-2)), "inner dim mismatch");
    TORCH_CHECK(sa.ptr == nullptr || sa.n == 0 || sa.n == m, "a_scale extent must match m");
    TORCH_CHECK(sb.ptr == nullptr || sb.n == 0 || sb.n == n, "w_scale extent must match n");

    const bool batched_out = a.dim() == 3 || b.dim() == 3;
    torch::Tensor output = batched_out
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
        TORCH_CHECK(bias_t.device() == a.device(),
                    "quantized gemm bias must be on the same device as operands");
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

    const DeviceFacts dev = device_facts();
    Lookup(dt_a, dt_b, dev)(p, stream.stream(), tag_a, tag_b, dev);
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

} // namespace gemm
} // namespace astrai
