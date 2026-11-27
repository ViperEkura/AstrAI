/*
 * The fp8 training linear — forward *and* backward — in C++: quantize x/w with
 * the delayed-scaling rings, run the pre-quantized GEMMs, keep the transposed
 * operands for the backward, update the rings.
 *
 * It lives in the gemm module because that module owns the GEMM dispatch state
 * (plan table, planner mode, staging switches) — a second copy in another .so
 * would let `set_planner` configure one and the training path launch through the
 * other. Quantize comes in through api/torch/quantize.h, GEMM through
 * api/torch/gemm.h: the same code the standalone bindings run.
 *
 * The composition is C++ because a ``torch::autograd::Function`` runs both
 * directions inside the engine's call — only the dispatcher entry stays in
 * Python, and the per-linear host cost of the old Python chain is gone.
 *
 * Not here by design: the autocast region and recipe/format policy
 * (astrai/extension/policy/quantization/autocast.py), the module slot table
 * (astrai/extension/policy/quantization/autocast.py), and the rings / activation cache / snapshot
 * (ring.h, cache.h, state.h). The state owner and checkpoint
 * interface live in runtime.cu in this module.
 */

#include <c10/util/Optional.h>
#include <torch/extension.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "linear.h"
#include "state.h"
#include <api/torch/gemm.h>
#include <api/torch/quantize.h>

namespace astrai {
namespace fp8 {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

namespace {

/*
 * Composed forward / backward
 */

struct Fp8FwdOut {
    Tensor out;
    Tensor sx, sw;
    Tensor x8T, w8T; // K-contiguous transposed casts (undefined when absent)
};

Fp8FwdOut fp8_forward_impl(const Tensor& x,
                           const Tensor& w,
                           const c10::optional<Tensor>& bias,
                           bool update_rings,
                           bool dynamic,
                           int64_t history_len,
                           int64_t margin,
                           at::ScalarType fmt_a,
                           at::ScalarType fmt_b,
                           int64_t slot) {
    Fp8FwdOut res;
    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end() - 1);
    out_shape.push_back(w.size(0));

    State& st = state();
    if (dynamic) {
        // Current-amax scaling: measure, then quantize — no rings, no cache.
        const auto dyn = [&](const Tensor& t) {
            return scale_from_amax(amax_of(t), fmt_a, margin);
        };
        res.sx = dyn(x.reshape({-1, w.size(1)}));
        res.sw = dyn(w);
        const auto qx = quant::quantize(x, res.sx.reciprocal(), fmt_a, c10::nullopt);
        const Tensor w8 = quant::quantize(w, res.sw.reciprocal(), fmt_a, c10::nullopt).out;
        res.out = gemm::quant_gemm_impl(qx.out.reshape({-1, qx.out.size(-1)}), w8, res.sx, res.sw,
                                        false, true, bias)
                      .reshape(out_shape);
        return res;
    }

    /*
     * Slot-addressed when the dispatcher knows the module (``slot >= 0``): the
     * meta then belongs to the module path and survives a replaced weight
     * parameter. Bare calls, benches and every existing test pass no slot and
     * keep the address-keyed lookup byte for byte.
     */
    auto meta = get_meta(w, history_len, margin, slot);
    /*
     * Host-side seed only when a ring has never been used: both branches need
     * a valid scale to cast with.
     */
    if (!meta->w.initialized)
        meta->w.seed(w, fmt_a, margin);
    const bool cache_activation =
        update_rings && st.act_cache_enabled.load(std::memory_order_relaxed);
    c10::optional<ActivationCast> cached_activation;
    if (cache_activation) {
        cached_activation = st.act_cache.find(x, fmt_a, fmt_b, history_len, margin);
        if (cached_activation.has_value())
            meta->x = cached_activation->ring;
    }
    if (!meta->x->initialized)
        meta->x->seed(x, fmt_a, margin);
    // The weight history advances once per version; the bytes always come from w.
    auto& ring = meta->w;
    const auto version = w.is_inference() ? -1 : version_of(w);
    const bool same_scale =
        version >= 0 && version == meta->weight_version && fmt_a == meta->weight_format;
    res.sw = same_scale ? ring.pub_scale() : ring.scale();
    if (update_rings)
        res.sw = res.sw.clone();
    const auto qw =
        update_rings && !same_scale
            ? quant::quantize_scaled(w, ring.scale_recip(), ring.state, history_len, ring.idx,
                                     fmt_a, fmt_b, margin, quant_max_of(fmt_a), ring.pub_scale(),
                                     ring.pub_recip())
            : quant::quantize(w, same_scale ? ring.pub_recip() : ring.scale_recip(), fmt_a,
                              update_rings ? c10::make_optional(fmt_b) : c10::nullopt);
    if (update_rings && !same_scale) {
        ring.advance();
        meta->weight_version = version;
        meta->weight_format = fmt_a;
    }
    res.w8T = qw.out_t;

    // A later microbatch can advance the ring before this backward runs.
    res.sx = cached_activation ? cached_activation->scale : meta->x->scale();
    if (update_rings && !cached_activation)
        res.sx = res.sx.clone();
    Tensor x8;
    if (update_rings) {
        if (cached_activation.has_value()) {
            x8 = cached_activation->out;
            res.x8T = cached_activation->out_t;
        } else {
            const auto qx = quant::quantize_scaled(
                x, meta->x->scale_recip(), meta->x->state, history_len, meta->x->idx, fmt_a, fmt_b,
                margin, quant_max_of(fmt_a), meta->x->pub_scale(), meta->x->pub_recip());
            x8 = qx.out;
            res.x8T = qx.out_t;
            meta->x->advance();
            if (cache_activation) {
                st.act_cache.insert(x, qx.out, qx.out_t, res.sx, meta->x, margin);
            }
        }
    } else {
        x8 = quant::quantize(x, meta->x->scale_recip(), fmt_a, c10::nullopt).out;
    }
    res.out = gemm::quant_gemm_impl(x8.reshape({-1, x8.size(-1)}), qw.out, res.sx, res.sw, false,
                                    true, bias)
                  .reshape(out_shape);
    return res;
}

struct Fp8BwdIn {
    Tensor x, w, sx, sw, x8T, w8T;
    bool dynamic = false;
    bool need_bias_grad = false;
    int64_t history_len = 16, margin = 0;
    at::ScalarType fmt_b = at::kFloat8_e5m2;
};

tensor_list fp8_backward_impl(const Tensor& g, const Fp8BwdIn& in) {
    const Tensor g2 = g.reshape({-1, g.size(-1)});
    Tensor sx = in.sx, sw = in.sw, sg;
    std::shared_ptr<Fp8Meta> meta;
    if (in.dynamic) {
        const auto dyn = [&](const Tensor& t) {
            return scale_from_amax(amax_of(t), in.fmt_b, in.margin);
        };
        sg = dyn(g2);
        sw = dyn(in.w);
        sx = dyn(in.x);
    } else {
        meta = get_meta(in.w, in.history_len, in.margin, -1);
        if (!meta->g.initialized)
            meta->g.seed(g2, in.fmt_b, in.margin);
        sg = meta->g.scale(); // ring view: g's fold publishes the other pair
    }
    /*
     * NT fast path via transposed quantize outputs: g8 + w8T gives grad_x, g8T +
     * x8T gives grad_w. g is consumed in both orientations (one dual pass feeds
     * both); x8T/w8T came from the forward, so the
     * backward re-reads neither x nor w.
     */
    const auto qg = in.dynamic ? quant::quantize(g2, sg.reciprocal(), in.fmt_b, in.fmt_b)
                               : quant::quantize_scaled(g2, meta->g.scale_recip(), meta->g.state,
                                                        in.history_len, meta->g.idx, in.fmt_b,
                                                        in.fmt_b, in.margin, quant_max_of(in.fmt_b),
                                                        meta->g.pub_scale(), meta->g.pub_recip());
    Tensor x8T = in.x8T;
    if (!x8T.defined()) {
        x8T = quant::quantize(in.x.reshape({-1, in.x.size(-1)}), sx.reciprocal(), c10::nullopt,
                              in.fmt_b)
                  .out_t;
    }
    Tensor w8T = in.w8T;
    if (!w8T.defined()) {
        w8T = quant::quantize(in.w, sw.reciprocal(), c10::nullopt, in.fmt_b).out_t;
    }
    Tensor grad_x =
        gemm::quant_gemm_impl(qg.out, w8T, sg, sw, false, true, c10::nullopt).reshape(in.x.sizes());
    Tensor grad_w = gemm::quant_gemm_impl(qg.out_t, x8T, sg, sx, false, true, c10::nullopt);
    // A bias-free linear must not pay g2.sum(0) — another full gradient read.
    Tensor grad_b;
    if (in.need_bias_grad)
        grad_b = g2.sum(0).to(at::kBFloat16);
    if (meta)
        meta->g.advance();
    return {grad_x, grad_w, grad_b};
}

/*
 * apply() demands one returned gradient per forward argument — undefined for the
 * non-tensor ones, which the engine filters out. The slot id is forward-only:
 * backward reaches its meta through the weight the forward saved.
 */
constexpr size_t kFp8TensorInputs = 3; // x, w, bias
constexpr size_t kFp8ScalarInputs = 8; // update_rings .. slot

} // namespace

/*
 * The autograd node
 */

/*
 * ``forward`` runs inside the engine with grad mode off, so ``update_rings``
 * comes from the dispatcher (the caller's mode): a no-grad pass — checkpointing
 * recompute, inference — reads the rings without folding or advancing them.
 */
class Fp8Linear : public torch::autograd::Function<Fp8Linear> {
  public:
    static Tensor forward(AutogradContext* ctx,
                          Tensor x,
                          Tensor w,
                          Tensor bias,
                          bool update_rings,
                          bool need_bias_grad,
                          bool dynamic,
                          int64_t history_len,
                          int64_t margin,
                          at::ScalarType fmt_a,
                          at::ScalarType fmt_b,
                          int64_t slot) {
        c10::optional<Tensor> bias_opt = c10::nullopt;
        if (bias.numel() > 0)
            bias_opt = bias;
        const Fp8FwdOut res = fp8_forward_impl(x, w, bias_opt, update_rings, dynamic, history_len,
                                               margin, fmt_a, fmt_b, slot);
        ctx->save_for_backward({x, w});
        if (res.x8T.defined())
            ctx->saved_data["x8T"] = res.x8T;
        if (res.w8T.defined())
            ctx->saved_data["w8T"] = res.w8T;
        ctx->saved_data["sx"] = res.sx;
        ctx->saved_data["sw"] = res.sw;
        ctx->saved_data["dynamic"] = dynamic;
        ctx->saved_data["need_bias_grad"] = need_bias_grad;
        ctx->saved_data["history_len"] = history_len;
        ctx->saved_data["margin"] = margin;
        ctx->saved_data["fmt_b"] = static_cast<int64_t>(fmt_b);
        return res.out;
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
        const auto saved = ctx->get_saved_variables();
        Fp8BwdIn in;
        in.x = saved[0];
        in.w = saved[1];
        if (ctx->saved_data.count("x8T"))
            in.x8T = ctx->saved_data["x8T"].toTensor();
        if (ctx->saved_data.count("w8T"))
            in.w8T = ctx->saved_data["w8T"].toTensor();
        in.sx = ctx->saved_data["sx"].toTensor();
        in.sw = ctx->saved_data["sw"].toTensor();
        in.dynamic = ctx->saved_data["dynamic"].toBool();
        in.need_bias_grad = ctx->saved_data["need_bias_grad"].toBool();
        in.history_len = ctx->saved_data["history_len"].toInt();
        in.margin = ctx->saved_data["margin"].toInt();
        in.fmt_b = static_cast<at::ScalarType>(ctx->saved_data["fmt_b"].toInt());
        tensor_list g = fp8_backward_impl(grad_outputs[0], in);
        g.resize(kFp8TensorInputs + kFp8ScalarInputs, Tensor());
        return g;
    }
};

// The dispatcher's entry: one Python->C++ crossing per linear.
Tensor fp8_linear(const Tensor& x,
                  const Tensor& w,
                  const c10::optional<Tensor>& bias,
                  bool update_rings,
                  bool need_bias_grad,
                  bool dynamic,
                  int64_t history_len,
                  int64_t margin,
                  at::ScalarType fmt_a,
                  at::ScalarType fmt_b,
                  int64_t slot) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "fp8 linear needs CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
                "fp8 linear takes bf16 operands");
    TORCH_CHECK(x.size(-1) == w.size(1), "fp8 linear: inner dim mismatch");
    Tensor bias_t;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() > 0, "fp8 linear: bias must be non-empty");
        bias_t = *bias;
    } else {
        /*
         * One empty placeholder per device: no-bias is the hot path (Linear
         * defaults to bias=False) and a per-call empty({0}) is an allocation.
         */
        static std::mutex mu;
        static std::unordered_map<c10::DeviceIndex, Tensor> cache;
        const auto dev = x.device().index();
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(dev);
        if (it == cache.end())
            it = cache.emplace(dev, torch::empty({0}, x.options())).first;
        bias_t = it->second;
    }
    return Fp8Linear::apply(x, w, bias_t, update_rings, need_bias_grad, dynamic, history_len,
                            margin, fmt_a, fmt_b, slot);
}

} // namespace fp8
} // namespace astrai
