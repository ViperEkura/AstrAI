/*
 * The fp8 training linear — forward *and* backward — in C++: quantize x/w with
 * the delayed-scaling rings, run the pre-quantized GEMMs, keep the transposed
 * operands for the backward, update the rings.
 *
 * It lives in the gemm module because that module owns the GEMM dispatch state
 * (plan table, planner mode, staging switches) — a second copy in another .so
 * would let `set_planner` configure one and the training path launch through the
 * other. Quantize comes in through api/quantize.h, GEMM through
 * api/gemm.h: the same code the standalone bindings run.
 *
 * The composition is C++ because a ``torch::autograd::Function`` runs both
 * directions inside the engine's call — only the dispatcher entry stays in
 * Python, and the per-linear host cost of the old Python chain is gone.
 *
 * Not here by design: the autocast region and recipe/format policy
 * (astrai/extension/quantize.py), the module slot table
 * (astrai/extension/policy/quantization/autocast.py), and the rings / cast caches / snapshot
 * (fp8_ring.h, fp8_cache.h, fp8_state.h). The state owner and checkpoint
 * interface live in fp8_runtime.cu in this module.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "fp8_linear.h"
#include "fp8_state.h"
#include <api/gemm.h>
#include <api/quantize.h>

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
    /*
     * The GEMMs' dequant scales. ``sx`` is a ring *view* (no clone) that dies at
     * that ring's next fold, so it is read only within this call and the backward
     * after it; ``sw`` is a ring view or, on a cast-cache hit, the cache's
     * immutable copy — either way it must describe ``w8``'s bytes.
     */
    Tensor sx, sw;
    Tensor x8T, w8T; // K-contiguous transposed casts (undefined when absent)
};

/*
 * The recipe as resolved for one call; the policy layer reads its region config
 * once and passes the fields down.
 */
struct Fp8Cfg {
    bool dynamic = false;
    int64_t history_len = 16;
    int64_t margin = 0;
    at::ScalarType fmt_a = at::kFloat8_e4m3fn;
    at::ScalarType fmt_b = at::kFloat8_e5m2;
};

/*
 * One quantize pass through the shared launcher, with the counters tests and
 * benches read. ``pub_scale``/``pub_recip`` redirect the fold's publication into
 * the double buffer's other pair; undefined with no ring.
 */
quant::QuantizeOutputs run_quant(const Tensor& t,
                                 const Tensor& scale,
                                 quant::QuantLayout layout,
                                 at::ScalarType fmt_a,
                                 c10::optional<at::ScalarType> fmt_b,
                                 const c10::optional<Tensor>& ring,
                                 int64_t idx,
                                 const Fp8Cfg& cfg,
                                 const Tensor& pub_scale = Tensor(),
                                 const Tensor& pub_recip = Tensor()) {
    state().n_quantize.fetch_add(1, std::memory_order_relaxed);
    return quant::run_quantize(
        t, scale, layout, fmt_a, fmt_b, ring, idx, fp8_max_of(fmt_a),
        std::pow(2.0, static_cast<double>(cfg.margin)),
        pub_scale.defined() ? c10::optional<Tensor>(pub_scale) : c10::nullopt,
        pub_recip.defined() ? c10::optional<Tensor>(pub_recip) : c10::nullopt,
        /*
         * The composed ring's state trails a double-buffered scale pair, so the
         * history length is stated, never derived from numel.
         */
        cfg.history_len);
}

Tensor run_gemm(const Tensor& a,
                const Tensor& b,
                const c10::optional<Tensor>& a_scale,
                const c10::optional<Tensor>& b_scale,
                const c10::optional<Tensor>& bias,
                bool trans_b) {
    state().n_gemm.fetch_add(1, std::memory_order_relaxed);
    return gemm::quant_gemm_impl(a, b, a_scale, b_scale, false, trans_b, bias);
}

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
    Fp8Cfg cfg{dynamic, history_len, margin, fmt_a, fmt_b};
    Fp8FwdOut res;
    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end() - 1);
    out_shape.push_back(w.size(0));

    if (dynamic) {
        // Current-amax scaling: measure, then quantize — no rings, no cache.
        const auto dyn = [&](const Tensor& t) {
            return scale_from_amax(amax_of(t), fmt_a, margin);
        };
        res.sx = dyn(x.reshape({-1, w.size(1)}));
        res.sw = dyn(w);
        const auto qx = run_quant(x, res.sx.reciprocal(), quant::QuantLayout::RowMajor, fmt_a,
                                  c10::nullopt, c10::nullopt, 0, cfg);
        const Tensor w8 = is_fp8(w.scalar_type())
                              ? w
                              : run_quant(w, res.sw.reciprocal(), quant::QuantLayout::RowMajor,
                                          fmt_a, c10::nullopt, c10::nullopt, 0, cfg)
                                    .out;
        res.out = run_gemm(qx.out.reshape({-1, qx.out.size(-1)}), w8, res.sx, res.sw, bias, true)
                      .reshape(out_shape);
        return res;
    }

    State& st = state();
    /*
     * Slot-addressed when the dispatcher knows the module (``slot >= 0``): the
     * meta then belongs to the module path and survives a replaced weight
     * parameter. Bare calls, benches and every existing test pass no slot and
     * keep the address-keyed lookup byte for byte.
     */
    auto meta = get_meta(w, history_len, margin, false, slot);
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
    /*
     * Pre-quantized fp8 weights: this branch takes w as-is, but the entry check
     * admits bf16 only — and the ring seed off fp8 values is not the scale the
     * weight was quantized with. Enabling it needs an explicit ``w_scale``
     * argument, not a relaxed check.
     */
    const bool w_pre = is_fp8(w.scalar_type());

    /*
     * Ring view of the current scale pair: the fold publishes into the other
     * pair, so this slot still holds the multiplier the cast used when the GEMM
     * reads it. It dies at this ring's next fold — consumed well before that in
     * training order (the backward precedes the same linear's next forward).
     */
    res.sx = meta->x->scale();
    Tensor w8;
    if (!w_pre && meta->cast.valid(w, fmt_a, fmt_b, st.generation)) {
        st.n_cast_hit.fetch_add(1, std::memory_order_relaxed);
        w8 = meta->cast.w8;
        res.sw = meta->cast.sw;
        res.w8T = meta->cast.w8T;
    } else {
        st.n_cast_miss.fetch_add(1, std::memory_order_relaxed);
        res.sw = meta->w.scale();
        if (w_pre) {
            w8 = w;
        } else if (update_rings) {
            const auto qw = run_quant(w, meta->w.scale_recip(), quant::QuantLayout::Dual, fmt_a,
                                      fmt_b, meta->w.state, meta->w.idx, cfg, meta->w.pub_scale(),
                                      meta->w.pub_recip());
            w8 = qw.out;
            res.w8T = qw.out_t;
            meta->w.advance();
            // The entry must outlive the ring pair it came from: immutable copy.
            meta->cast.fill(w, fmt_a, fmt_b, st.generation, w8, res.w8T, res.sw.clone());
        } else {
            /*
             * Ring-free cast (no-grad): folding here would advance the window a
             * second time per step and desynchronize the recompute.
             */
            w8 = run_quant(w, meta->w.scale_recip(), quant::QuantLayout::RowMajor, fmt_a,
                           c10::nullopt, c10::nullopt, 0, cfg)
                     .out;
        }
    }

    Tensor x8;
    if (update_rings) {
        if (cached_activation.has_value()) {
            st.n_act_hit.fetch_add(1, std::memory_order_relaxed);
            x8 = cached_activation->x8;
            res.x8T = cached_activation->x8T;
        } else {
            if (cache_activation)
                st.n_act_miss.fetch_add(1, std::memory_order_relaxed);
            const auto qx = run_quant(x, meta->x->scale_recip(), quant::QuantLayout::Dual, fmt_a,
                                      fmt_b, meta->x->state, meta->x->idx, cfg,
                                      meta->x->pub_scale(), meta->x->pub_recip());
            x8 = qx.out;
            res.x8T = qx.out_t;
            meta->x->advance();
            if (cache_activation) {
                st.act_cache.insert(x, fmt_a, fmt_b, history_len, margin, qx.out, qx.out_t,
                                    meta->x);
            }
        }
    } else {
        x8 = run_quant(x, meta->x->scale_recip(), quant::QuantLayout::RowMajor, fmt_a, c10::nullopt,
                       c10::nullopt, 0, cfg)
                 .out;
    }
    res.out =
        run_gemm(x8.reshape({-1, x8.size(-1)}), w8, res.sx, res.sw, bias, true).reshape(out_shape);
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
    const Fp8Cfg cfg{in.dynamic, in.history_len, in.margin, in.fmt_b, in.fmt_b};
    const Tensor g2 = g.reshape({-1, g.size(-1)});
    Tensor sx = in.sx, sw = in.sw, sg;
    c10::optional<Tensor> g_ring = c10::nullopt;
    int64_t g_idx = 0;
    std::shared_ptr<Fp8Meta> meta;
    if (in.dynamic) {
        const auto dyn = [&](const Tensor& t) {
            return scale_from_amax(amax_of(t), in.fmt_b, in.margin);
        };
        sg = dyn(g2);
        sw = dyn(in.w);
        sx = dyn(in.x);
    } else {
        meta = get_meta(in.w, in.history_len, in.margin, false);
        if (!meta->g.initialized)
            meta->g.seed(g2, in.fmt_b, in.margin);
        sg = meta->g.scale(); // ring view: g's fold publishes the other pair
        g_ring = meta->g.state;
        g_idx = meta->g.idx;
    }
    /*
     * NT fast path via transposed quantize outputs: g8 + w8T gives grad_x, g8T +
     * x8T gives grad_w. g is consumed in both orientations (one dual pass feeds
     * both); x8T/w8T came from the forward or the weight cast cache, so the
     * backward re-reads neither x nor w.
     */
    const bool delayed = !in.dynamic;
    const auto qg = run_quant(g2, delayed ? meta->g.scale_recip() : sg.reciprocal(),
                              quant::QuantLayout::Dual, in.fmt_b, c10::nullopt, g_ring, g_idx, cfg,
                              delayed ? meta->g.pub_scale() : Tensor(),
                              delayed ? meta->g.pub_recip() : Tensor());
    Tensor x8T = in.x8T;
    if (!x8T.defined()) {
        x8T =
            run_quant(in.x.reshape({-1, in.x.size(-1)}), sx.reciprocal(),
                      quant::QuantLayout::Transposed, in.fmt_b, c10::nullopt, c10::nullopt, 0, cfg)
                .out_t;
    }
    Tensor grad_x;
    if (is_fp8(in.w.scalar_type())) {
        /*
         * Pre-quantized weight has no transposed copy (grad_w unaffected).
         * Unreachable through the entry check today — see w_pre above.
         */
        grad_x = run_gemm(qg.out, in.w, sg, sw, c10::nullopt, false).reshape(in.x.sizes());
    } else {
        Tensor w8T = in.w8T;
        if (!w8T.defined()) {
            w8T = run_quant(in.w, sw.reciprocal(), quant::QuantLayout::Transposed, in.fmt_b,
                            c10::nullopt, c10::nullopt, 0, cfg)
                      .out_t;
        }
        grad_x = run_gemm(qg.out, w8T, sg, sw, c10::nullopt, true).reshape(in.x.sizes());
    }
    Tensor grad_w = run_gemm(qg.out_t, x8T, sg, sx, c10::nullopt, true);
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
