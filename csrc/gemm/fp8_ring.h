#pragma once
/* FP8 scale recipe and delayed-scaling ring. */
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

#include <api/dtype.h>
#include <api/quantize_common.h>

namespace astrai {
namespace fp8 {

using torch::Tensor;

/*
 * Recipe constants
 */

/*
 * The fp8 format range: one source for the host-side seed formula (the kernel
 * publishes with the same number, passed through QuantParams).
 */
inline float fp8_max_of(at::ScalarType fmt) {
    TORCH_CHECK(fmt == scalar_type_v<fp8_e4m3> || fmt == scalar_type_v<fp8_e5m2>,
                "fp8 linear: format must be float8_e4m3fn or float8_e5m2");
    return fmt == scalar_type_v<fp8_e4m3> ? ElemTrait<fp8_e4m3>::kFiniteMax
                                          : ElemTrait<fp8_e5m2>::kFiniteMax;
}

inline bool is_fp8(at::ScalarType dt) {
    return dt == scalar_type_v<fp8_e4m3> || dt == scalar_type_v<fp8_e5m2>;
}

/*
 * scale = (peak / fp8_max) / 2^margin, clamped — the host mirror of the
 * kernel's publish, used only where the host owns it (seed, dynamic recipe).
 */
inline Tensor scale_from_amax(const Tensor& window_or_amax, at::ScalarType fmt, int64_t margin) {
    const Tensor peak = window_or_amax.max();
    const double pow2 = std::pow(2.0, static_cast<double>(margin));
    return (peak / fp8_max_of(fmt) / pow2).clamp_min(1e-12);
}

/*
 * Raw-domain and detached: the rings fill their history with it, and an
 * in-place op on a non-grad buffer must not drag a grad graph in.
 */
inline Tensor amax_of(const Tensor& t) {
    return t.detach().abs().amax().to(at::kFloat).clamp_min(1e-12);
}

/*
 * Delayed-scaling rings
 */

/*
 * One operand's ring with a double-buffered scale pair (offsets: RingLayout).
 * The fold reads recip[cur] and publishes into pair[1-cur], so this step's GEMMs
 * keep reading pair[cur] and the host needs no snapshot clone; advance() flips
 * cur once per fold.
 */
struct ScaleRing {
    Tensor state;
    Tensor hist;
    Tensor pscale[2];
    Tensor precip[2];
    int64_t idx = 0;
    int cur = 0; // the pair this step reads; the fold publishes 1-cur
    bool initialized = false;

    ScaleRing() = default;

    ScaleRing(const torch::TensorOptions& opts, int64_t history_len) {
        const int64_t n = history_len;
        TORCH_CHECK(n > 0, "fp8 linear: history_len must be positive");
        const quant::RingLayout layout{n};
        state = torch::zeros({layout.size()}, opts);
        hist = state.narrow(0, 0, n);
        pscale[0] = state.narrow(0, layout.scale(0), 1);
        precip[0] = state.narrow(0, layout.recip(0), 1);
        pscale[1] = state.narrow(0, layout.scale(1), 1);
        precip[1] = state.narrow(0, layout.recip(1), 1);
    }

    const Tensor& scale() const { return pscale[cur]; }
    const Tensor& scale_recip() const { return precip[cur]; }
    const Tensor& pub_scale() const { return pscale[cur ^ 1]; }
    const Tensor& pub_recip() const { return precip[cur ^ 1]; }

    void advance() {
        idx = (idx + 1) % hist.numel();
        cur ^= 1; // the just-published pair becomes the next step's current
    }

    // Seed / restore only: the fold republishes both slots in-kernel each step.
    void publish_recip() { at::reciprocal_out(precip[cur], pscale[cur]); }

    void seed(const Tensor& t, at::ScalarType fmt, int64_t margin) {
        const Tensor amax = amax_of(t);
        hist.fill_(amax);
        cur = 0;
        pscale[0].copy_(scale_from_amax(hist, fmt, margin));
        publish_recip();
        initialized = true;
    }
};

} // namespace fp8
} // namespace astrai
