#pragma once
// The fp8 training state machine: the delayed-scaling rings (double-buffered
// scale pairs), the version-keyed weight cast cache, the per-weight meta
// registry and its checkpoint snapshot/restore — everything the composed
// linear (fp8_linear.cu) keeps between calls. Split out for readability only;
// every function is ``inline``, so a second includer would still share one
// registry per module (an anonymous namespace would give a silent copy per TU).
//
// Snapshots bind in registration order (data_ptr is meaningless across
// processes); the ring's slot offsets are RingLayout's (quantize/common.h).

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/TensorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
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

#include "quantize/launch.cuh"

namespace astrai {
namespace fp8 {

using torch::Tensor;

// ---------------------------------------------------------------------------
// Recipe constants
// ---------------------------------------------------------------------------

// The fp8 format range: one source for the host-side seed formula (the kernel
// publishes with the same number, passed through QuantParams).
inline float fp8_max_of(at::ScalarType fmt) {
    TORCH_CHECK(fmt == at::kFloat8_e4m3fn || fmt == at::kFloat8_e5m2,
                "fp8 linear: format must be float8_e4m3fn or float8_e5m2");
    return fmt == at::kFloat8_e4m3fn ? 448.0f : 57344.0f;
}

inline bool is_fp8(at::ScalarType dt) {
    return dt == at::kFloat8_e4m3fn || dt == at::kFloat8_e5m2;
}

// scale = (peak / fp8_max) / 2^margin, clamped, with peak the max over the
// amax window — the host mirror of the kernel's publish, used only where the
// host owns the publication (seed and the dynamic recipe).
inline Tensor scale_from_amax(const Tensor& window_or_amax, at::ScalarType fmt,
                              int64_t margin) {
    const Tensor peak = window_or_amax.max();
    const double pow2 = std::pow(2.0, static_cast<double>(margin));
    return (peak / fp8_max_of(fmt) / pow2).clamp_min(1e-12);
}

// amax in the raw domain, detached: the rings fill their history with it, and
// an in-place op on a non-grad buffer must not drag a grad graph in.
inline Tensor amax_of(const Tensor& t) {
    return t.detach().abs().amax().to(at::kFloat).clamp_min(1e-12);
}

// ---------------------------------------------------------------------------
// Delayed-scaling rings
// ---------------------------------------------------------------------------

// One operand's ring, with a double-buffered scale pair. Offsets are
// RingLayout's; layout [hist n | scale0 | recip0 | amax | done | scratch |
// scale1 | recip1]. The fold reads recip[cur] and publishes into pair[1-cur],
// so this step's GEMMs keep reading pair[cur] and the host needs no snapshot
// clone. ``cur`` flips once per fold (advance()); pair 0 keeps the legacy
// offsets so the seed path stays put.
struct ScaleRing {
    Tensor state;
    Tensor hist;
    Tensor pscale[2];
    Tensor precip[2];
    int64_t idx = 0;
    int cur = 0;  // the pair this step reads; the fold publishes 1-cur
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
        cur ^= 1;  // the just-published pair becomes the next step's current
    }

    // Mirrors the scale's reciprocal into its slot — seed / restore only; the
    // fold republishes both slots in-kernel every step.
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

// A tensor's autograd version counter: the cast cache's validity key (every
// in-place update bumps it, optimizer steps included).
inline int64_t version_of(const Tensor& t) {
    return t.unsafeGetTensorImpl()->version_counter().current_version();
}

// Version-keyed weight cast cache: the fp8 pair together with the scale they
// were cast with, so the entry is self-consistent — the GEMM dequant reads
// the very scale the values were quantized with, whatever the ring has
// published since. A hit also skips the in-kernel amax fold and the ring
// advance: an unchanged weight has an unchanged amax, so the fold would
// rewrite the history window with the same value. The ring therefore tracks
// optimizer steps — the only steps that can move a weight's amax.
struct WeightCast {
    int64_t version = -1;
    int64_t generation = -1;
    at::ScalarType fmt_a = at::kFloat8_e4m3fn;
    at::ScalarType fmt_b = at::kFloat8_e4m3fn;
    Tensor w8, w8T, sw;

    bool valid(const Tensor& w, at::ScalarType fa, at::ScalarType fb,
               int64_t gen) const {
        return w8.defined() && version == version_of(w) && generation == gen &&
               fmt_a == fa && fmt_b == fb;
    }

    void fill(const Tensor& w, at::ScalarType fa, at::ScalarType fb,
              int64_t gen, Tensor a, Tensor b, Tensor s) {
        version = version_of(w);
        generation = gen;
        fmt_a = fa;
        fmt_b = fb;
        w8 = std::move(a);
        w8T = std::move(b);
        sw = std::move(s);
    }
};

struct Fp8Meta {
    ScaleRing w, x, g;
    WeightCast cast;
    // The weight this meta is for. ``weight_ptr`` is the key's address;
    // ``anchor`` is a *weak* handle that keeps a dead TensorImpl (and its
    // Storage) from being freed, so while it locks the recorded address cannot
    // have been recycled — which makes the address comparison an identity, and
    // an unlocked anchor the sound eviction signal. A strong reference would
    // pin the replaced weight instead of dropping the entry.
    void* weight_ptr = nullptr;
    c10::weak_intrusive_ptr<c10::TensorImpl> anchor;
    std::string key;             // this meta's registry key (eviction path)
    std::vector<int64_t> shape;  // the owning weight's (snapshot key)
    at::ScalarType dtype = at::kBFloat16;
    int64_t history_len = 0;
    int64_t margin = 0;
    bool dynamic = false;

    // A weak pointer has no empty state (its null singleton is a real target),
    // and a meta always belongs to a weight. ``owner`` must be live.
    explicit Fp8Meta(const Tensor& owner)
        : weight_ptr(owner.data_ptr()), anchor(owner.getIntrusivePtr()) {}

    bool owns(const Tensor& w) const {
        return anchor.lock().get() != nullptr && weight_ptr == w.data_ptr();
    }
    bool alive() const { return anchor.lock().get() != nullptr; }
};

// ---------------------------------------------------------------------------
// Process-wide state: the meta registry, the generation counter and the
// checkpoint snapshot. The key is the weight's (data_ptr, shape, dtype); the
// lookup also requires the meta's anchor to own that buffer (Fp8Meta::owns).
// ---------------------------------------------------------------------------

struct State {
    std::mutex mu;
    std::unordered_map<std::string, std::shared_ptr<Fp8Meta>> by_key;
    std::vector<std::shared_ptr<Fp8Meta>> order;  // registration order (A1)
    // Snapshots queued by load_state_dict, consumed in registration order as
    // the metas they belong to are restored (a resume restores before any
    // forward runs).
    std::vector<py::dict> pending;
    int64_t generation = 0;  // bumped by restore / reset / recipe rebuild
    // Test/bench observability.
    std::atomic<int64_t> n_quantize{0};
    std::atomic<int64_t> n_gemm{0};
    std::atomic<int64_t> n_cast_hit{0};
    std::atomic<int64_t> n_cast_miss{0};
};

inline State& state() {
    static State s;
    return s;
}

inline std::string meta_key(const Tensor& w) {
    std::string key = std::to_string(reinterpret_cast<uintptr_t>(w.data_ptr()));
    key += '|';
    for (const auto s : w.sizes()) {
        key += std::to_string(s);
        key += ',';
    }
    key += '|';
    key += std::to_string(static_cast<int>(w.scalar_type()));
    return key;
}

// Drop one meta from both registry views; a leftover in ``order`` would shift
// the snapshot's FIFO binding for every later linear.
inline void drop_meta(State& st, const std::shared_ptr<Fp8Meta>& meta) {
    st.by_key.erase(meta->key);
    st.order.erase(std::remove(st.order.begin(), st.order.end(), meta),
                   st.order.end());
}

// The snapshot's dtype field uses the Python ``str(torch.dtype)`` spelling —
// the format the checkpoint bridge established before this op existed, so a
// snapshot written by either side restores on the other.
inline std::string torch_dtype_str(at::ScalarType t) {
    switch (t) {
        case at::kBFloat16: return "torch.bfloat16";
        case at::kHalf: return "torch.float16";
        case at::kFloat: return "torch.float32";
        case at::kDouble: return "torch.float64";
        case at::kChar: return "torch.int8";
        case at::kByte: return "torch.uint8";
        case at::kShort: return "torch.int16";
        case at::kInt: return "torch.int32";
        case at::kLong: return "torch.int64";
        case at::kBool: return "torch.bool";
        case at::kFloat8_e4m3fn: return "torch.float8_e4m3fn";
        case at::kFloat8_e5m2: return "torch.float8_e5m2";
        default: return c10::toString(t);
    }
}

inline bool dtype_str_matches(const std::string& s, at::ScalarType t) {
    return s == torch_dtype_str(t) || s == std::string(c10::toString(t));
}

// Restore one ring from a snapshot entry. False geometry means the buffer
// changed since the save (a recipe change across the checkpoint boundary):
// the ring stays fresh and re-seeds on next use.
inline void restore_ring(ScaleRing& ring, const py::object& sd,
                         bool& geometry_ok) {
    if (sd.is_none()) return;
    py::dict d = sd.cast<py::dict>();
    Tensor saved = d["state"].cast<Tensor>();
    if (saved.numel() != ring.state.numel()) {
        geometry_ok = false;
        return;
    }
    ring.state.copy_(saved.to(ring.state.device()));
    ring.cur = d.contains("cur") ? py::cast<int>(d["cur"]) : 0;
    ring.publish_recip();  // snapshots older than the recip slot restore 0
    ring.idx = py::cast<int64_t>(d["idx"]);
    ring.initialized = py::cast<bool>(d["initialized"]);
}

// Consume queued restore entries in registration order, matching by
// (shape, dtype) — data_ptr is meaningless across processes, and re-binding
// relies on the same registration-order contract TE documents for amax
// reduction.
inline void restore_pending_locked(std::shared_ptr<Fp8Meta>& meta) {
    State& st = state();
    for (size_t i = 0; i < st.pending.size(); ++i) {
        const py::dict entry = st.pending[i];
        bool match = py::len(entry["shape"]) == meta->shape.size();
        for (size_t d = 0; match && d < meta->shape.size(); ++d)
            match = py::cast<int64_t>(entry["shape"][py::int_(d)]) ==
                    meta->shape[d];
        match = match && dtype_str_matches(
                             py::cast<std::string>(entry["dtype"]), meta->dtype);
        if (!match) continue;
        bool geometry_ok = true;
        restore_ring(meta->w, entry["w"], geometry_ok);
        restore_ring(meta->x, entry["x"], geometry_ok);
        restore_ring(meta->g, entry["g"], geometry_ok);
        st.pending.erase(st.pending.begin() + static_cast<long>(i));
        return;
    }
}

// Look up (or create) the rings for a weight. An entry is rebuilt — with a
// generation bump that invalidates the cast cache — when the recipe changed
// (stale geometry and fold constants) or when it does not own this buffer (the
// address was recycled after its weight died). Fresh rings re-seed on the next
// forward.
inline std::shared_ptr<Fp8Meta> get_meta(const Tensor& w, int64_t history_len,
                                         int64_t margin, bool dynamic) {
    const std::string key = meta_key(w);
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    auto it = st.by_key.find(key);
    if (it != st.by_key.end()) {
        auto meta = it->second;
        if (meta->history_len == history_len && meta->margin == margin &&
            meta->dynamic == dynamic && meta->owns(w)) {
            return meta;
        }
        drop_meta(st, meta);
        st.generation += 1;
    }
    const auto opts = torch::TensorOptions().dtype(at::kFloat).device(
        w.device());
    auto meta = std::make_shared<Fp8Meta>(w);
    meta->key = key;
    meta->w = ScaleRing(opts, history_len);
    meta->x = ScaleRing(opts, history_len);
    meta->g = ScaleRing(opts, history_len);
    meta->shape.assign(w.sizes().begin(), w.sizes().end());
    meta->dtype = w.scalar_type();
    meta->history_len = history_len;
    meta->margin = margin;
    meta->dynamic = dynamic;
    st.by_key.emplace(key, meta);
    st.order.push_back(meta);
    restore_pending_locked(meta);
    return meta;
}

}  // namespace fp8
}  // namespace astrai
