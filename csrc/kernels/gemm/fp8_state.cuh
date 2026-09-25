#pragma once
// The fp8 training state machine: delayed-scaling rings (double-buffered scale
// pairs), the weight/activation cast caches, the per-weight meta registry and
// its checkpoint snapshot — everything the composed linear keeps between calls.
// Split out for readability only; every function is ``inline`` so all includers
// share one registry (an anonymous namespace would give a copy per TU).
//
// A meta is addressed by its module's *slot* (a path, published by
// astrai/extension/fp8_slots.py) or by the weight's (data_ptr, shape, dtype);
// the slot survives a replaced parameter, the address does not. Snapshots bind
// slotted entries by name, others by registration order. Ring offsets:
// RingLayout (quantize/common.h).

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

// scale = (peak / fp8_max) / 2^margin, clamped — the host mirror of the
// kernel's publish, used only where the host owns it (seed, dynamic recipe).
inline Tensor scale_from_amax(const Tensor& window_or_amax, at::ScalarType fmt,
                              int64_t margin) {
    const Tensor peak = window_or_amax.max();
    const double pow2 = std::pow(2.0, static_cast<double>(margin));
    return (peak / fp8_max_of(fmt) / pow2).clamp_min(1e-12);
}

// Raw-domain and detached: the rings fill their history with it, and an
// in-place op on a non-grad buffer must not drag a grad graph in.
inline Tensor amax_of(const Tensor& t) {
    return t.detach().abs().amax().to(at::kFloat).clamp_min(1e-12);
}

// ---------------------------------------------------------------------------
// Delayed-scaling rings
// ---------------------------------------------------------------------------

// One operand's ring with a double-buffered scale pair (offsets: RingLayout).
// The fold reads recip[cur] and publishes into pair[1-cur], so this step's GEMMs
// keep reading pair[cur] and the host needs no snapshot clone; advance() flips
// cur once per fold.
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

// The cast cache's validity key: every in-place update bumps it (optimizer
// steps included).
inline int64_t version_of(const Tensor& t) {
    return t.unsafeGetTensorImpl()->version_counter().current_version();
}

// Version-keyed weight cast cache: the fp8 pair plus the scale they were cast
// with, so the GEMM always dequantizes with the very scale used. A hit also
// skips the amax fold and the ring advance — an unchanged weight has an
// unchanged amax, so the ring tracks optimizer steps instead of calls.
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

// One quantized activation, kept so the next consumer of the same tensor skips
// the cast (q/k/v share an input, up/gate another). Identity is a *strong*
// anchor plus the version counter: an activation's address is recycled almost
// immediately, so a bare ``data_ptr`` would match a different tensor. No scale
// snapshot — a hit reads the shared ring's current pair, so anything advancing
// that ring between cast and consumer would decouple the two.
struct ActivationCast {
    c10::intrusive_ptr<c10::TensorImpl> anchor;
    int64_t version = -1;
    at::ScalarType fmt_a = at::kFloat8_e4m3fn;
    at::ScalarType fmt_b = at::kFloat8_e5m2;
    int64_t history_len = 0;
    int64_t margin = 0;
    int device = -1;
    int64_t bytes = 0;
    uint64_t last_use = 0;
    Tensor x8, x8T;
    std::shared_ptr<ScaleRing> ring;

    bool matches(const Tensor& x, at::ScalarType fa, at::ScalarType fb,
                 int64_t hist_len, int64_t scale_margin) const {
        return anchor.get() == x.unsafeGetTensorImpl() &&
               version == version_of(x) && fmt_a == fa && fmt_b == fb &&
               history_len == hist_len && margin == scale_margin &&
               device == x.device().index();
    }
};

struct Fp8Meta {
    ScaleRing w;
    // Shared with the activation cache: every consumer of one tensor must fold
    // once and read the one published scale.
    std::shared_ptr<ScaleRing> x = std::make_shared<ScaleRing>();
    ScaleRing g;
    WeightCast cast;
    // Weak on purpose: it keeps a dead TensorImpl's address from being recycled
    // (so the pointer comparison stays an identity) and its expiry *is* the
    // eviction signal — a strong ref would pin the replaced weight.
    void* weight_ptr = nullptr;
    c10::weak_intrusive_ptr<c10::TensorImpl> anchor;
    std::string key;             // this meta's registry key (eviction path)
    std::vector<int64_t> shape;  // the owning weight's (snapshot key)
    at::ScalarType dtype = at::kBFloat16;
    int64_t history_len = 0;
    int64_t margin = 0;
    bool dynamic = false;

    // Slot addressing (fp8_slots.py): the state belongs to the *module*, so a
    // replaced weight parameter re-points the identity and keeps the rings.
    // ``slot_name`` is the snapshot key, ``role`` the Python-side policy glob;
    // all three stay empty for an unslotted meta (bare call, bench).
    int64_t slot = -1;
    std::string slot_name;
    std::string role;

    // A weak pointer has no empty state and a meta always has an owner, which
    // must be live.
    explicit Fp8Meta(const Tensor& owner)
        : weight_ptr(owner.data_ptr()), anchor(owner.getIntrusivePtr()) {}

    bool owns(const Tensor& w) const {
        return anchor.lock().get() != nullptr && weight_ptr == w.data_ptr();
    }
    bool alive() const { return anchor.lock().get() != nullptr; }
};

// LRU over at most 8 entries / 64 MiB (source tensor plus both fp8 outputs).
// Bounded because entries hold strong references, and cleared by the caller —
// autocast exit, fp8_reset, fp8_set_act_cache(false) — so anchors never outlive
// the region that produced them.
struct ActivationCache {
    static constexpr size_t kMaxEntries = 8;
    static constexpr int64_t kMaxBytes = 64LL << 20;

    std::mutex mu;
    std::vector<ActivationCast> entries;
    int64_t bytes = 0;
    uint64_t clock = 0;

    c10::optional<ActivationCast> find(const Tensor& x, at::ScalarType fa,
                                       at::ScalarType fb, int64_t history_len,
                                       int64_t margin) {
        std::lock_guard<std::mutex> lock(mu);
        for (auto& entry : entries) {
            if (entry.matches(x, fa, fb, history_len, margin)) {
                entry.last_use = ++clock;
                return entry;
            }
        }
        return c10::nullopt;
    }

    void insert(const Tensor& x, at::ScalarType fa, at::ScalarType fb,
                int64_t history_len, int64_t margin, Tensor row, Tensor transposed,
                std::shared_ptr<ScaleRing> ring) {
        const int64_t input_bytes = x.numel() * x.element_size();
        const int64_t output_bytes =
            row.numel() * row.element_size() +
            transposed.numel() * transposed.element_size();
        const int64_t entry_bytes = input_bytes + output_bytes;
        if (entry_bytes > kMaxBytes) return;

        std::lock_guard<std::mutex> lock(mu);
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->anchor.get() == x.unsafeGetTensorImpl()) {
                bytes -= it->bytes;
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
        while (!entries.empty() &&
               (entries.size() >= kMaxEntries || bytes + entry_bytes > kMaxBytes)) {
            auto oldest = std::min_element(
                entries.begin(), entries.end(),
                [](const ActivationCast& a, const ActivationCast& b) {
                    return a.last_use < b.last_use;
                });
            bytes -= oldest->bytes;
            entries.erase(oldest);
        }
        ActivationCast entry;
        entry.anchor = x.getIntrusivePtr();
        entry.version = version_of(x);
        entry.fmt_a = fa;
        entry.fmt_b = fb;
        entry.history_len = history_len;
        entry.margin = margin;
        entry.device = x.device().index();
        entry.bytes = entry_bytes;
        entry.last_use = ++clock;
        entry.x8 = std::move(row);
        entry.x8T = std::move(transposed);
        entry.ring = std::move(ring);
        entries.push_back(std::move(entry));
        bytes += entry_bytes;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mu);
        return entries.size();
    }

    int64_t byte_size() {
        std::lock_guard<std::mutex> lock(mu);
        return bytes;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu);
        entries.clear();
        bytes = 0;
    }
};

// ---------------------------------------------------------------------------
// Process-wide state: the meta registry, the generation counter, the snapshot.
// ---------------------------------------------------------------------------

struct SlotInfo {
    std::string name;  // module path — the snapshot's stable key
    std::string role;  // Python-side glob (``layers.*.mlp.up``)
};

struct State {
    std::mutex mu;
    std::unordered_map<std::string, std::shared_ptr<Fp8Meta>> by_key;
    std::unordered_map<int64_t, std::shared_ptr<Fp8Meta>> by_slot;
    // Slot id -> (module path, role), published once by fp8_slots.py. Metadata,
    // not training state: fp8_reset leaves it alone, a later fp8_set_slots
    // replaces the lot.
    std::unordered_map<int64_t, SlotInfo> slots;
    std::vector<std::shared_ptr<Fp8Meta>> order;  // registration order (A1)
    ActivationCache act_cache;
    std::atomic<bool> act_cache_enabled{true};
    std::atomic<int64_t> n_act_hit{0};
    std::atomic<int64_t> n_act_miss{0};

    // Snapshot entries queued by load_state_dict, consumed by slot name or —
    // unslotted — by (shape, dtype) in this order. Unmatched entries stay
    // queued: a model-shape change across the checkpoint just re-seeds.
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

// Drop a meta from every registry view — a leftover in ``order`` would shift the
// fallback binding of later unslotted metas. The identity checks matter: a key is
// claimed by only one meta (first ``emplace`` wins), so never erase another's.
inline void drop_meta(State& st, const std::shared_ptr<Fp8Meta>& meta) {
    auto it = st.by_key.find(meta->key);
    if (it != st.by_key.end() && it->second == meta) st.by_key.erase(it);
    if (meta->slot >= 0) {
        auto sit = st.by_slot.find(meta->slot);
        if (sit != st.by_slot.end() && sit->second == meta)
            st.by_slot.erase(sit);
    }
    st.order.erase(std::remove(st.order.begin(), st.order.end(), meta),
                   st.order.end());
}

// Re-point a slot's meta at a new weight tensor, keeping the rings: TP/FSDP swap
// the Parameter, not the module. The cast cache must still go — it keys on the
// version counter, and a fresh parameter starts at a version its predecessor may
// have been cast at. A kept ring means the first fold after the swap still uses
// the previous scale (a wild magnitude change clips for one step).
inline void rebind_meta_locked(const std::shared_ptr<Fp8Meta>& meta,
                               const Tensor& w) {
    State& st = state();
    auto it = st.by_key.find(meta->key);
    if (it != st.by_key.end() && it->second == meta) st.by_key.erase(it);
    meta->key = meta_key(w);
    meta->weight_ptr = w.data_ptr();
    meta->anchor = c10::weak_intrusive_ptr<c10::TensorImpl>(w.getIntrusivePtr());
    meta->shape.assign(w.sizes().begin(), w.sizes().end());
    meta->dtype = w.scalar_type();
    meta->cast = WeightCast{};
    st.by_key.emplace(meta->key, meta);
}

// The snapshot's dtype field uses Python's ``str(torch.dtype)`` spelling — the
// format the checkpoint bridge established, so either side restores the other's.
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

// Restore one ring. False geometry = the buffer changed since the save (recipe
// change across the checkpoint): the ring stays fresh and re-seeds on next use.
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

// Consume queued entries: a named entry binds only to the module that owns the
// name (so two same-shaped linears cannot swap state), unnamed ones fall back to
// (shape, dtype) in queue order — data_ptr is meaningless across processes.
inline void restore_pending_locked(std::shared_ptr<Fp8Meta>& meta) {
    State& st = state();
    if (!meta->slot_name.empty()) {
        for (size_t i = 0; i < st.pending.size(); ++i) {
            const py::dict entry = st.pending[i];
            if (!entry.contains("slot_name")) continue;
            if (py::cast<std::string>(entry["slot_name"]) != meta->slot_name)
                continue;
            bool geometry_ok = true;
            restore_ring(meta->w, entry["w"], geometry_ok);
            restore_ring(*meta->x, entry["x"], geometry_ok);
            restore_ring(meta->g, entry["g"], geometry_ok);
            st.pending.erase(st.pending.begin() + static_cast<long>(i));
            return;
        }
    }
    for (size_t i = 0; i < st.pending.size(); ++i) {
        const py::dict entry = st.pending[i];
        // A named entry waits for its module: binding it here by shape would
        // reintroduce exactly the cross-wiring the names exist to prevent.
        if (entry.contains("slot_name") &&
            !py::cast<std::string>(entry["slot_name"]).empty())
            continue;
        bool match = py::len(entry["shape"]) == meta->shape.size();
        for (size_t d = 0; match && d < meta->shape.size(); ++d)
            match = py::cast<int64_t>(entry["shape"][py::int_(d)]) ==
                    meta->shape[d];
        match = match && dtype_str_matches(
                             py::cast<std::string>(entry["dtype"]), meta->dtype);
        if (!match) continue;
        bool geometry_ok = true;
        restore_ring(meta->w, entry["w"], geometry_ok);
        restore_ring(*meta->x, entry["x"], geometry_ok);
        restore_ring(meta->g, entry["g"], geometry_ok);
        st.pending.erase(st.pending.begin() + static_cast<long>(i));
        return;
    }
}

// Find or create a weight's rings. A slot survives a replaced weight tensor —
// only the identity is re-pointed (rebind_meta_locked). A recipe change, or a
// failed ownership check on the address path, rebuilds with a generation bump
// that invalidates the cast caches; fresh rings re-seed on the next forward.
// Without a slot the address-keyed path runs unchanged (bare calls, benches, and
// the backward pass, which reaches its meta through the weight forward saved).
inline std::shared_ptr<Fp8Meta> get_meta(const Tensor& w, int64_t history_len,
                                         int64_t margin, bool dynamic,
                                         int64_t slot = -1,
                                         const std::string& slot_name = "",
                                         const std::string& role = "") {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    // The published slot table wins; the arguments cover tests that want a name
    // without registering one.
    std::string name = slot_name, role_glob = role;
    if (slot >= 0) {
        auto rit = st.slots.find(slot);
        if (rit != st.slots.end()) {
            name = rit->second.name;
            role_glob = rit->second.role;
        }
    }
    const auto recipe_ok = [&](const std::shared_ptr<Fp8Meta>& meta) {
        return meta->history_len == history_len && meta->margin == margin &&
               meta->dynamic == dynamic;
    };
    if (slot >= 0) {
        auto sit = st.by_slot.find(slot);
        if (sit != st.by_slot.end()) {
            auto meta = sit->second;
            if (recipe_ok(meta)) {
                if (!meta->owns(w)) rebind_meta_locked(meta, w);
                meta->slot_name = name;
                meta->role = role_glob;
                return meta;
            }
            drop_meta(st, meta);
            st.generation += 1;
        }
    }
    const std::string key = meta_key(w);
    auto it = st.by_key.find(key);
    if (it != st.by_key.end()) {
        auto meta = it->second;
        if (recipe_ok(meta) && meta->owns(w)) {
            // A weight first reached bare (address path) adopts the slot here.
            if (slot >= 0) {
                if (meta->slot < 0) {
                    meta->slot = slot;
                    st.by_slot.emplace(slot, meta);
                }
                meta->slot_name = name;
                meta->role = role_glob;
            }
            return meta;
        }
        drop_meta(st, meta);
        st.generation += 1;
    }
    const auto opts = torch::TensorOptions().dtype(at::kFloat).device(
        w.device());
    auto meta = std::make_shared<Fp8Meta>(w);
    meta->key = key;
    meta->slot = slot;
    meta->slot_name = name;
    meta->role = role_glob;
    meta->w = ScaleRing(opts, history_len);
    meta->x = std::make_shared<ScaleRing>(opts, history_len);
    meta->g = ScaleRing(opts, history_len);
    meta->shape.assign(w.sizes().begin(), w.sizes().end());
    meta->dtype = w.scalar_type();
    meta->history_len = history_len;
    meta->margin = margin;
    meta->dynamic = dynamic;
    st.by_key.emplace(key, meta);
    if (slot >= 0) st.by_slot.emplace(slot, meta);
    st.order.push_back(meta);
    restore_pending_locked(meta);
    return meta;
}

}  // namespace fp8
}  // namespace astrai
