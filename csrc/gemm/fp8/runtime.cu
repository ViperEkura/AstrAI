/* FP8 process state owner, checkpoint interface, and Python bindings. */
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "linear.h"
#include "state.h"
#include <api/torch/dtype.h>

namespace astrai {
namespace fp8 {

State& state() {
    static State s;
    return s;
}

namespace {
std::string meta_key(const Tensor& w) {
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

/*
 * Drop a meta from every registry view — a leftover in ``order`` would shift the
 * fallback binding of later unslotted metas. The identity checks matter: a key is
 * claimed by only one meta (first ``emplace`` wins), so never erase another's.
 */
void drop_meta(State& st, const std::shared_ptr<Fp8Meta>& meta) {
    auto it = st.by_key.find(meta->key);
    if (it != st.by_key.end() && it->second == meta)
        st.by_key.erase(it);
    if (meta->slot >= 0) {
        auto sit = st.by_slot.find(meta->slot);
        if (sit != st.by_slot.end() && sit->second == meta)
            st.by_slot.erase(sit);
    }
    st.order.erase(std::remove(st.order.begin(), st.order.end(), meta), st.order.end());
}

/*
 * Keep the delayed scales when a module replaces its parameter.
 */
void rebind_meta_locked(const std::shared_ptr<Fp8Meta>& meta, const Tensor& w) {
    State& st = state();
    auto it = st.by_key.find(meta->key);
    if (it != st.by_key.end() && it->second == meta)
        st.by_key.erase(it);
    meta->key = meta_key(w);
    meta->weight_ptr = w.data_ptr();
    meta->anchor = c10::weak_intrusive_ptr<c10::TensorImpl>(w.getIntrusivePtr());
    meta->shape.assign(w.sizes().begin(), w.sizes().end());
    meta->dtype = w.scalar_type();
    meta->weight_version = -1;
    st.by_key.emplace(meta->key, meta);
}

/* A changed recipe leaves its fresh ring uninitialized for the next seed. */
void restore_ring(ScaleRing& ring, const c10::optional<RingSnapshot>& saved) {
    if (!saved.has_value() || saved->state.numel() != ring.state.numel())
        return;
    TORCH_CHECK(saved->idx < ring.hist.numel(), "snapshot ring idx out of range");
    ring.state.copy_(saved->state.to(ring.state.device()));
    ring.cur = saved->cur;
    ring.publish_recip();
    ring.idx = saved->idx;
    ring.initialized = saved->initialized;
}

void restore_pending_locked(const std::shared_ptr<Fp8Meta>& meta) {
    auto& pending = state().pending;
    const auto restore = [&](size_t i) {
        const auto& entry = pending[i];
        meta->weight_version = -1;
        restore_ring(meta->w, entry.w);
        restore_ring(*meta->x, entry.x);
        restore_ring(meta->g, entry.g);
        pending.erase(pending.begin() + static_cast<long>(i));
    };
    if (!meta->slot_name.empty()) {
        for (size_t i = 0; i < pending.size(); ++i) {
            if (pending[i].slot_name == meta->slot_name) {
                restore(i);
                return;
            }
        }
    }
    for (size_t i = 0; i < pending.size(); ++i) {
        const auto& entry = pending[i];
        if (!entry.slot_name.empty() || entry.shape != meta->shape ||
            !dtype_str_matches(entry.dtype, meta->dtype))
            continue;
        restore(i);
        return;
    }
}
} // namespace

/*
 * Find or create a weight's rings. A slot survives a replaced weight tensor —
 * only the identity is re-pointed (rebind_meta_locked). A recipe change, or a
 * failed ownership check on the address path, rebuilds the rings for the next
 * forward. Without a slot the address-keyed path runs unchanged (bare calls,
 * benches, and the backward pass, which reaches its meta through the weight
 * forward saved).
 */
std::shared_ptr<Fp8Meta>
get_meta(const Tensor& w, int64_t history_len, int64_t margin, int64_t slot) {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    /*
     * Slot names and roles come from the published module table.
     */
    std::string name, role_glob;
    if (slot >= 0) {
        auto rit = st.slots.find(slot);
        if (rit != st.slots.end()) {
            name = rit->second.name;
            role_glob = rit->second.role;
        }
    }
    const auto recipe_ok = [&](const std::shared_ptr<Fp8Meta>& meta) {
        return meta->w.hist.numel() == history_len && meta->margin == margin;
    };
    if (slot >= 0) {
        auto sit = st.by_slot.find(slot);
        if (sit != st.by_slot.end()) {
            auto meta = sit->second;
            if (recipe_ok(meta)) {
                if (!meta->owns(w))
                    rebind_meta_locked(meta, w);
                meta->slot_name = name;
                meta->role = role_glob;
                return meta;
            }
            drop_meta(st, meta);
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
    }
    const auto opts = torch::TensorOptions().dtype(at::kFloat).device(w.device());
    auto meta = std::make_shared<Fp8Meta>(w);
    meta->key = key;
    meta->slot = slot;
    meta->slot_name = name;
    meta->role = role_glob;
    meta->w = ScaleRing(opts, history_len);
    meta->x = std::make_shared<ScaleRing>(opts, history_len);
    meta->g = ScaleRing(opts, history_len);
    meta->margin = margin;
    st.by_key.emplace(key, meta);
    if (slot >= 0)
        st.by_slot.emplace(slot, meta);
    st.order.push_back(meta);
    restore_pending_locked(meta);
    return meta;
}

/*
 * Checkpoint snapshot (A1) and test observability
 */

/*
 * One ring's checkpoint slice: the buffer plus which scale pair is current (a
 * restore that assumed pair 0 would hand the next forward the wrong multiplier).
 * ``restore_ring`` recomputes the reciprocal rather than trusting the slot —
 * snapshots predating it carry 0 there.
 */
py::dict ring_state_dict(const ScaleRing& r) {
    py::dict d;
    d["state"] = r.state.detach().clone();
    d["idx"] = r.idx;
    d["cur"] = r.cur; // which scale pair is current (double-buffered)
    d["initialized"] = r.initialized;
    return d;
}

/*
 * Shape matches the Python snapshot (version / entries with shape+dtype plus the
 * three rings), so the trainer's checkpoint-extras bridge is unchanged. v2 adds
 * the slot name/role; a loader ignoring them still binds by (shape, dtype).
 */
py::dict fp8_state_dict() {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    /*
     * Save only what a resume can bind: an orphan (dead weight) is unreachable,
     * and in ``order`` it would shift later bindings. Collect first — drop_meta
     * erases from ``order``, so pruning while walking it invalidates the iterator.
     */
    std::vector<std::shared_ptr<Fp8Meta>> dead;
    for (const auto& meta : st.order)
        if (!meta->alive())
            dead.push_back(meta);
    for (const auto& meta : dead)
        drop_meta(st, meta);
    /*
     * Slotted entries first, in slot order (they bind by name, so the file reads
     * as the module list); unslotted ones trail in registration order.
     */
    std::vector<std::shared_ptr<Fp8Meta>> slotted, rest;
    for (const auto& meta : st.order)
        (meta->slot >= 0 ? slotted : rest).push_back(meta);
    std::sort(slotted.begin(), slotted.end(),
              [](const std::shared_ptr<Fp8Meta>& a, const std::shared_ptr<Fp8Meta>& b) {
                  return a->slot < b->slot;
              });
    const auto entry_of = [](const std::shared_ptr<Fp8Meta>& meta) {
        py::dict e;
        e["shape"] = py::cast(meta->shape);
        e["dtype"] = torch_dtype_str(meta->dtype);
        e["slot_name"] = meta->slot_name;
        e["role"] = meta->role;
        e["w"] = ring_state_dict(meta->w);
        e["x"] = ring_state_dict(*meta->x);
        e["g"] = ring_state_dict(meta->g);
        return e;
    };
    py::list entries;
    for (const auto& meta : slotted)
        entries.append(entry_of(meta));
    for (const auto& meta : rest)
        entries.append(entry_of(meta));
    py::dict out;
    out["version"] = 2;
    out["entries"] = entries;
    return out;
}

namespace {

/* Decode completely before replacing live state; malformed entries fail here. */
c10::optional<RingSnapshot> decode_ring(py::handle obj) {
    if (obj.is_none())
        return c10::nullopt;
    TORCH_CHECK(py::isinstance<py::dict>(obj), "snapshot ring must be a dict or None");
    auto d = py::reinterpret_borrow<py::dict>(obj);
    TORCH_CHECK(d.contains("state") && d.contains("idx") && d.contains("initialized"),
                "snapshot ring requires state, idx and initialized");
    RingSnapshot ring{d["state"].cast<Tensor>(), d["idx"].cast<int64_t>(),
                      d.contains("cur") ? d["cur"].cast<int>() : 0, d["initialized"].cast<bool>()};
    TORCH_CHECK(ring.state.defined() && ring.state.dim() == 1 &&
                    ring.state.scalar_type() == at::kFloat,
                "snapshot ring state must be a 1D float32 tensor");
    TORCH_CHECK(ring.cur == 0 || ring.cur == 1, "snapshot ring cur must be 0 or 1");
    // Both the old single-pair extent and the double-buffer extent remain valid.
    TORCH_CHECK(ring.state.numel() >= quant::RingLayout{1}.size(1) && ring.idx >= 0 &&
                    ring.idx < ring.state.numel() - quant::RingLayout{0}.size(1),
                "snapshot ring idx out of range or state too small");
    return ring;
}

MetaSnapshot decode_meta(py::handle obj) {
    TORCH_CHECK(py::isinstance<py::dict>(obj), "snapshot entry must be a dict");
    auto d = py::reinterpret_borrow<py::dict>(obj);
    TORCH_CHECK(d.contains("shape") && d.contains("dtype") && d.contains("w") && d.contains("x") &&
                    d.contains("g"),
                "snapshot entry requires shape, dtype and w/x/g rings");
    MetaSnapshot entry;
    entry.shape = d["shape"].cast<std::vector<int64_t>>();
    entry.dtype = d["dtype"].cast<std::string>();
    entry.slot_name = d.contains("slot_name") ? d["slot_name"].cast<std::string>() : "";
    entry.role = d.contains("role") ? d["role"].cast<std::string>() : "";
    entry.w = decode_ring(d["w"]);
    entry.x = decode_ring(d["x"]);
    entry.g = decode_ring(d["g"]);
    return entry;
}
} // namespace

void fp8_load_state_dict(py::dict sd) {
    TORCH_CHECK(sd.contains("entries"), "FP8 snapshot requires entries");
    std::vector<MetaSnapshot> pending;
    for (auto entry : sd["entries"].cast<py::list>())
        pending.push_back(decode_meta(entry));
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    st.pending = std::move(pending);
    st.act_cache.clear();
    for (auto& meta : st.order)
        restore_pending_locked(meta);
}

/*
 * Publish the Python slot table (id -> module path, role glob); replaces the
 * whole map. Names are metadata: fp8_reset drops training state, not the model's
 * shape, so it leaves them alone.
 */
void fp8_set_slots(py::list entries) {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    std::unordered_map<int64_t, SlotInfo> slots;
    for (auto item : entries) {
        auto tuple = item.cast<py::tuple>();
        if (tuple.size() != 3)
            continue;
        SlotInfo info;
        info.name = tuple[1].cast<std::string>();
        info.role = tuple[2].cast<std::string>();
        slots.emplace(tuple[0].cast<int64_t>(), std::move(info));
    }
    st.slots = std::move(slots);
}

void fp8_reset() {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    st.by_key.clear();
    st.by_slot.clear();
    st.order.clear();
    st.pending.clear();
    st.act_cache.clear();
}

void fp8_set_act_cache(bool enabled) {
    State& st = state();
    st.act_cache_enabled.store(enabled, std::memory_order_relaxed);
    if (!enabled)
        st.act_cache.clear();
}

void fp8_clear_act_cache() { state().act_cache.clear(); }

py::dict fp8_debug_meta(const Tensor& w, int64_t history_len, int64_t margin, int64_t slot) {
    auto meta = get_meta(w, history_len, margin, slot);
    auto ring = [](const ScaleRing& r) {
        py::dict d;
        d["state"] = r.state;
        d["hist"] = r.hist;
        d["scale"] = r.scale();
        d["scale_recip"] = r.scale_recip();
        d["idx"] = r.idx;
        d["cur"] = r.cur;
        d["initialized"] = r.initialized;
        return d;
    };
    py::dict out;
    out["w"] = ring(meta->w);
    out["x"] = ring(*meta->x);
    out["g"] = ring(meta->g);
    out["slot"] = meta->slot;
    out["slot_name"] = meta->slot_name;
    out["role"] = meta->role;
    return out;
}

void bind_fp8(py::module& m) {
    m.def("fp8_linear", &fp8_linear, py::arg("x"), py::arg("w"), py::arg("bias") = py::none(),
          py::arg("update_rings") = true, py::arg("need_bias_grad") = false,
          py::arg("dynamic") = false, py::arg("history_len") = 16, py::arg("margin") = 0,
          py::arg("fmt_a") = at::kFloat8_e4m3fn, py::arg("fmt_b") = at::kFloat8_e5m2,
          py::arg("slot") = -1);
    m.def("fp8_state_dict", &fp8_state_dict);
    m.def("fp8_load_state_dict", &fp8_load_state_dict);
    m.def("fp8_set_slots", &fp8_set_slots, py::arg("entries"));
    m.def("fp8_reset", &fp8_reset);
    m.def("fp8_set_act_cache", &fp8_set_act_cache, py::arg("enabled"));
    m.def("fp8_clear_act_cache", &fp8_clear_act_cache);
    m.def("fp8_debug_meta", &fp8_debug_meta, py::arg("w"), py::arg("history_len") = 16,
          py::arg("margin") = 0, py::arg("slot") = -1);
}

} // namespace fp8
} // namespace astrai
