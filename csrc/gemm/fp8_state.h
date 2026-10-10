#pragma once
/* Process-wide FP8 metadata registry, slots, and checkpoint restore. */
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "fp8_cache.h"

namespace astrai {
namespace fp8 {

using torch::Tensor;

/*
 * Process-wide state: the meta registry, the generation counter, the snapshot.
 */

struct SlotInfo {
    std::string name; // module path — the snapshot's stable key
    std::string role; // Python-side glob (``layers.*.mlp.up``)
};

struct State {
    std::mutex mu;
    std::unordered_map<std::string, std::shared_ptr<Fp8Meta>> by_key;
    std::unordered_map<int64_t, std::shared_ptr<Fp8Meta>> by_slot;
    /*
     * Slot id -> (module path, role), published once by the Python slot
     * not training state: fp8_reset leaves it alone, a later fp8_set_slots
     * replaces the lot.
     */
    std::unordered_map<int64_t, SlotInfo> slots;
    std::vector<std::shared_ptr<Fp8Meta>> order; // registration order (A1)
    ActivationCache act_cache;
    std::atomic<bool> act_cache_enabled{true};
    std::atomic<int64_t> n_act_hit{0};
    std::atomic<int64_t> n_act_miss{0};

    /*
     * Snapshot entries queued by load_state_dict, consumed by slot name or —
     * unslotted — by (shape, dtype) in this order. Unmatched entries stay
     * queued: a model-shape change across the checkpoint just re-seeds.
     */
    std::vector<py::dict> pending;
    int64_t generation = 0; // bumped by restore / reset / recipe rebuild
    // Test/bench observability.
    std::atomic<int64_t> n_quantize{0};
    std::atomic<int64_t> n_gemm{0};
    std::atomic<int64_t> n_cast_hit{0};
    std::atomic<int64_t> n_cast_miss{0};
};

State& state();

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

/*
 * Drop a meta from every registry view — a leftover in ``order`` would shift the
 * fallback binding of later unslotted metas. The identity checks matter: a key is
 * claimed by only one meta (first ``emplace`` wins), so never erase another's.
 */
inline void drop_meta(State& st, const std::shared_ptr<Fp8Meta>& meta) {
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
 * Re-point a slot's meta at a new weight tensor, keeping the rings: TP/FSDP swap
 * the Parameter, not the module. The cast cache must still go — it keys on the
 * version counter, and a fresh parameter starts at a version its predecessor may
 * have been cast at. A kept ring means the first fold after the swap still uses
 * the previous scale (a wild magnitude change clips for one step).
 */
inline void rebind_meta_locked(const std::shared_ptr<Fp8Meta>& meta, const Tensor& w) {
    State& st = state();
    auto it = st.by_key.find(meta->key);
    if (it != st.by_key.end() && it->second == meta)
        st.by_key.erase(it);
    meta->key = meta_key(w);
    meta->weight_ptr = w.data_ptr();
    meta->anchor = c10::weak_intrusive_ptr<c10::TensorImpl>(w.getIntrusivePtr());
    meta->shape.assign(w.sizes().begin(), w.sizes().end());
    meta->dtype = w.scalar_type();
    meta->cast = WeightCast{};
    st.by_key.emplace(meta->key, meta);
}

/*
 * The snapshot's dtype field uses Python's ``str(torch.dtype)`` spelling — the
 * format the checkpoint bridge established, so either side restores the other's.
 */
inline std::string torch_dtype_str(at::ScalarType t) {
    switch (t) {
    case at::kBFloat16:
        return "torch.bfloat16";
    case at::kHalf:
        return "torch.float16";
    case at::kFloat:
        return "torch.float32";
    case at::kDouble:
        return "torch.float64";
    case at::kChar:
        return "torch.int8";
    case at::kByte:
        return "torch.uint8";
    case at::kShort:
        return "torch.int16";
    case at::kInt:
        return "torch.int32";
    case at::kLong:
        return "torch.int64";
    case at::kBool:
        return "torch.bool";
    case at::kFloat8_e4m3fn:
        return "torch.float8_e4m3fn";
    case at::kFloat8_e5m2:
        return "torch.float8_e5m2";
    default:
        return c10::toString(t);
    }
}

inline bool dtype_str_matches(const std::string& s, at::ScalarType t) {
    return s == torch_dtype_str(t) || s == std::string(c10::toString(t));
}

/*
 * Restore one ring. False geometry = the buffer changed since the save (recipe
 * change across the checkpoint): the ring stays fresh and re-seeds on next use.
 */
inline void restore_ring(ScaleRing& ring, const py::object& sd, bool& geometry_ok) {
    if (sd.is_none())
        return;
    py::dict d = sd.cast<py::dict>();
    Tensor saved = d["state"].cast<Tensor>();
    if (saved.numel() != ring.state.numel()) {
        geometry_ok = false;
        return;
    }
    ring.state.copy_(saved.to(ring.state.device()));
    ring.cur = d.contains("cur") ? py::cast<int>(d["cur"]) : 0;
    ring.publish_recip(); // snapshots older than the recip slot restore 0
    ring.idx = py::cast<int64_t>(d["idx"]);
    ring.initialized = py::cast<bool>(d["initialized"]);
}

/*
 * Consume queued entries: a named entry binds only to the module that owns the
 * name (so two same-shaped linears cannot swap state), unnamed ones fall back to
 * (shape, dtype) in queue order — data_ptr is meaningless across processes.
 */
inline void restore_pending_locked(std::shared_ptr<Fp8Meta>& meta) {
    State& st = state();
    if (!meta->slot_name.empty()) {
        for (size_t i = 0; i < st.pending.size(); ++i) {
            const py::dict entry = st.pending[i];
            if (!entry.contains("slot_name"))
                continue;
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
        /*
         * A named entry waits for its module: binding it here by shape would
         * reintroduce exactly the cross-wiring the names exist to prevent.
         */
        if (entry.contains("slot_name") && !py::cast<std::string>(entry["slot_name"]).empty())
            continue;
        bool match = py::len(entry["shape"]) == meta->shape.size();
        for (size_t d = 0; match && d < meta->shape.size(); ++d)
            match = py::cast<int64_t>(entry["shape"][py::int_(d)]) == meta->shape[d];
        match = match && dtype_str_matches(py::cast<std::string>(entry["dtype"]), meta->dtype);
        if (!match)
            continue;
        bool geometry_ok = true;
        restore_ring(meta->w, entry["w"], geometry_ok);
        restore_ring(*meta->x, entry["x"], geometry_ok);
        restore_ring(meta->g, entry["g"], geometry_ok);
        st.pending.erase(st.pending.begin() + static_cast<long>(i));
        return;
    }
}

/*
 * Find or create a weight's rings. A slot survives a replaced weight tensor —
 * only the identity is re-pointed (rebind_meta_locked). A recipe change, or a
 * failed ownership check on the address path, rebuilds with a generation bump
 * that invalidates the cast caches; fresh rings re-seed on the next forward.
 * Without a slot the address-keyed path runs unchanged (bare calls, benches, and
 * the backward pass, which reaches its meta through the weight forward saved).
 */
inline std::shared_ptr<Fp8Meta> get_meta(const Tensor& w,
                                         int64_t history_len,
                                         int64_t margin,
                                         bool dynamic,
                                         int64_t slot = -1,
                                         const std::string& slot_name = "",
                                         const std::string& role = "") {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    /*
     * The published slot table wins; the arguments cover tests that want a name
     * without registering one.
     */
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
                if (!meta->owns(w))
                    rebind_meta_locked(meta, w);
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
    const auto opts = torch::TensorOptions().dtype(at::kFloat).device(w.device());
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
    if (slot >= 0)
        st.by_slot.emplace(slot, meta);
    st.order.push_back(meta);
    restore_pending_locked(meta);
    return meta;
}

} // namespace fp8
} // namespace astrai
