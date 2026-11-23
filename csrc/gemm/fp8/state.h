#pragma once
/* Typed FP8 registry and snapshots; Python conversion belongs to runtime.cu. */
#include <c10/core/TensorImpl.h>
#include <c10/util/intrusive_ptr.h>

#include "cache.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace astrai {
namespace fp8 {
using torch::Tensor;

struct Fp8Meta {
    ScaleRing w;
    /*
     * Shared with the activation cache: every consumer of one tensor must fold
     * once and read the one published scale.
     */
    std::shared_ptr<ScaleRing> x;
    ScaleRing g;
    int64_t weight_version = -1; // fold weight amax once per update
    at::ScalarType weight_format = at::ScalarType::Undefined;
    /*
     * Weak on purpose: it keeps a dead TensorImpl's address from being recycled
     * (so the pointer comparison stays an identity) and its expiry *is* the
     * eviction signal — a strong ref would pin the replaced weight.
     */
    void* weight_ptr = nullptr;
    c10::weak_intrusive_ptr<c10::TensorImpl> anchor;
    std::string key;            // this meta's registry key (eviction path)
    std::vector<int64_t> shape; // the owning weight's (snapshot key)
    at::ScalarType dtype = at::kBFloat16;
    int64_t margin = 0;

    /*
     * Slot addressing (slot addressing in autocast.py): the state belongs to the *module*, so a
     * replaced weight parameter re-points the identity and keeps the rings.
     * ``slot_name`` is the snapshot key, ``role`` the Python-side policy glob;
     * all three stay empty for an unslotted meta (bare call, bench).
     */
    int64_t slot = -1;
    std::string slot_name;
    std::string role;

    /*
     * A weak pointer has no empty state and a meta always has an owner, which
     * must be live.
     */
    explicit Fp8Meta(const Tensor& owner)
        : weight_ptr(owner.data_ptr()), anchor(owner.getIntrusivePtr()),
          shape(owner.sizes().begin(), owner.sizes().end()), dtype(owner.scalar_type()) {}

    bool owns(const Tensor& w) const {
        return anchor.lock().get() != nullptr && weight_ptr == w.data_ptr();
    }
    bool alive() const { return anchor.lock().get() != nullptr; }
};

struct SlotInfo {
    std::string name;
    std::string role;
};

struct RingSnapshot {
    Tensor state;
    int64_t idx;
    int cur;
    bool initialized;
};

struct MetaSnapshot {
    std::vector<int64_t> shape;
    std::string dtype;
    std::string slot_name;
    std::string role;
    c10::optional<RingSnapshot> w, x, g;
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

    /*
     * Snapshot entries queued by load_state_dict, consumed by slot name or —
     * unslotted — by (shape, dtype) in this order. Unmatched entries stay
     * queued: a model-shape change across the checkpoint just re-seeds.
     */
    std::vector<MetaSnapshot> pending;
};

State& state();
std::shared_ptr<Fp8Meta>
get_meta(const Tensor& w, int64_t history_len, int64_t margin, int64_t slot);
} // namespace fp8
} // namespace astrai
