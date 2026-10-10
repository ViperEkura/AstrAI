#pragma once
/* Versioned weight casts, activation LRU, and per-weight metadata. */
#include <c10/core/TensorImpl.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "fp8_ring.h"

namespace astrai {
namespace fp8 {

using torch::Tensor;

/*
 * The cast cache's validity key: every in-place update bumps it (optimizer
 * steps included).
 */
inline int64_t version_of(const Tensor& t) {
    return t.unsafeGetTensorImpl()->version_counter().current_version();
}

/*
 * Version-keyed weight cast cache: the fp8 pair plus the scale they were cast
 * with, so the GEMM always dequantizes with the very scale used. A hit also
 * skips the amax fold and the ring advance — an unchanged weight has an
 * unchanged amax, so the ring tracks optimizer steps instead of calls.
 */
struct WeightCast {
    int64_t version = -1;
    int64_t generation = -1;
    at::ScalarType fmt_a = at::kFloat8_e4m3fn;
    at::ScalarType fmt_b = at::kFloat8_e4m3fn;
    Tensor w8, w8T, sw;

    bool valid(const Tensor& w, at::ScalarType fa, at::ScalarType fb, int64_t gen) const {
        return w8.defined() && version == version_of(w) && generation == gen && fmt_a == fa &&
               fmt_b == fb;
    }

    void fill(const Tensor& w,
              at::ScalarType fa,
              at::ScalarType fb,
              int64_t gen,
              Tensor a,
              Tensor b,
              Tensor s) {
        version = version_of(w);
        generation = gen;
        fmt_a = fa;
        fmt_b = fb;
        w8 = std::move(a);
        w8T = std::move(b);
        sw = std::move(s);
    }
};

/*
 * One quantized activation, kept so the next consumer of the same tensor skips
 * the cast (q/k/v share an input, up/gate another). Identity is a *strong*
 * anchor plus the version counter: an activation's address is recycled almost
 * immediately, so a bare ``data_ptr`` would match a different tensor. No scale
 * snapshot — a hit reads the shared ring's current pair, so anything advancing
 * that ring between cast and consumer would decouple the two.
 */
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

    bool matches(const Tensor& x,
                 at::ScalarType fa,
                 at::ScalarType fb,
                 int64_t hist_len,
                 int64_t scale_margin) const {
        return anchor.get() == x.unsafeGetTensorImpl() && version == version_of(x) && fmt_a == fa &&
               fmt_b == fb && history_len == hist_len && margin == scale_margin &&
               device == x.device().index();
    }
};

struct Fp8Meta {
    ScaleRing w;
    /*
     * Shared with the activation cache: every consumer of one tensor must fold
     * once and read the one published scale.
     */
    std::shared_ptr<ScaleRing> x = std::make_shared<ScaleRing>();
    ScaleRing g;
    WeightCast cast;
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
    int64_t history_len = 0;
    int64_t margin = 0;
    bool dynamic = false;

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
        : weight_ptr(owner.data_ptr()), anchor(owner.getIntrusivePtr()) {}

    bool owns(const Tensor& w) const {
        return anchor.lock().get() != nullptr && weight_ptr == w.data_ptr();
    }
    bool alive() const { return anchor.lock().get() != nullptr; }
};

/*
 * LRU over at most 8 entries / 64 MiB (source tensor plus both fp8 outputs).
 * Bounded because entries hold strong references, and cleared by the caller —
 * autocast exit, fp8_reset, fp8_set_act_cache(false) — so anchors never outlive
 * the region that produced them.
 */
struct ActivationCache {
    static constexpr size_t kMaxEntries = 8;
    static constexpr int64_t kMaxBytes = 64LL << 20;

    std::mutex mu;
    std::vector<ActivationCast> entries;
    int64_t bytes = 0;
    uint64_t clock = 0;

    c10::optional<ActivationCast> find(const Tensor& x,
                                       at::ScalarType fa,
                                       at::ScalarType fb,
                                       int64_t history_len,
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

    void insert(const Tensor& x,
                at::ScalarType fa,
                at::ScalarType fb,
                int64_t history_len,
                int64_t margin,
                Tensor row,
                Tensor transposed,
                std::shared_ptr<ScaleRing> ring) {
        const int64_t input_bytes = x.numel() * x.element_size();
        const int64_t output_bytes =
            row.numel() * row.element_size() + transposed.numel() * transposed.element_size();
        const int64_t entry_bytes = input_bytes + output_bytes;
        if (entry_bytes > kMaxBytes)
            return;

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
            auto oldest = std::min_element(entries.begin(), entries.end(),
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

} // namespace fp8
} // namespace astrai
