#pragma once
/* Bounded cache for activations shared by multiple projections. */
#include <c10/util/Optional.h>

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <utility>

#include "ring.h"

namespace astrai {
namespace fp8 {

using torch::Tensor;

inline int64_t version_of(const Tensor& t) {
    return t.unsafeGetTensorImpl()->version_counter().current_version();
}

/* A strong input identity prevents a recycled TensorImpl from becoming a hit. */
struct ActivationCast {
    Tensor input, out, out_t, scale;
    std::shared_ptr<ScaleRing> ring;
    int64_t version;
    int64_t margin;
    int64_t bytes;

    bool matches(const Tensor& x,
                 at::ScalarType dtype,
                 at::ScalarType tdtype,
                 int64_t hist_len,
                 int64_t scale_margin) const {
        return input.unsafeGetTensorImpl() == x.unsafeGetTensorImpl() && version == version_of(x) &&
               out.scalar_type() == dtype && out_t.scalar_type() == tdtype &&
               out.device() == x.device() && ring->hist.numel() == hist_len &&
               margin == scale_margin;
    }
};

/* Consumers share one cast and ring; autocast exit/reset releases strong inputs. */
class ActivationCache {
    static constexpr size_t kMaxEntries = 8;
    static constexpr int64_t kMaxBytes = 64LL << 20;

    std::mutex mu;
    std::list<ActivationCast> entries; // oldest first
    int64_t bytes = 0;

  public:
    c10::optional<ActivationCast> find(const Tensor& x,
                                       at::ScalarType dtype,
                                       at::ScalarType tdtype,
                                       int64_t history_len,
                                       int64_t margin) {
        std::lock_guard<std::mutex> lock(mu);
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->matches(x, dtype, tdtype, history_len, margin)) {
                entries.splice(entries.end(), entries, it);
                return *it;
            }
        }
        return c10::nullopt;
    }

    void insert(const Tensor& input,
                Tensor out,
                Tensor out_t,
                Tensor scale,
                std::shared_ptr<ScaleRing> ring,
                int64_t margin) {
        const int64_t entry_bytes = input.numel() * input.element_size() +
                                    out.numel() * out.element_size() +
                                    out_t.numel() * out_t.element_size();
        if (entry_bytes > kMaxBytes)
            return;

        std::lock_guard<std::mutex> lock(mu);
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->input.unsafeGetTensorImpl() == input.unsafeGetTensorImpl()) {
                bytes -= it->bytes;
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
        while (!entries.empty() &&
               (entries.size() >= kMaxEntries || bytes + entry_bytes > kMaxBytes)) {
            bytes -= entries.front().bytes;
            entries.pop_front();
        }
        entries.push_back({input, std::move(out), std::move(out_t), std::move(scale),
                           std::move(ring), version_of(input), margin, entry_bytes});
        bytes += entry_bytes;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu);
        entries.clear();
        bytes = 0;
    }
};

} // namespace fp8
} // namespace astrai
