#pragma once

// POD-only header; no CUDA or torch dependencies.

namespace astrai {
namespace attention {

// Standard workspace capacity: decode splits per (batch, q_head).
constexpr int MAX_SPLITS = 32;

/* Converts the softmax scale for exp2: scale_log2 = scale * log2(e). */
constexpr float LOG2E = 1.44269504088896340736f;

/*
 * Shared by contiguous and paged kernels; each call uses one KVSource policy.
 * Void* keeps the POD dtype-agnostic; default initializers keep optional
 * pointers and flags safe. Preserve aggregate/trivially-copyable properties for
 * zero-init, memcpy packing, and by-value kernel arguments.
 */
struct AttentionParams {
    const void* q_ptr = nullptr;
    const void* k_ptr = nullptr;
    const void* v_ptr = nullptr;
    const void* new_k_ptr = nullptr;
    const void* new_v_ptr = nullptr;

    void* o_ptr = nullptr;

    // Decode split-KV workspace (FP32 online-softmax accumulators).
    float* o_part = nullptr;
    float* ml_part = nullptr;

    // Packed boundaries for varlen and chunked cache queries.
    const int* cu_seqlens_q = nullptr;
    const int* cu_seqlens_k = nullptr;
    // Paged cache metadata for the current request batch.
    const int* page_table = nullptr;    // [batch, logical blocks] -> physical page
    const int* cache_seqlens = nullptr; // materialized tokens before this step
    const bool* mask = nullptr;

    // Shape
    int batch;
    int q_head;
    int kv_head;
    int head_dim;
    int q_len;  // Per-request in contiguous mode; total_q in paged mode.
    int kv_len; // KV tokens (dense/varlen) or per-request capacity (cache).

    // Tensor strides.
    int q_b_stride;
    int q_h_stride;
    int q_l_stride;
    int kv_b_stride;
    int kv_h_stride;
    int kv_l_stride;
    int new_kv_token_stride;
    int new_kv_h_stride;

    /*
     * Mask shape in [batch, head, query, key] order, read from the mask tensor.
     * Missing axes have size one. Singleton B/H/Q axes broadcast; query/key
     * sizes bound coverage independently of zero strides from expanded storage.
     */
    int mask_q_size;
    int mask_k_size;
    int mask_b_stride;
    int mask_h_stride;
    int mask_q_stride;

    int page_size = 1;
    int table_stride = 0;
    int num_pages = 0;

    // Launch metadata and attention behavior; the boolean stays last to avoid padding.
    int kv_split_size = 0; // Fixed logical KV segment size, in tokens. Zero selects adaptive.
    int split_stride = MAX_SPLITS; // Partial workspace slots per output row.
    int num_splits = 0;
    float scale;
    bool is_causal = false;
};

} // namespace attention
} // namespace astrai
