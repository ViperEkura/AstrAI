#pragma once

// POD-only header; no CUDA or torch dependencies.

namespace astrai {
namespace attention {

/* Kernels use BHLD; BLHD inputs transpose dimensions 1 and 2 at entry. */
enum TensorLayout : int {
    BHLD = 0, // [batch, n_heads, seq_len, head_dim]
    BLHD = 1, // [batch, seq_len, n_heads, head_dim]
};

// Split-KV workspace cap: max decode splits per (batch, q_head).
constexpr int MAX_SPLITS = 32;

/* Query rows per host tile; must match Q_TILE_ROWS in inference/workspace.py. */
constexpr int HOST_Q_TILE_ROWS = 64;

/* Converts the softmax scale for exp2: scale_log2 = scale * log2(e). */
constexpr float LOG2E = 1.44269504088896340736f;

/*
 * Shared by contiguous and paged kernels; each call uses one KVSource policy.
 * Void* keeps the POD dtype-agnostic; default initializers keep optional
 * pointers and flags safe. Preserve aggregate/trivially-copyable properties for
 * zero-init, memcpy packing, and by-value kernel arguments.
 */
struct AttentionParams {
    // Group input/output and workspace pointers before scalar metadata.
    const void* q_ptr = nullptr;
    void* o_ptr = nullptr;

    const void* k_ptr = nullptr;
    const void* v_ptr = nullptr;
    const void* new_k_ptr = nullptr;
    const void* new_v_ptr = nullptr;

    // Decode split-KV workspace (FP32 online-softmax accumulators).
    float* o_part = nullptr;
    float* ml_part = nullptr;

    // Paged KV addressing, lengths, and packed query work map.
    const int* req_to_token = nullptr;     // [num_reqs, max_context_len]
    const int* req_pool_indices = nullptr; // [batch]
    const int* kv_indptr = nullptr;        // [batch + 1]
    const int* qo_indptr = nullptr;        // [batch + 1] or nullptr for decode
    const int* q_tile_to_batch = nullptr;  // [num_q_tiles], prefill only
    const int* q_tile_to_index = nullptr;  // [num_q_tiles], prefill only
    const int* cache_seqlens = nullptr; // past lengths; set for block-table storage
    int page_size = 1;
    int table_stride = 0;
    int num_pages = 0;
    const bool* mask = nullptr;

    // Shape
    int batch;
    int q_head;
    int kv_head;
    int head_dim;
    int q_len;  // Per-request in contiguous mode; total_q in paged mode.
    int kv_len; // Contiguous mode; paged mode uses kv_indptr.

    // Tensor strides.
    int q_b_stride;
    int q_h_stride;
    int q_l_stride;
    int q_d_stride;
    int kv_b_stride;
    int kv_h_stride;
    int kv_l_stride;
    int kv_d_stride;
    int new_kv_b_stride;
    int new_kv_h_stride;

    /*
     * Mask shape in [batch, head, query, key] order, read from the mask tensor.
     * Missing axes have size one. Singleton B/H/Q axes broadcast; query/key
     * sizes bound coverage independently of zero strides from expanded storage.
     */
    int mask_b_size;
    int mask_h_size;
    int mask_q_size;
    int mask_k_size;
    int mask_b_stride;
    int mask_h_stride;
    int mask_q_stride;
    int mask_k_stride;

    // Launch metadata and attention behavior; the boolean stays last to avoid padding.
    int num_q_tiles;
    int max_context_len; // req_to_token stride (dim 1)
    int kv_split_size = 0; // Fixed logical KV segment size, in tokens. Zero selects adaptive.
    int split_stride = MAX_SPLITS; // Partial workspace slots per output row.
    int num_splits = 0;
    float scale;
    bool is_causal = false;
};

} // namespace attention
} // namespace astrai
