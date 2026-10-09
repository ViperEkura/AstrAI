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
    // Shape
    int batch;
    int q_head;
    int kv_head;
    int head_dim;
    int q_len;  // Per-request in contiguous mode; total_q in paged mode.
    int kv_len; // Contiguous mode; paged mode uses kv_indptr.

    // Attention behavior
    float scale;
    bool is_causal = false;

    // pointers (element type = the kernel's element-type parameter)
    const void* __restrict__ q_ptr = nullptr;
    // Keep K/V pointer pairs aligned for shared parameter loads.
    alignas(16) const void* __restrict__ k_ptr = nullptr;
    const void* __restrict__ v_ptr = nullptr;
    const bool* __restrict__ mask = nullptr;
    void* __restrict__ o_ptr = nullptr;

    alignas(16) const void* __restrict__ new_k_ptr = nullptr;
    const void* __restrict__ new_v_ptr = nullptr;

    // strides
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

    int mask_b_stride;
    int mask_h_stride;
    int mask_l_stride;
    int mask_k_len = 0;
    int mask_q_len = 0;

    // Paged K/V addressing
    const int* __restrict__ req_to_token = nullptr;     // [num_reqs, max_context_len]
    const int* __restrict__ req_pool_indices = nullptr; // [batch]
    const int* __restrict__ kv_indptr = nullptr;        // [batch + 1]
    const int* __restrict__ qo_indptr = nullptr;        // [batch + 1] or nullptr for decode
    const int* __restrict__ q_tile_to_batch = nullptr;  // [num_q_tiles], prefill only
    const int* __restrict__ q_tile_to_index = nullptr;  // [num_q_tiles], prefill only
    int num_q_tiles;
    int max_context_len; // req_to_token stride (dim 1)

    // Decode split-KV workspace (fp32 online-softmax accumulators, always)
    int num_splits = 0;
    bool direct_output = false;
    float* __restrict__ o_part = nullptr;
    float* __restrict__ ml_part = nullptr;
};

} // namespace attention
} // namespace astrai
