#pragma once

// Pure POD header (no CUDA, no torch).

namespace astrai {
namespace attention {

// Tensor layout for Q/K/V tensors passed to attention kernels.
// Internally, kernels always operate on BHLD [batch, n_heads, seq_len, head_dim].
// When the caller passes BLHD, dims 1 and 2 are transposed at entry.
enum TensorLayout : int {
    BHLD = 0,  // [batch, n_heads, seq_len, head_dim]
    BLHD = 1,  // [batch, seq_len, n_heads, head_dim]
};

// Split-KV workspace cap: max decode splits per (batch, q_head).
constexpr int MAX_SPLITS = 32;

// Paged-prefill host Q-tile granularity in q rows: one q_tile_to_index unit
// covers this many query rows of one request.  Must match Q_TILE_ROWS in
// astrai/inference/workspace.py, which builds the device-side tile maps.
constexpr int HOST_Q_TILE_ROWS = 64;


// Unified attention params covering BOTH addressing modes:
//   - Contiguous K/V: dense [batch, kv_head, kv_len, head_dim] tensors (k/v).
//   - Paged (SGLang-style): flat pool [size, kv_head, head_dim] + req_to_token.
// Each kernel selects the addressing via a KVSource policy (see
// layout_policies.cuh); a given call only touches the fields of one mode, so
// this is a POD shared by both paths rather than two parallel structs that
// drift out of sync.
//
// Dtype-agnostic by design: q/k/v/o pointers are void* and the element type is
// a compile-time kernel parameter, so ONE struct (and one packer) serves every
// precision. Each instantiation casts these pointers once at entry (through the
// KV policy's Elem); device code never branches on a dtype, and nothing in the
// struct has to name one.
//
// Pointer/flag members carry default member initializers: the pointers gate
// optional paths via null checks (new_k_ptr, mask, o_part, ...), so a stack
// `AttentionParams p;` left partially packed must never see garbage
// non-null pointers or a garbage use_mask/causal_offset — that class of bug
// reads through wild addresses. NSDMI keeps the struct an aggregate (C++17)
// and trivially copyable, so `= {}`, memcpy-style packing and by-value kernel
// params all behave exactly as before.
struct AttentionParams {
    // Shape
    int batch;
    int q_head;
    int kv_head;
    int head_dim;
    int q_len;   // Per-request in contiguous mode; total_q in paged mode.
    int kv_len;  // Contiguous mode; paged mode uses kv_indptr.

    // Attention behavior
    float scale;
    // -1 = non-causal; >=0 = absolute position of first Q token
    int causal_offset = -1;
    int use_mask = 0;

    // pointers (element type = the kernel's element-type parameter)
    const void* __restrict__ q_ptr = nullptr;
    const void* __restrict__ k_ptr = nullptr;
    const void* __restrict__ v_ptr = nullptr;
    const bool* __restrict__ mask = nullptr;
    void* __restrict__ o_ptr = nullptr;

    const void* __restrict__ new_k_ptr = nullptr;
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

    // Paged K/V addressing
    const int* __restrict__ req_to_token = nullptr;      // [num_reqs, max_context_len]
    const int* __restrict__ req_pool_indices = nullptr;  // [batch]
    const int* __restrict__ kv_indptr = nullptr;         // [batch + 1]
    const int* __restrict__ qo_indptr = nullptr;         // [batch + 1] or nullptr for decode
    const int* __restrict__ q_tile_to_batch = nullptr;   // [num_q_tiles], prefill only
    const int* __restrict__ q_tile_to_index = nullptr;   // [num_q_tiles], prefill only
    int num_q_tiles;
    int max_context_len; // req_to_token stride (dim 1)

    // Decode split-KV workspace (fp32 online-softmax accumulators, always)
    int num_splits;
    float* __restrict__ o_part = nullptr;
    float* __restrict__ ml_part = nullptr;
};

}  // namespace attention
}  // namespace astrai
