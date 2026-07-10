#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

using bf16 = __nv_bfloat16;
using std::min;

struct AttentionParams {
    int batch;
    int q_head;
    int kv_head;
    int q_len;
    int kv_len;
    int head_dim;
    int use_mask;
    int is_causal;
    int causal_offset;
    float scale;
    const bf16* __restrict__ q;
    const bf16* __restrict__ k;
    const bf16* __restrict__ v;
    const bool* __restrict__ mask;
    
    bf16* __restrict__ o;
    float* __restrict__ o_part;
    float* __restrict__ ml_part;
    int num_splits;
};
