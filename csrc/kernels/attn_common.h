#pragma once


template<typename T, typename AT = float>
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
    int num_splits;
    float scale;
    
    const T* __restrict__ q;
    const T* __restrict__ k;
    const T* __restrict__ v;
    const bool* __restrict__ mask;
    
    T* __restrict__ o;
    AT* __restrict__ o_part;
    AT* __restrict__ ml_part;
};
