#pragma once

#include <cassert>
#include <api/native/attention.h>
#include <memory/layout_policies.cuh>

namespace astrai::attention {
// Same ragged query schedule as continuous training, with paged KV and no saved
// LSE.
struct CacheQSchedule : VarlenQSchedule {};

template <typename T> __global__ void append_kv(AttentionParams p) {
  const int token = blockIdx.x;
  int lo = 0, hi = p.batch;
  while (lo + 1 < hi) {
    int mid = lo + (hi - lo) / 2;
    if (p.qo_indptr[mid] <= token)
      lo = mid;
    else
      hi = mid;
  }
  int begin = p.qo_indptr[lo], end = p.qo_indptr[lo + 1];
  assert(begin >= 0 && begin <= token && token < end &&
                     end <= p.q_len);
  int past = p.cache_seqlens[lo];
  assert(past >= 0 && past <= p.max_context_len - (end - begin));
  int pos = past + token - begin;
  int page = p.req_to_token[(int64_t)lo * p.table_stride + pos / p.page_size];
  assert(page >= 0 && page < p.num_pages);
  int64_t slot = (int64_t)page * p.page_size + pos % p.page_size;
  for (int i = threadIdx.x; i < p.kv_head * p.head_dim; i += blockDim.x) {
    int64_t src = (int64_t)token * p.new_kv_b_stride +
                  (i / p.head_dim) * p.new_kv_h_stride + i % p.head_dim;
    int64_t dst = slot * p.kv_head * p.head_dim + i;
    static_cast<T *>(const_cast<void *>(p.k_ptr))[dst] =
        static_cast<const T *>(p.new_k_ptr)[src];
    static_cast<T *>(const_cast<void *>(p.v_ptr))[dst] =
        static_cast<const T *>(p.new_v_ptr)[src];
  }
}

} // namespace astrai::attention
