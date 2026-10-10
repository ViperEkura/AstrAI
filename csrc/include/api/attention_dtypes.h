#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include <api/dtype.h>

namespace astrai {
namespace attention {

// Supported attention element types. ScalarTypeOf<T> supplies the torch dtype.
#define ASTRAI_ATTN_DTYPE_LIST(X) X(bf16)

template <template <typename> class DispatchFn, typename ParamsT>
inline void attn_dtype_dispatch(at::ScalarType st, ParamsT& p, cudaStream_t stream) {
    switch (st) {
#define ASTRAI_ATTN_DTYPE_CASE(T)                                                                  \
    case scalar_type_v<T>:                                                                         \
        return DispatchFn<T>::run(p, stream);
        ASTRAI_ATTN_DTYPE_LIST(ASTRAI_ATTN_DTYPE_CASE)
#undef ASTRAI_ATTN_DTYPE_CASE
    }
    std::string supported;
#define ASTRAI_ATTN_DTYPE_NAME(T)                                                                  \
    supported += std::string(supported.empty() ? "" : ", ") + c10::toString(scalar_type_v<T>);
    ASTRAI_ATTN_DTYPE_LIST(ASTRAI_ATTN_DTYPE_NAME)
#undef ASTRAI_ATTN_DTYPE_NAME
    throw std::runtime_error("attention has no kernel for " + std::string(c10::toString(st)) +
                             " (instantiated: " + supported + ")");
}

} // namespace attention
} // namespace astrai
