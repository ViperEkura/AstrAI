#pragma once
// Attention's instantiation list — the one place that says which precisions
// this family has kernels for, in the same dialect as gemm's pair table
// (ASTRAI_GEMM_PAIRS): torch's own ScalarType names the dtype at the boundary,
// the C++ element type is what the kernels take as a template parameter.
//
//     X(torch scalar type, element type)
//
// Nothing else in the family names a dtype: the four entries switch on
// q.scalar_type() over this list, and the refusal below is generated from the
// same rows, so the supported set cannot drift into hand-kept message text.
// The C++ harnesses (csrc/tests, no torch) never touch this header — they call
// the dispatchers with an element type directly.
//
// Adding a precision = a row here + its ElemTrait (utils/dtype.cuh) and, for
// a tensor-core dtype, the MmaShapeFor/MmaOp cell (common/mma.cuh). Verified in
// a scratch build with fp16: nothing else needs an edit.

#include <string>

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <utils/dtype.cuh>

namespace astrai {
namespace attention {

// The element types are spelled namespace-qualified: a row is expanded at the
// entries' file scope, where only `astrai::attention` is open.
#define ASTRAI_ATTN_DTYPE_LIST(X) \
    X(at::kBFloat16, astrai::bf16)

// A scalar type attention has no kernel for: say which ones it does have, read
// off the list above.
inline void attn_dtype_unsupported(at::ScalarType st) {
    std::string instantiated;
#define ASTRAI_ATTN_DTYPE_NAME_ROW(tag, type)                                  \
    instantiated += std::string(instantiated.empty() ? "" : ", ") +            \
                    c10::toString(tag);
    ASTRAI_ATTN_DTYPE_LIST(ASTRAI_ATTN_DTYPE_NAME_ROW)
#undef ASTRAI_ATTN_DTYPE_NAME_ROW
    TORCH_CHECK(false, "attention has no kernel for ", c10::toString(st),
                " (instantiated: ", instantiated, ")");
}

}  // namespace attention
}  // namespace astrai
