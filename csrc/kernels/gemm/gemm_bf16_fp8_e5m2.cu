// W-F8A16 instantiation unit, e5m2 weights: bf16 x fp8 e5m2 (in-register
// hardware widen). One explicit gemm_dispatch instantiation per unit keeps
// the heavy template work in parallel nvcc jobs; gemm.cu addresses the
// specialization through its extern template declarations.
#include "gemm.cuh"

namespace astrai {
namespace gemm {

template void gemm_dispatch<__nv_bfloat16, __nv_fp8_e5m2>(GemmParams,
                                                          cudaStream_t, bool,
                                                          bool);

}  // namespace gemm
}  // namespace astrai
