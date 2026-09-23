// fp8 x fp8 e4m3 instantiation unit. One explicit gemm_dispatch
// instantiation per unit keeps the heavy template work in parallel nvcc
// jobs; gemm.cu addresses the specialization through its extern template
// declarations.
#include "gemm.cuh"

namespace astrai {
namespace gemm {

ASTRAI_GEMM_INSTANTIATE(__nv_fp8_e4m3, __nv_fp8_e4m3);

}  // namespace gemm
}  // namespace astrai
