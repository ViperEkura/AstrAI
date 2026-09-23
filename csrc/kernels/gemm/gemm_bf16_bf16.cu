// W16A16 instantiation unit: bf16 x bf16 (no scales). One explicit
// gemm_dispatch instantiation per unit keeps the heavy template work in
// parallel nvcc jobs; gemm.cu addresses the specialization through its
// extern template declarations.
#include "gemm.cuh"

namespace astrai {
namespace gemm {

ASTRAI_GEMM_INSTANTIATE(__nv_bfloat16, __nv_bfloat16);

}  // namespace gemm
}  // namespace astrai
