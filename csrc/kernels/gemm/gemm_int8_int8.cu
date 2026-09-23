// W8A8 instantiation unit: int8 x int8 (both scales; native s8 mma, s32
// accumulators). One explicit gemm_dispatch instantiation per unit keeps
// the heavy template work in parallel nvcc jobs; gemm.cu addresses the
// specialization through its extern template declarations.
#include "gemm.cuh"

namespace astrai {
namespace gemm {

ASTRAI_GEMM_INSTANTIATE(int8_t, int8_t);

}  // namespace gemm
}  // namespace astrai
