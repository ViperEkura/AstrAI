// W8A8 instantiation unit: int8 x int8 (both scales; native s8 mma, s32
// accumulators). One explicit gemm_dispatch instantiation per unit keeps
// the heavy template work in parallel nvcc jobs; gemm.cu addresses the
// specialization through its extern template declarations.
#include "gemm.cuh"

namespace astrai {
namespace gemm {

template void gemm_dispatch<int8_t, int8_t>(GemmParams, cudaStream_t, bool,
                                            bool);

}  // namespace gemm
}  // namespace astrai
