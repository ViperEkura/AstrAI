// nvcc -std=c++17 -arch=sm_120a -I csrc/include csrc/tests/mma_test.cu -o /tmp/mma_test
#include <cstdio>
#include <cuda_runtime.h>

#include <mma/mma.cuh>
#include <utils/arch.h>

using namespace astrai;

static_assert(supports(CudaTarget{80, 0}, 120));
static_assert(!supports(CudaTarget{89, 0}, 80));
static_assert(supports(CudaTarget{120, 'a'}, 120));
static_assert(!supports(CudaTarget{120, 'a'}, 121));
static_assert(supports(CudaTarget{120, 'f'}, 121));
static_assert(!supports(CudaTarget{120, 'f'}, 130));
static_assert(!supports(CudaTarget{121, 'f'}, 120));
static_assert(!supports(CudaTarget{0, 0}, 120));

template <typename Op, unsigned Packed, int K> __device__ void check_atom(int* failures) {
    typename Op::AFrag a;
    typename Op::BFrag b;
    typename Op::CFrag c, d;
    for (int i = 0; i < 4; ++i) {
        a[i] = Packed;
        c[i] = 3;
        d[i] = 99;
    }
    for (int i = 0; i < 2; ++i)
        b[i] = Packed;
    Op::fma(d, a, b, c);
    for (int i = 0; i < 4; ++i)
        if (d[i] != K + 3)
            atomicAdd(failures, 1);
    Op::fma(d, a, b, d);
    for (int i = 0; i < 4; ++i)
        if (d[i] != 2 * K + 3)
            atomicAdd(failures, 1);
}

__global__ void check_atoms(int* failures) {
    check_atom<MmaOpImpl<__nv_bfloat16>, 0x3f803f80, 16>(failures);
    check_atom<MmaOpImpl<int8_t>, 0x01010101, 32>(failures);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    check_atom<MmaOpImpl<__nv_fp8_e4m3>, 0x38383838, 32>(failures);
    check_atom<MmaOpImpl<__nv_fp8_e5m2>, 0x3c3c3c3c, 32>(failures);
#endif
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) ||                                                         \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ == 1200)
    check_atom<MxMmaOp<__nv_fp8_e4m3>, 0x38383838, 32>(failures);
    check_atom<MxMmaOp<__nv_fp8_e5m2>, 0x3c3c3c3c, 32>(failures);
#endif
}

int main() {
    int* failures = nullptr;
    cudaError_t status = cudaMallocManaged(&failures, sizeof(int));
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        return 1;
    }
    *failures = 0;
    check_atoms<<<1, 32>>>(failures);
    status = cudaDeviceSynchronize();
    const int count = *failures;
    cudaFree(failures);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        return 1;
    }
    std::printf("MMA fragment errors: %d\n", count);
    return count != 0;
}
