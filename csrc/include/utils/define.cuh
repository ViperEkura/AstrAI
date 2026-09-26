// Function-qualifier macros — the one spelling shared by every stage, so a
// qualifier typo cannot fork the vocabulary (layout_policies.cuh and
// mma/mma.cuh used to carry private copies of DEVICE_FORCEINLINE). Pure
// CUDA, no torch: the standalone harnesses compile this through their
// -I csrc/include. The pair deliberately carries no `static`: on a
// struct-member helper `static` was dead, and at namespace scope these are
// header-inline helpers where internal linkage would only hide ODR
// accidents from the linker.

#pragma once

#define HOST_FORCEINLINE __host__ __forceinline__
#define DEVICE_FORCEINLINE __device__ __forceinline__
