#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

#include "common.h"

// Torch-bound entry validation for the fp8 family. quantize/common.h stays
// torch-free (pure POD/traits); this header owns the runtime capability
// gate shared by the quantize and quant_gemm bindings.

namespace astrai {
namespace quant {

// Raise unless the device can run fp8 tensor-core MMA (sm_89+).
// getDeviceProperties is ATen-cached, so this stays cheap per call.
inline void check_fp8_device(int device_index) {
    const auto* prop = at::cuda::getDeviceProperties(device_index);
    TORCH_CHECK(sm_at_least(prop->major, prop->minor, kMinSmForFp8Major,
                            kMinSmForFp8Minor),
                "FP8 MMA requires compute capability 8.9+");
}

}  // namespace quant
}  // namespace astrai
