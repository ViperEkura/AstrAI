// Entry points of the GDN kernel module.
//
// `gated_deltanet_fwd` prepares the chunked forward's operands and
// `gated_deltanet_bwd` is the first stage of the reverse pass. The chunked
// kernels themselves are not in the tree yet.

#pragma once

#include <torch/extension.h>

#include <vector>

namespace astrai {
namespace gdn {

std::vector<torch::Tensor> gated_deltanet_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    double eps,
    int64_t chunk
);

std::vector<torch::Tensor> gated_deltanet_bwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v_new,
    torch::Tensor h,
    torch::Tensor g,
    torch::Tensor do_grad,
    double scale
);

}  // namespace gdn
}  // namespace astrai
