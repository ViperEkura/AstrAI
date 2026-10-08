#pragma once

#include <api/gemm_common.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace astrai::symmetric {

void syrk_out(torch::Tensor x, torch::Tensor output,
              c10::optional<torch::Tensor> addend, float alpha, float beta,
              std::string tile);
void symm_out(torch::Tensor symmetric, torch::Tensor x, torch::Tensor output,
              c10::optional<torch::Tensor> addend, float alpha, float beta,
              std::string tile, int raster);
py::dict plan(std::string operation, int64_t rows, int64_t cols, int64_t batch_size,
              std::string input_layout, std::string output_layout, bool addend, int device);
py::list tiles(std::string operation);

// Private launch boundary. Tensor validation and parameter packing stay in entry.cu.
void launch_syrk(gemm::GemmParams p, const c10::optional<torch::Tensor>& addend,
                 float alpha, float beta, const std::string& tile,
                 bool column_input, cudaStream_t stream);
void launch_symm(gemm::GemmParams p, const c10::optional<torch::Tensor>& addend,
                 float alpha, float beta, const std::string& tile,
                 bool column_input, bool column_output, cudaStream_t stream);

} // namespace astrai::symmetric
