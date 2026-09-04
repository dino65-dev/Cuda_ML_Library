#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kBlockWeights = 256;
constexpr int kGroupsPerBlock = 64;
constexpr int kCodeBytes = 40;

__device__ __forceinline__ int load_code(const uint8_t* bytes, int group) {
  const int bit = group * 5;
  const int byte = bit >> 3;
  const int shift = bit & 7;
  int value = static_cast<int>(bytes[byte]) >> shift;
  if (shift > 3) {
    value |= static_cast<int>(bytes[byte + 1]) << (8 - shift);
  }
  return value & 31;
}

__device__ __forceinline__ float sign_at(int code, int position) {
  const int zero = code & 3;
  if (position == zero) {
    return 0.0f;
  }
  int sign_position = position;
  if (position > zero) {
    --sign_position;
  }
  return ((code >> (2 + sign_position)) & 1) ? 1.0f : -1.0f;
}

__global__ void stq_gemv_kernel(
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    const float* __restrict__ activation,
    float* __restrict__ output,
    int blocks_per_row,
    int out_features,
    int batch) {
  const int row = blockIdx.x;
  const int sample = blockIdx.y;
  if (row >= out_features || sample >= batch) {
    return;
  }
  const int tid = threadIdx.x;
  const int total_groups = blocks_per_row * kGroupsPerBlock;
  const uint8_t* row_packed = packed + static_cast<int64_t>(row) * blocks_per_row * kCodeBytes;
  const __half* row_scales = scales + static_cast<int64_t>(row) * blocks_per_row;
  const float* x = activation + static_cast<int64_t>(sample) * total_groups * 4;

  float partial = 0.0f;
  for (int group_index = tid; group_index < total_groups; group_index += blockDim.x) {
    const int block = group_index >> 6;
    const int in_block_group = group_index & 63;
    const int code = load_code(row_packed + block * kCodeBytes, in_block_group);
    const int input_base = group_index * 4;
    const float dot = sign_at(code, 0) * x[input_base] +
                      sign_at(code, 1) * x[input_base + 1] +
                      sign_at(code, 2) * x[input_base + 2] +
                      sign_at(code, 3) * x[input_base + 3];
    partial = fmaf(__half2float(row_scales[block]), dot, partial);
  }

  __shared__ float scratch[kThreads];
  scratch[tid] = partial;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[static_cast<int64_t>(sample) * out_features + row] = scratch[0];
  }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor stq_gemv(
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const torch::Tensor& activation) {
  check_cuda_contiguous(packed, "packed");
  check_cuda_contiguous(scales, "scales");
  check_cuda_contiguous(activation, "activation");
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");
  TORCH_CHECK(scales.scalar_type() == torch::kHalf, "scales must be float16");
  TORCH_CHECK(activation.scalar_type() == torch::kFloat, "activation must be float32");
  TORCH_CHECK(packed.dim() == 3 && packed.size(2) == kCodeBytes,
              "packed must be [out_features, blocks_per_row, 40]");
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == packed.size(0) && scales.size(1) == packed.size(1),
              "scales must be [out_features, blocks_per_row]");
  TORCH_CHECK(activation.dim() == 2 && activation.size(1) == packed.size(1) * kBlockWeights,
              "activation must be [batch, blocks_per_row * 256]");
  TORCH_CHECK(packed.device() == scales.device() && packed.device() == activation.device(),
              "all tensors must share a CUDA device");

  c10::cuda::CUDAGuard guard(activation.device());
  auto output = torch::empty({activation.size(0), packed.size(0)}, activation.options());
  if (output.numel() == 0) {
    return output;
  }
  const dim3 grid(packed.size(0), activation.size(0));
  stq_gemv_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream(activation.get_device())>>>(
      packed.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
      activation.data_ptr<float>(), output.data_ptr<float>(), packed.size(1), packed.size(0), activation.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("stq_gemv", &stq_gemv, "packed STQ1_0 GEMV");
}
