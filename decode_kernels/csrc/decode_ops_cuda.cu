#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__device__ __forceinline__ float as_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value) {
  return static_cast<scalar_t>(value);
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ __forceinline__ float block_max(float value, float* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  return scratch[0];
}

template <typename scalar_t>
__global__ void residual_rms_norm_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual_out,
    int64_t hidden,
    float eps) {
  extern __shared__ float scratch[];
  const int64_t row = blockIdx.x;
  const int64_t base = row * hidden;
  float sum_sq = 0.0f;
  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float value = as_float(input[base + col]) + as_float(residual[base + col]);
    sum_sq = fmaf(value, value, sum_sq);
  }
  sum_sq = block_sum(sum_sq, scratch);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(hidden) + eps);
  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float value = as_float(input[base + col]) + as_float(residual[base + col]);
    residual_out[base + col] = from_float<scalar_t>(value);
    output[base + col] = from_float<scalar_t>(value * inv_rms * as_float(weight[col]));
  }
}

template <typename scalar_t>
__global__ void rms_norm_quantize_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t hidden,
    float eps) {
  extern __shared__ float scratch[];
  const int64_t row = blockIdx.x;
  const int64_t base = row * hidden;
  float sum_sq = 0.0f;
  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float value = as_float(input[base + col]);
    sum_sq = fmaf(value, value, sum_sq);
  }
  sum_sq = block_sum(sum_sq, scratch);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(hidden) + eps);

  float local_absmax = 0.0f;
  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float value = as_float(input[base + col]) * inv_rms * as_float(weight[col]);
    local_absmax = fmaxf(local_absmax, fabsf(value));
  }
  const float absmax = block_max(local_absmax, scratch);
  const float scale = fmaxf(absmax / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) {
    scales[row] = scale;
  }
  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float value = as_float(input[base + col]) * inv_rms * as_float(weight[col]);
    const float quantized = fminf(127.0f, fmaxf(-127.0f, nearbyintf(value / scale)));
    output[base + col] = static_cast<int8_t>(quantized);
  }
}

template <typename scalar_t>
__global__ void rope_qk_norm_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ q_weight,
    const scalar_t* __restrict__ k_weight,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    int64_t sequence,
    int64_t q_heads,
    int64_t k_heads,
    int64_t head_dim,
    float eps) {
  extern __shared__ float scratch[];
  const int64_t heads_per_token = q_heads + k_heads;
  const int64_t token = blockIdx.x / heads_per_token;
  const int64_t combined_head = blockIdx.x % heads_per_token;
  const int64_t position = token % sequence;
  const bool is_q = combined_head < q_heads;
  const int64_t head = is_q ? combined_head : combined_head - q_heads;
  const int64_t heads = is_q ? q_heads : k_heads;
  const scalar_t* source = is_q ? q : k;
  scalar_t* destination = is_q ? q_out : k_out;
  const scalar_t* norm_weight = is_q ? q_weight : k_weight;
  const int64_t base = token * heads * head_dim + head * head_dim;

  float sum_sq = 0.0f;
  for (int64_t col = threadIdx.x; col < head_dim; col += blockDim.x) {
    const float value = as_float(source[base + col]);
    sum_sq = fmaf(value, value, sum_sq);
  }
  sum_sq = block_sum(sum_sq, scratch);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(head_dim) + eps);
  const int64_t half = head_dim / 2;
  for (int64_t col = threadIdx.x; col < half; col += blockDim.x) {
    const float first = as_float(source[base + col]) * inv_rms * as_float(norm_weight[col]);
    const float second = as_float(source[base + half + col]) * inv_rms * as_float(norm_weight[half + col]);
    const float cosine = as_float(cos[position * half + col]);
    const float sine = as_float(sin[position * half + col]);
    destination[base + col] = from_float<scalar_t>(first * cosine - second * sine);
    destination[base + half + col] = from_float<scalar_t>(first * sine + second * cosine);
  }
}

template <typename scalar_t>
__global__ void kv_cache_append_kernel(
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slots,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    int64_t tokens,
    int64_t capacity,
    int64_t row_width) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = tokens * row_width;
  if (linear >= elements) {
    return;
  }
  const int64_t token = linear / row_width;
  const int64_t offset = linear % row_width;
  const int64_t slot = slots[token];
  if (slot >= 0 && slot < capacity) {
    key_cache[slot * row_width + offset] = key[linear];
    value_cache[slot * row_width + offset] = value[linear];
  }
}

template <typename scalar_t>
__global__ void bias_swiglu_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    const scalar_t* __restrict__ gate_bias,
    const scalar_t* __restrict__ up_bias,
    scalar_t* __restrict__ output,
    int64_t elements,
    int64_t hidden) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const int64_t col = index % hidden;
  const float gate_value = as_float(gate[index]) + (gate_bias ? as_float(gate_bias[col]) : 0.0f);
  const float up_value = as_float(up[index]) + (up_bias ? as_float(up_bias[col]) : 0.0f);
  const float silu = gate_value / (1.0f + expf(-gate_value));
  output[index] = from_float<scalar_t>(silu * up_value);
}

int64_t row_count(const torch::Tensor& tensor) {
  return tensor.numel() / tensor.size(-1);
}

cudaStream_t current_stream(const torch::Tensor& tensor) {
  return at::cuda::getCurrentCUDAStream(tensor.get_device());
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> residual_rms_norm_cuda(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty_like(input);
  auto residual_out = torch::empty_like(input);
  const int64_t rows = row_count(input);
  if (rows == 0) {
    return {output, residual_out};
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "residual_rms_norm_cuda", [&] {
        residual_rms_norm_kernel<scalar_t><<<rows, kThreads, kThreads * sizeof(float), current_stream(input)>>>(
            input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), residual_out.data_ptr<scalar_t>(), input.size(-1), static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, residual_out};
}

std::tuple<torch::Tensor, torch::Tensor> rms_norm_quantize_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps) {
  c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty(input.sizes(), input.options().dtype(torch::kInt8));
  std::vector<int64_t> scale_shape(input.sizes().begin(), input.sizes().end() - 1);
  auto scales = torch::empty(scale_shape, input.options().dtype(torch::kFloat));
  const int64_t rows = row_count(input);
  if (rows == 0) {
    return {output, scales};
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "rms_norm_quantize_cuda", [&] {
        rms_norm_quantize_kernel<scalar_t><<<rows, kThreads, kThreads * sizeof(float), current_stream(input)>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<int8_t>(),
            scales.data_ptr<float>(), input.size(-1), static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}

std::tuple<torch::Tensor, torch::Tensor> rope_qk_norm_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    double eps) {
  c10::cuda::CUDAGuard guard(q.device());
  auto q_out = torch::empty_like(q);
  auto k_out = torch::empty_like(k);
  const int64_t blocks = q.size(0) * q.size(1) * (q.size(2) + k.size(2));
  if (blocks == 0) {
    return {q_out, k_out};
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "rope_qk_norm_cuda", [&] {
        rope_qk_norm_kernel<scalar_t><<<blocks, kThreads, kThreads * sizeof(float), current_stream(q)>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), q_weight.data_ptr<scalar_t>(),
            k_weight.data_ptr<scalar_t>(), cos.data_ptr<scalar_t>(), sin.data_ptr<scalar_t>(),
            q_out.data_ptr<scalar_t>(), k_out.data_ptr<scalar_t>(), q.size(1), q.size(2), k.size(2),
            q.size(3), static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_out, k_out};
}

void kv_cache_append_cuda(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    const torch::Tensor& slots,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  c10::cuda::CUDAGuard guard(key_cache.device());
  const int64_t row_width = key.size(1) * key.size(2);
  const int64_t elements = key.size(0) * row_width;
  if (elements == 0) {
    return;
  }
  const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, key.scalar_type(), "kv_cache_append_cuda", [&] {
        kv_cache_append_kernel<scalar_t><<<blocks, kThreads, 0, current_stream(key_cache)>>>(
            key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(), slots.data_ptr<int64_t>(),
            key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(), key.size(0), key_cache.size(0), row_width);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor bias_swiglu_cuda(
    const torch::Tensor& gate,
    const torch::Tensor& up,
    const std::optional<torch::Tensor>& gate_bias,
    const std::optional<torch::Tensor>& up_bias) {
  c10::cuda::CUDAGuard guard(gate.device());
  auto output = torch::empty_like(gate);
  if (gate.numel() == 0) {
    return output;
  }
  const int blocks = static_cast<int>((gate.numel() + kThreads - 1) / kThreads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, gate.scalar_type(), "bias_swiglu_cuda", [&] {
        const scalar_t* gate_bias_ptr = gate_bias.has_value() ? gate_bias->data_ptr<scalar_t>() : nullptr;
        const scalar_t* up_bias_ptr = up_bias.has_value() ? up_bias->data_ptr<scalar_t>() : nullptr;
        bias_swiglu_kernel<scalar_t><<<blocks, kThreads, 0, current_stream(gate)>>>(
            gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), gate_bias_ptr, up_bias_ptr,
            output.data_ptr<scalar_t>(), gate.numel(), gate.size(-1));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
