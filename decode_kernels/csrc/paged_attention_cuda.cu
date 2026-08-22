#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__device__ __forceinline__ float load_float(const scalar_t* ptr, int64_t index) {
  return static_cast<float>(ptr[index]);
}

template <typename scalar_t>
__global__ void paged_append_float_kernel(
    scalar_t* key_cache, scalar_t* value_cache, const int64_t* block_tables,
    const int64_t* positions, const scalar_t* key, const scalar_t* value,
    int batch, int max_blocks, int num_blocks, int block_size, int kv_heads, int head_dim) {
  const int64_t elements = static_cast<int64_t>(batch) * kv_heads * head_dim;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < elements; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int d = linear % head_dim;
    const int64_t row = linear / head_dim;
    const int head = row % kv_heads;
    const int request = row / kv_heads;
    const int64_t position = positions[request];
    if (position < 0) continue;
    const int logical_block = position / block_size;
    if (logical_block >= max_blocks) continue;
    const int64_t physical = block_tables[static_cast<int64_t>(request) * max_blocks + logical_block];
    if (physical < 0 || physical >= num_blocks) continue;
    const int offset = position % block_size;
    const int64_t cache_index = (((physical * block_size + offset) * kv_heads + head) * head_dim + d);
    key_cache[cache_index] = key[linear];
    value_cache[cache_index] = value[linear];
  }
}

template <typename scalar_t>
__global__ void paged_append_int8_kernel(
    int8_t* key_cache, int8_t* value_cache, float* key_scales, float* value_scales,
    const int64_t* block_tables, const int64_t* positions,
    const scalar_t* key, const scalar_t* value,
    int batch, int max_blocks, int num_blocks, int block_size, int kv_heads, int head_dim) {
  const int request = blockIdx.x / kv_heads;
  const int head = blockIdx.x % kv_heads;
  const int64_t position = positions[request];
  if (position < 0) return;
  const int logical_block = position / block_size;
  if (logical_block >= max_blocks) return;
  const int64_t physical = block_tables[static_cast<int64_t>(request) * max_blocks + logical_block];
  if (physical < 0 || physical >= num_blocks) return;
  const int offset = position % block_size;
  const int64_t update_base = (static_cast<int64_t>(request) * kv_heads + head) * head_dim;

  __shared__ float reduction[kThreads];
  float key_max = 0.0f, value_max = 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    key_max = fmaxf(key_max, fabsf(load_float(key, update_base + d)));
    value_max = fmaxf(value_max, fabsf(load_float(value, update_base + d)));
  }
  reduction[threadIdx.x] = key_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  const float ks = fmaxf(reduction[0] / 127.0f, 1.0e-12f);
  reduction[threadIdx.x] = value_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  const float vs = fmaxf(reduction[0] / 127.0f, 1.0e-12f);
  const int64_t scale_index = (physical * block_size + offset) * kv_heads + head;
  if (threadIdx.x == 0) { key_scales[scale_index] = ks; value_scales[scale_index] = vs; }
  const int64_t cache_base = scale_index * head_dim;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    key_cache[cache_base + d] = static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, nearbyintf(load_float(key, update_base + d) / ks))));
    value_cache[cache_base + d] = static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, nearbyintf(load_float(value, update_base + d) / vs))));
  }
}

template <typename query_t, typename cache_t, bool Quantized>
__global__ void paged_attention_split_kernel(
    const query_t* query, const cache_t* key_cache, const cache_t* value_cache,
    const float* key_scales, const float* value_scales,
    const int64_t* block_tables, const int64_t* lengths,
    float* partial_output, float* partial_max, float* partial_sum,
    int batch, int query_heads, int kv_heads, int head_dim,
    int num_blocks, int block_size, int max_blocks, int splits, float scale) {
  const int split = blockIdx.x % splits;
  const int head_and_request = blockIdx.x / splits;
  const int q_head = head_and_request % query_heads;
  const int request = head_and_request / query_heads;
  const int kv_head = q_head / (query_heads / kv_heads);
  const int length = static_cast<int>(lengths[request]);
  const int tokens_per_split = (length + splits - 1) / splits;
  const int begin = split * tokens_per_split;
  const int end = min(length, begin + tokens_per_split);
  const int64_t partial_row = (static_cast<int64_t>(request) * query_heads + q_head) * splits + split;
  if (begin >= end) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
      partial_output[partial_row * head_dim + d] = 0.0f;
    if (threadIdx.x == 0) { partial_max[partial_row] = -INFINITY; partial_sum[partial_row] = 0.0f; }
    return;
  }

  // One warp owns a query-head/split. Each lane holds up to eight output
  // dimensions in registers (head_dim <= 256), so dot reduction needs only
  // warp shuffles and online softmax needs no block-wide barriers.
  constexpr int kItemsPerLane = 8;
  float accumulator[kItemsPerLane] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  const int lane = threadIdx.x;
  const int64_t query_base = (static_cast<int64_t>(request) * query_heads + q_head) * head_dim;
  for (int position = begin; position < end; ++position) {
    const int logical_block = position / block_size;
    const int offset = position % block_size;
    const int64_t physical = logical_block < max_blocks
        ? block_tables[static_cast<int64_t>(request) * max_blocks + logical_block] : -1;
    float dot = 0.0f;
    int64_t cache_base = 0;
    float ks = 1.0f, vs = 1.0f;
    if (physical >= 0 && physical < num_blocks) {
      const int64_t scale_index = (physical * block_size + offset) * kv_heads + kv_head;
      cache_base = scale_index * head_dim;
      if (Quantized) { ks = key_scales[scale_index]; vs = value_scales[scale_index]; }
      #pragma unroll
      for (int item = 0; item < kItemsPerLane; ++item) {
        const int d = lane + item * 32;
        if (d < head_dim)
          dot = fmaf(load_float(query, query_base + d), load_float(key_cache, cache_base + d) * ks, dot);
      }
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    float alpha = 0.0f, beta = 0.0f;
    if (lane == 0) {
      const float score = dot * scale;
      const float next_max = fmaxf(running_max, score);
      alpha = expf(running_max - next_max);
      beta = expf(score - next_max);
      running_sum = running_sum * alpha + beta;
      running_max = next_max;
    }
    alpha = __shfl_sync(0xffffffffu, alpha, 0);
    beta = __shfl_sync(0xffffffffu, beta, 0);
    #pragma unroll
    for (int item = 0; item < kItemsPerLane; ++item) {
      const int d = lane + item * 32;
      if (d < head_dim && physical >= 0 && physical < num_blocks)
        accumulator[item] = accumulator[item] * alpha + beta * load_float(value_cache, cache_base + d) * vs;
    }
  }
  #pragma unroll
  for (int item = 0; item < kItemsPerLane; ++item) {
    const int d = lane + item * 32;
    if (d < head_dim) partial_output[partial_row * head_dim + d] = accumulator[item];
  }
  if (lane == 0) { partial_max[partial_row] = running_max; partial_sum[partial_row] = running_sum; }
}

template <typename scalar_t>
__global__ void paged_attention_reduce_kernel(
    const float* partial_output, const float* partial_max, const float* partial_sum,
    scalar_t* output, int rows, int head_dim, int splits) {
  const int row = blockIdx.x;
  float global_max = -INFINITY;
  if (threadIdx.x == 0) {
    for (int split = 0; split < splits; ++split)
      global_max = fmaxf(global_max, partial_max[row * splits + split]);
  }
  __shared__ float shared_max;
  if (threadIdx.x == 0) shared_max = global_max;
  __syncthreads();
  float denominator = 0.0f;
  for (int split = 0; split < splits; ++split)
    denominator += partial_sum[row * splits + split] * expf(partial_max[row * splits + split] - shared_max);
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float numerator = 0.0f;
    for (int split = 0; split < splits; ++split)
      numerator += partial_output[(static_cast<int64_t>(row) * splits + split) * head_dim + d] *
                   expf(partial_max[row * splits + split] - shared_max);
    output[static_cast<int64_t>(row) * head_dim + d] =
        static_cast<scalar_t>(denominator > 0.0f ? numerator / denominator : 0.0f);
  }
}

int blocks_for(int64_t elements) { return static_cast<int>(std::min<int64_t>((elements + kThreads - 1) / kThreads, 4096)); }

}  // namespace

void paged_kv_append_cuda(
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor key_scales, torch::Tensor value_scales,
    const torch::Tensor& block_tables, const torch::Tensor& positions,
    const torch::Tensor& key, const torch::Tensor& value) {
  c10::cuda::CUDAGuard guard(key_cache.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int batch = key.size(0), kv_heads = key.size(1), head_dim = key.size(2);
  const int num_blocks = key_cache.size(0), block_size = key_cache.size(1), max_blocks = block_tables.size(1);
  if (key_cache.scalar_type() == torch::kInt8) {
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, key.scalar_type(), "paged_append_int8", [&] {
      paged_append_int8_kernel<scalar_t><<<batch * kv_heads, kThreads, 0, stream>>>(
          key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(), key_scales.data_ptr<float>(), value_scales.data_ptr<float>(),
          block_tables.data_ptr<int64_t>(), positions.data_ptr<int64_t>(), key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
          batch, max_blocks, num_blocks, block_size, kv_heads, head_dim);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, key.scalar_type(), "paged_append_float", [&] {
      const int64_t elements = static_cast<int64_t>(batch) * kv_heads * head_dim;
      paged_append_float_kernel<scalar_t><<<blocks_for(elements), kThreads, 0, stream>>>(
          key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(), block_tables.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
          key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(), batch, max_blocks, num_blocks, block_size, kv_heads, head_dim);
    });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void paged_decode_attention_cuda(
    const torch::Tensor& query, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, const torch::Tensor& key_scales,
    const torch::Tensor& value_scales, const torch::Tensor& block_tables,
    const torch::Tensor& sequence_lengths, torch::Tensor partial_output,
    torch::Tensor partial_max, torch::Tensor partial_sum, torch::Tensor output,
    int64_t num_splits, double scale) {
  c10::cuda::CUDAGuard guard(query.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int batch = query.size(0), q_heads = query.size(1), head_dim = query.size(2);
  const int num_blocks = key_cache.size(0), block_size = key_cache.size(1), kv_heads = key_cache.size(2), max_blocks = block_tables.size(1);
  const int threads = 32;
  const int grid = batch * q_heads * static_cast<int>(num_splits);
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, query.scalar_type(), "paged_decode_attention", [&] {
    if (key_cache.scalar_type() == torch::kInt8) {
      paged_attention_split_kernel<scalar_t, int8_t, true><<<grid, threads, 0, stream>>>(
          query.data_ptr<scalar_t>(), key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(), key_scales.data_ptr<float>(), value_scales.data_ptr<float>(),
          block_tables.data_ptr<int64_t>(), sequence_lengths.data_ptr<int64_t>(), partial_output.data_ptr<float>(), partial_max.data_ptr<float>(), partial_sum.data_ptr<float>(),
          batch, q_heads, kv_heads, head_dim, num_blocks, block_size, max_blocks, num_splits, static_cast<float>(scale));
    } else {
      paged_attention_split_kernel<scalar_t, scalar_t, false><<<grid, threads, 0, stream>>>(
          query.data_ptr<scalar_t>(), key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(), nullptr, nullptr,
          block_tables.data_ptr<int64_t>(), sequence_lengths.data_ptr<int64_t>(), partial_output.data_ptr<float>(), partial_max.data_ptr<float>(), partial_sum.data_ptr<float>(),
          batch, q_heads, kv_heads, head_dim, num_blocks, block_size, max_blocks, num_splits, static_cast<float>(scale));
    }
    paged_attention_reduce_kernel<scalar_t><<<batch * q_heads, threads, 0, stream>>>(
        partial_output.data_ptr<float>(), partial_max.data_ptr<float>(), partial_sum.data_ptr<float>(), output.data_ptr<scalar_t>(), batch * q_heads, head_dim, num_splits);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
