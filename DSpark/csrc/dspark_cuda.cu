#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "dspark_cuda.h"

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kVocabItemsPerThread = 4;

template <typename scalar_t>
__device__ __forceinline__ float as_float(scalar_t value) {
    return static_cast<float>(value);
}

// The default DSpark head is a low-rank Markov correction:
//   logits[b, v] += W2[v, :] dot W1[previous_token[b], :].
// W2 is supplied transposed as [rank, vocab], so every warp reads contiguous
// vocabulary lanes. The latent row is staged once per CTA and the vocabulary
// bias is never materialized.
template <typename scalar_t>
__global__ __launch_bounds__(kThreads, 2) void markov_logits_kernel(
    const scalar_t* __restrict__ base,
    const int64_t* __restrict__ previous_ids,
    const scalar_t* __restrict__ embedding,
    const scalar_t* __restrict__ projection_t,
    scalar_t* __restrict__ output,
    int vocab_size,
    int rank,
    int vocab_tiles) {
    extern __shared__ float latent[];

    const int batch = blockIdx.x / vocab_tiles;
    const int tile = blockIdx.x - batch * vocab_tiles;
    const int64_t token = previous_ids[batch];

    if (token < 0 || token >= vocab_size) {
        const int tile_start = tile * blockDim.x * kVocabItemsPerThread;
#pragma unroll
        for (int item = 0; item < kVocabItemsPerThread; ++item) {
            const int vocab = tile_start + item * blockDim.x + threadIdx.x;
            if (vocab < vocab_size) {
                const int64_t offset =
                    static_cast<int64_t>(batch) * vocab_size + vocab;
                output[offset] = base[offset];
            }
        }
        return;
    }

    for (int r = threadIdx.x; r < rank; r += blockDim.x) {
        latent[r] = as_float(embedding[token * rank + r]);
    }
    __syncthreads();

    const int tile_start = tile * blockDim.x * kVocabItemsPerThread;
    float correction[kVocabItemsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
    for (int r = 0; r < rank; ++r) {
        const float latent_value = latent[r];
#pragma unroll
        for (int item = 0; item < kVocabItemsPerThread; ++item) {
            const int vocab = tile_start + item * blockDim.x + threadIdx.x;
            if (vocab < vocab_size) {
                correction[item] = fmaf(
                    latent_value,
                    as_float(projection_t[
                        static_cast<int64_t>(r) * vocab_size + vocab]),
                    correction[item]);
            }
        }
    }

#pragma unroll
    for (int item = 0; item < kVocabItemsPerThread; ++item) {
        const int vocab = tile_start + item * blockDim.x + threadIdx.x;
        if (vocab < vocab_size) {
            const int64_t offset =
                static_cast<int64_t>(batch) * vocab_size + vocab;
            output[offset] =
                static_cast<scalar_t>(as_float(base[offset]) + correction[item]);
        }
    }
}

// One warp handles one request. DSpark blocks are deliberately short (7 in
// the paper); keeping the full calibrated prefix scan inside a warp removes
// shared-memory traffic and all inter-CTA synchronization.
template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void build_candidates_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ temperatures,
    float* __restrict__ survival,
    uint64_t* __restrict__ keys,
    int32_t* __restrict__ candidate_ids,
    int request_count,
    int proposal_length) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const int request = blockIdx.x * warps_per_block + warp;
    if (request >= request_count) {
        return;
    }

    float prefix = 1.0f;
    if (lane < proposal_length) {
        const int index = request * proposal_length + lane;
        const float temperature = temperatures[lane];
        const float scaled_logit = as_float(logits[index]) / temperature;
        prefix = 1.0f / (1.0f + __expf(-scaled_logit));
    }

#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const float previous = __shfl_up_sync(0xffffffffu, prefix, offset);
        if (lane >= offset) {
            prefix *= previous;
        }
    }

    if (lane < proposal_length) {
        const int index = request * proposal_length + lane;
        survival[index] = prefix;

        // Positive IEEE-754 floats preserve numeric order in their bit pattern.
        // Position is the low-word tie break, so a prefix can never admit a
        // later equal-probability token before its predecessor.
        const uint64_t probability_bits = static_cast<uint64_t>(__float_as_uint(prefix));
        const uint32_t position_tie_break = 0xffffffffu - static_cast<uint32_t>(lane);
        keys[index] = (probability_bits << 32) | position_tie_break;
        candidate_ids[index] = index;
    }
}

__global__ __launch_bounds__(kThreads) void unpack_probabilities_kernel(
    const uint64_t* __restrict__ sorted_keys,
    float* __restrict__ sorted_probabilities,
    int candidate_count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < candidate_count;
         index += blockDim.x * gridDim.x) {
        const uint32_t bits = static_cast<uint32_t>(sorted_keys[index] >> 32);
        sorted_probabilities[index] = __uint_as_float(bits);
    }
}

// A candidate is admitted only while it strictly improves expected token
// throughput. atomicMin finds the first drop, which is equivalent to the
// paper's causal early break but does not serialize the evaluation on one lane.
__global__ __launch_bounds__(kThreads) void find_first_drop_kernel(
    const float* __restrict__ prefix_probability_sums,
    const float* __restrict__ step_curve,
    int32_t* __restrict__ first_drop,
    int request_count,
    int candidate_count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < candidate_count;
         index += blockDim.x * gridDim.x) {
        const float current_expected =
            static_cast<float>(request_count) + prefix_probability_sums[index];
        const int current_batch = request_count + index + 1;
        const float current_throughput = current_expected * step_curve[current_batch];

        float previous_expected = static_cast<float>(request_count);
        if (index > 0) {
            previous_expected += prefix_probability_sums[index - 1];
        }
        const int previous_batch = request_count + index;
        const float previous_throughput =
            previous_expected * step_curve[previous_batch];

        if (!(current_throughput > previous_throughput)) {
            atomicMin(first_drop, index);
        }
    }
}

__global__ __launch_bounds__(kThreads) void scatter_schedule_kernel(
    const int32_t* __restrict__ sorted_candidate_ids,
    const int32_t* __restrict__ selected_count,
    int32_t* __restrict__ lengths,
    int candidate_count,
    int proposal_length) {
    const int count = *selected_count;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < candidate_count && index < count;
         index += blockDim.x * gridDim.x) {
        const int candidate = sorted_candidate_ids[index];
        const int request = candidate / proposal_length;
        const int position = candidate - request * proposal_length;
        atomicMax(lengths + request, position + 1);
    }
}

__global__ void finalize_schedule_kernel(
    const float* __restrict__ prefix_probability_sums,
    const float* __restrict__ step_curve,
    const int32_t* __restrict__ selected_count,
    float* __restrict__ expected_tokens,
    float* __restrict__ expected_throughput,
    int request_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int count = *selected_count;
    float expected = static_cast<float>(request_count);
    if (count > 0) {
        expected += prefix_probability_sums[count - 1];
    }
    expected_tokens[0] = expected;
    expected_throughput[0] = expected * step_curve[request_count + count];
}

inline int launch_blocks(int elements) {
    return std::min((elements + kThreads - 1) / kThreads, 4096);
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor dspark_markov_logits_cuda(
    const torch::Tensor& base_logits,
    const torch::Tensor& previous_token_ids,
    const torch::Tensor& token_embedding,
    const torch::Tensor& projection_t) {
    check_cuda_contiguous(base_logits, "base_logits");
    check_cuda_contiguous(previous_token_ids, "previous_token_ids");
    check_cuda_contiguous(token_embedding, "token_embedding");
    check_cuda_contiguous(projection_t, "projection_t");

    TORCH_CHECK(base_logits.dim() == 2, "base_logits must have shape [batch, vocab]");
    TORCH_CHECK(previous_token_ids.dim() == 1, "previous_token_ids must have shape [batch]");
    TORCH_CHECK(token_embedding.dim() == 2, "token_embedding must have shape [vocab, rank]");
    TORCH_CHECK(projection_t.dim() == 2, "projection_t must have shape [rank, vocab]");
    TORCH_CHECK(previous_token_ids.scalar_type() == torch::kInt64,
                "previous_token_ids must be torch.int64");
    TORCH_CHECK(base_logits.scalar_type() == token_embedding.scalar_type() &&
                    base_logits.scalar_type() == projection_t.scalar_type(),
                "base_logits, token_embedding, and projection_t must have the same dtype");
    TORCH_CHECK(base_logits.device() == previous_token_ids.device() &&
                    base_logits.device() == token_embedding.device() &&
                    base_logits.device() == projection_t.device(),
                "all inputs must be on the same CUDA device");
    TORCH_CHECK(base_logits.scalar_type() == torch::kFloat32 ||
                    base_logits.scalar_type() == torch::kFloat16 ||
                    base_logits.scalar_type() == torch::kBFloat16,
                "Markov kernel supports float32, float16, and bfloat16");

    const int64_t batch = base_logits.size(0);
    const int64_t vocab = base_logits.size(1);
    const int64_t rank = token_embedding.size(1);
    TORCH_CHECK(batch > 0 && vocab > 0 && rank > 0, "input dimensions must be positive");
    TORCH_CHECK(batch <= std::numeric_limits<int>::max() &&
                    vocab <= std::numeric_limits<int>::max() &&
                    rank <= std::numeric_limits<int>::max(),
                "tensor dimensions exceed CUDA kernel limits");
    TORCH_CHECK(previous_token_ids.size(0) == batch, "batch dimension mismatch");
    TORCH_CHECK(token_embedding.size(0) == vocab, "embedding vocabulary mismatch");
    TORCH_CHECK(projection_t.size(0) == rank && projection_t.size(1) == vocab,
                "projection_t must have shape [rank, vocab]");
    TORCH_CHECK(rank <= 4096, "rank > 4096 is not supported by the fused kernel");

    c10::cuda::CUDAGuard device_guard(base_logits.device());
    const auto output = torch::empty_like(base_logits);
    constexpr int vocab_values_per_cta = kThreads * kVocabItemsPerThread;
    const int vocab_tiles =
        (static_cast<int>(vocab) + vocab_values_per_cta - 1) /
        vocab_values_per_cta;
    const int64_t grid_size = batch * vocab_tiles;
    TORCH_CHECK(grid_size <= std::numeric_limits<int>::max(), "launch grid is too large");
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kFloat16,
        torch::kBFloat16,
        base_logits.scalar_type(),
        "dspark_markov_logits_cuda",
        [&] {
            markov_logits_kernel<scalar_t><<<
                static_cast<int>(grid_size),
                kThreads,
                static_cast<size_t>(rank) * sizeof(float),
                stream>>>(
                base_logits.data_ptr<scalar_t>(),
                previous_token_ids.data_ptr<int64_t>(),
                token_embedding.data_ptr<scalar_t>(),
                projection_t.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<int>(vocab),
                static_cast<int>(rank),
                vocab_tiles);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> dspark_schedule_cuda(
    const torch::Tensor& confidence_logits,
    const torch::Tensor& step_curve,
    const torch::Tensor& temperatures) {
    check_cuda_contiguous(confidence_logits, "confidence_logits");
    check_cuda_contiguous(step_curve, "step_curve");
    check_cuda_contiguous(temperatures, "temperatures");

    TORCH_CHECK(confidence_logits.dim() == 2,
                "confidence_logits must have shape [requests, proposal_length]");
    TORCH_CHECK(step_curve.dim() == 1, "step_curve must be one-dimensional");
    TORCH_CHECK(temperatures.dim() == 1, "temperatures must be one-dimensional");
    TORCH_CHECK(step_curve.scalar_type() == torch::kFloat32,
                "step_curve must be torch.float32");
    TORCH_CHECK(temperatures.scalar_type() == torch::kFloat32,
                "temperatures must be torch.float32");
    TORCH_CHECK(confidence_logits.scalar_type() == torch::kFloat32 ||
                    confidence_logits.scalar_type() == torch::kFloat16 ||
                    confidence_logits.scalar_type() == torch::kBFloat16,
                "confidence_logits supports float32, float16, and bfloat16");
    TORCH_CHECK(confidence_logits.device() == step_curve.device() &&
                    confidence_logits.device() == temperatures.device(),
                "all inputs must be on the same CUDA device");

    const int64_t request_count_64 = confidence_logits.size(0);
    const int64_t proposal_length_64 = confidence_logits.size(1);
    TORCH_CHECK(request_count_64 > 0, "at least one active request is required");
    TORCH_CHECK(proposal_length_64 > 0 && proposal_length_64 <= kWarpSize,
                "proposal_length must be in [1, 32]");
    TORCH_CHECK(temperatures.size(0) == proposal_length_64,
                "temperatures must contain one value per proposal position");

    const int64_t candidate_count_64 = request_count_64 * proposal_length_64;
    TORCH_CHECK(request_count_64 <= std::numeric_limits<int>::max() &&
                    candidate_count_64 <= std::numeric_limits<int>::max(),
                "candidate count exceeds CUDA/CUB limits");
    const int request_count = static_cast<int>(request_count_64);
    const int proposal_length = static_cast<int>(proposal_length_64);
    const int candidate_count = static_cast<int>(candidate_count_64);
    TORCH_CHECK(step_curve.size(0) > request_count + candidate_count,
                "step_curve must be indexed through requests * (proposal_length + 1)");

    c10::cuda::CUDAGuard device_guard(confidence_logits.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto float_options = confidence_logits.options().dtype(torch::kFloat32);
    const auto int_options = confidence_logits.options().dtype(torch::kInt32);
    const auto long_options = confidence_logits.options().dtype(torch::kInt64);
    const auto byte_options = confidence_logits.options().dtype(torch::kUInt8);

    auto survival = torch::empty_like(confidence_logits, float_options);
    auto keys_in = torch::empty({candidate_count}, long_options);
    auto keys_out = torch::empty({candidate_count}, long_options);
    auto ids_in = torch::empty({candidate_count}, int_options);
    auto ids_out = torch::empty({candidate_count}, int_options);

    const int warps_per_block = kThreads / kWarpSize;
    const int candidate_blocks =
        (request_count + warps_per_block - 1) / warps_per_block;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kFloat16,
        torch::kBFloat16,
        confidence_logits.scalar_type(),
        "dspark_build_candidates_cuda",
        [&] {
            build_candidates_kernel<scalar_t><<<candidate_blocks, kThreads, 0, stream>>>(
                confidence_logits.data_ptr<scalar_t>(),
                temperatures.data_ptr<float>(),
                survival.data_ptr<float>(),
                reinterpret_cast<uint64_t*>(keys_in.data_ptr<int64_t>()),
                ids_in.data_ptr<int32_t>(),
                request_count,
                proposal_length);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    size_t sort_storage_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        sort_storage_bytes,
        reinterpret_cast<uint64_t*>(keys_in.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(keys_out.data_ptr<int64_t>()),
        ids_in.data_ptr<int32_t>(),
        ids_out.data_ptr<int32_t>(),
        candidate_count,
        0,
        64,
        stream));
    auto sort_storage = torch::empty(
        {static_cast<int64_t>(sort_storage_bytes)}, byte_options);
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        sort_storage.data_ptr(),
        sort_storage_bytes,
        reinterpret_cast<uint64_t*>(keys_in.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(keys_out.data_ptr<int64_t>()),
        ids_in.data_ptr<int32_t>(),
        ids_out.data_ptr<int32_t>(),
        candidate_count,
        0,
        64,
        stream));

    auto sorted_probabilities = torch::empty({candidate_count}, float_options);
    unpack_probabilities_kernel<<<launch_blocks(candidate_count), kThreads, 0, stream>>>(
        reinterpret_cast<uint64_t*>(keys_out.data_ptr<int64_t>()),
        sorted_probabilities.data_ptr<float>(),
        candidate_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto prefix_probability_sums = torch::empty({candidate_count}, float_options);
    size_t scan_storage_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr,
        scan_storage_bytes,
        sorted_probabilities.data_ptr<float>(),
        prefix_probability_sums.data_ptr<float>(),
        candidate_count,
        stream));
    auto scan_storage = torch::empty(
        {static_cast<int64_t>(scan_storage_bytes)}, byte_options);
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scan_storage.data_ptr(),
        scan_storage_bytes,
        sorted_probabilities.data_ptr<float>(),
        prefix_probability_sums.data_ptr<float>(),
        candidate_count,
        stream));

    // candidate_count is both the no-drop sentinel and the selected count when
    // every extension improves throughput.
    auto selected_count = torch::full({1}, candidate_count, int_options);
    find_first_drop_kernel<<<launch_blocks(candidate_count), kThreads, 0, stream>>>(
        prefix_probability_sums.data_ptr<float>(),
        step_curve.data_ptr<float>(),
        selected_count.data_ptr<int32_t>(),
        request_count,
        candidate_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto lengths = torch::zeros({request_count}, int_options);
    scatter_schedule_kernel<<<launch_blocks(candidate_count), kThreads, 0, stream>>>(
        ids_out.data_ptr<int32_t>(),
        selected_count.data_ptr<int32_t>(),
        lengths.data_ptr<int32_t>(),
        candidate_count,
        proposal_length);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto expected_tokens = torch::empty({1}, float_options);
    auto expected_throughput = torch::empty({1}, float_options);
    finalize_schedule_kernel<<<1, 1, 0, stream>>>(
        prefix_probability_sums.data_ptr<float>(),
        step_curve.data_ptr<float>(),
        selected_count.data_ptr<int32_t>(),
        expected_tokens.data_ptr<float>(),
        expected_throughput.data_ptr<float>(),
        request_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        lengths,
        survival,
        selected_count,
        expected_tokens,
        expected_throughput,
    };
}
