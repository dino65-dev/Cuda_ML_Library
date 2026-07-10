#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include "dspark_cuda.h"

torch::Tensor dspark_markov_logits(
    const torch::Tensor& base_logits,
    const torch::Tensor& previous_token_ids,
    const torch::Tensor& token_embedding,
    const torch::Tensor& projection_t) {
    TORCH_CHECK(base_logits.is_cuda(), "base_logits must be a CUDA tensor");
    TORCH_CHECK(previous_token_ids.is_cuda(),
                "previous_token_ids must be a CUDA tensor");
    TORCH_CHECK(token_embedding.is_cuda(),
                "token_embedding must be a CUDA tensor");
    TORCH_CHECK(projection_t.is_cuda(), "projection_t must be a CUDA tensor");
    TORCH_CHECK(base_logits.dim() == 2,
                "base_logits must have shape [batch, vocab]");
    TORCH_CHECK(previous_token_ids.dim() == 1,
                "previous_token_ids must have shape [batch]");
    TORCH_CHECK(token_embedding.dim() == 2,
                "token_embedding must have shape [vocab, rank]");
    TORCH_CHECK(projection_t.dim() == 2,
                "projection_t must have shape [rank, vocab]");
    TORCH_CHECK(previous_token_ids.scalar_type() == torch::kInt64,
                "previous_token_ids must be torch.int64");
    TORCH_CHECK(base_logits.scalar_type() == token_embedding.scalar_type() &&
                    base_logits.scalar_type() == projection_t.scalar_type(),
                "Markov tensors must have the same dtype");
    TORCH_CHECK(base_logits.device() == previous_token_ids.device() &&
                    base_logits.device() == token_embedding.device() &&
                    base_logits.device() == projection_t.device(),
                "Markov tensors must be on the same CUDA device");

    const auto batch = base_logits.size(0);
    const auto vocab = base_logits.size(1);
    const auto rank = token_embedding.size(1);
    TORCH_CHECK(previous_token_ids.size(0) == batch,
                "previous_token_ids batch dimension mismatch");
    TORCH_CHECK(token_embedding.size(0) == vocab,
                "token_embedding vocabulary mismatch");
    TORCH_CHECK(projection_t.size(0) == rank &&
                    projection_t.size(1) == vocab,
                "projection_t must have shape [rank, vocab]");

    // Pascal has no Tensor Cores and cuBLAS setup dominates at batch 1-2.
    // Keep newer architectures on GEMM, where vendor kernels are consistently
    // faster even for decode microbatches.
    const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
    if (properties->major < 7 && batch <= 2) {
        return dspark_markov_logits_cuda(
            base_logits,
            previous_token_ids,
            token_embedding,
            projection_t);
    }

    // One GEMM lets cuBLAS reuse W2 tiles across all requests and select a
    // Tensor Core implementation. addmm folds base_logits into the GEMM
    // update (beta=1), avoiding the separate vocabulary-sized add kernel used
    // by the eager `base + latent @ projection` reference.
    const auto latent = token_embedding.index_select(0, previous_token_ids);
    return at::addmm(base_logits, latent, projection_t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "markov_logits",
        &dspark_markov_logits,
        "Tensor-Core DSpark low-rank Markov correction (CUDA)");
    module.def(
        "markov_logits_raw",
        &dspark_markov_logits_cuda,
        "Experimental scalar DSpark Markov correction (CUDA)");
    module.def(
        "schedule",
        &dspark_schedule_cuda,
        "DSpark confidence-scheduled verification (CUDA)");
}
