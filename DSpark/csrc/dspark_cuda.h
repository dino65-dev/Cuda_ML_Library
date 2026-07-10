#pragma once

#include <torch/extension.h>

#include <vector>

torch::Tensor dspark_markov_logits_cuda(
    const torch::Tensor& base_logits,
    const torch::Tensor& previous_token_ids,
    const torch::Tensor& token_embedding,
    const torch::Tensor& projection_t);

torch::Tensor dspark_markov_logits(
    const torch::Tensor& base_logits,
    const torch::Tensor& previous_token_ids,
    const torch::Tensor& token_embedding,
    const torch::Tensor& projection_t);

std::vector<torch::Tensor> dspark_schedule_cuda(
    const torch::Tensor& confidence_logits,
    const torch::Tensor& step_curve,
    const torch::Tensor& temperatures);
