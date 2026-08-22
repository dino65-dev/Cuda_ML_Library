#include <torch/extension.h>

#include <optional>
#include <tuple>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor> residual_rms_norm_cuda(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps);

std::tuple<torch::Tensor, torch::Tensor> rms_norm_quantize_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps);

std::tuple<torch::Tensor, torch::Tensor> rope_qk_norm_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    double eps);

void kv_cache_append_cuda(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    const torch::Tensor& slots,
    const torch::Tensor& key,
    const torch::Tensor& value);

torch::Tensor bias_swiglu_cuda(
    const torch::Tensor& gate,
    const torch::Tensor& up,
    const std::optional<torch::Tensor>& gate_bias,
    const std::optional<torch::Tensor>& up_bias);

void paged_kv_append_cuda(
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor key_scales, torch::Tensor value_scales,
    const torch::Tensor& block_tables, const torch::Tensor& positions,
    const torch::Tensor& key, const torch::Tensor& value);

void paged_decode_attention_cuda(
    const torch::Tensor& query, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, const torch::Tensor& key_scales,
    const torch::Tensor& value_scales, const torch::Tensor& block_tables,
    const torch::Tensor& sequence_lengths, torch::Tensor partial_output,
    torch::Tensor partial_max, torch::Tensor partial_sum, torch::Tensor output,
    int64_t num_splits, double scale);

namespace {

void check_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_floating(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat ||
          tensor.scalar_type() == torch::kHalf ||
          tensor.scalar_type() == torch::kBFloat16,
      name,
      " must have dtype float32, float16, or bfloat16");
}

void check_same_device(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  TORCH_CHECK(lhs.device() == rhs.device(), "all tensors must be on the same CUDA device");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> residual_rms_norm(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  check_cuda(input, "input");
  check_cuda(residual, "residual");
  check_cuda(weight, "weight");
  check_floating(input, "input");
  TORCH_CHECK(input.sizes() == residual.sizes(), "input and residual must have identical shapes");
  TORCH_CHECK(input.scalar_type() == residual.scalar_type(), "input and residual must have identical dtypes");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == input.size(-1), "weight must have shape [hidden_size]");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "weight dtype must match input dtype");
  TORCH_CHECK(input.size(-1) > 0, "hidden_size must be positive");
  TORCH_CHECK(eps > 0.0, "eps must be positive");
  check_same_device(input, residual);
  check_same_device(input, weight);
  return residual_rms_norm_cuda(input, residual, weight, eps);
}

std::tuple<torch::Tensor, torch::Tensor> rms_norm_quantize(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps) {
  check_cuda(input, "input");
  check_cuda(weight, "weight");
  check_floating(input, "input");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == input.size(-1), "weight must have shape [hidden_size]");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "weight dtype must match input dtype");
  TORCH_CHECK(input.size(-1) > 0, "hidden_size must be positive");
  TORCH_CHECK(eps > 0.0, "eps must be positive");
  check_same_device(input, weight);
  return rms_norm_quantize_cuda(input, weight, eps);
}

std::tuple<torch::Tensor, torch::Tensor> rope_qk_norm(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    double eps) {
  for (const auto& item : std::vector<std::pair<const torch::Tensor*, const char*>>{
           {&q, "q"}, {&k, "k"}, {&q_weight, "q_weight"},
           {&k_weight, "k_weight"}, {&cos, "cos"}, {&sin, "sin"}}) {
    check_cuda(*item.first, item.second);
    check_floating(*item.first, item.second);
    check_same_device(q, *item.first);
  }
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q and k must have shape [batch, sequence, heads, head_dim]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) && q.size(3) == k.size(3),
              "q and k batch, sequence, and head_dim must match");
  const auto head_dim = q.size(3);
  TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0, "head_dim must be a positive even number");
  TORCH_CHECK(q_weight.dim() == 1 && q_weight.numel() == head_dim, "q_weight must have shape [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1 && k_weight.numel() == head_dim, "k_weight must have shape [head_dim]");
  TORCH_CHECK(cos.dim() == 2 && cos.size(0) == q.size(1) && cos.size(1) == head_dim / 2,
              "cos must have shape [sequence, head_dim / 2]");
  TORCH_CHECK(sin.sizes() == cos.sizes(), "sin must have the same shape as cos");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == q_weight.scalar_type() &&
              q.scalar_type() == k_weight.scalar_type() && q.scalar_type() == cos.scalar_type() &&
              q.scalar_type() == sin.scalar_type(), "all floating tensors must have the same dtype");
  TORCH_CHECK(eps > 0.0, "eps must be positive");
  return rope_qk_norm_cuda(q, k, q_weight, k_weight, cos, sin, eps);
}

void kv_cache_append(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    const torch::Tensor& slots,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  check_cuda(key_cache, "key_cache");
  check_cuda(value_cache, "value_cache");
  check_cuda(slots, "slots");
  check_cuda(key, "key");
  check_cuda(value, "value");
  check_floating(key_cache, "key_cache");
  TORCH_CHECK(slots.scalar_type() == torch::kLong, "slots must have dtype int64");
  TORCH_CHECK(key_cache.dim() == 3 && value_cache.sizes() == key_cache.sizes(),
              "key_cache and value_cache must have shape [capacity, heads, head_dim]");
  TORCH_CHECK(key.dim() == 3 && value.sizes() == key.sizes(),
              "key and value must have shape [tokens, heads, head_dim]");
  TORCH_CHECK(key.size(1) == key_cache.size(1) && key.size(2) == key_cache.size(2),
              "cache and update head dimensions must match");
  TORCH_CHECK(slots.dim() == 1 && slots.numel() == key.size(0), "slots must have shape [tokens]");
  TORCH_CHECK(key.scalar_type() == value.scalar_type() && key.scalar_type() == key_cache.scalar_type() &&
              key.scalar_type() == value_cache.scalar_type(), "cache and update dtypes must match");
  check_same_device(key_cache, value_cache);
  check_same_device(key_cache, slots);
  check_same_device(key_cache, key);
  check_same_device(key_cache, value);
  kv_cache_append_cuda(key_cache, value_cache, slots, key, value);
}

torch::Tensor bias_swiglu(
    const torch::Tensor& gate,
    const torch::Tensor& up,
    const std::optional<torch::Tensor>& gate_bias,
    const std::optional<torch::Tensor>& up_bias) {
  check_cuda(gate, "gate");
  check_cuda(up, "up");
  check_floating(gate, "gate");
  TORCH_CHECK(gate.dim() >= 1, "gate and up must have at least one dimension");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have identical shapes");
  TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate and up dtypes must match");
  check_same_device(gate, up);
  const auto hidden = gate.size(-1);
  for (const auto& item : std::vector<std::pair<const std::optional<torch::Tensor>*, const char*>>{
           {&gate_bias, "gate_bias"}, {&up_bias, "up_bias"}}) {
    if (item.first->has_value()) {
      const auto& bias = item.first->value();
      check_cuda(bias, item.second);
      TORCH_CHECK(bias.dim() == 1 && bias.numel() == hidden, item.second, " must have shape [hidden_size]");
      TORCH_CHECK(bias.scalar_type() == gate.scalar_type(), item.second, " dtype must match gate dtype");
      check_same_device(gate, bias);
    }
  }
  return bias_swiglu_cuda(gate, up, gate_bias, up_bias);
}

void paged_kv_append(
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor key_scales, torch::Tensor value_scales,
    const torch::Tensor& block_tables, const torch::Tensor& positions,
    const torch::Tensor& key, const torch::Tensor& value) {
  for (const auto& item : std::vector<std::pair<const torch::Tensor*, const char*>>{
           {&key_cache, "key_cache"}, {&value_cache, "value_cache"},
           {&block_tables, "block_tables"}, {&positions, "positions"},
           {&key, "key"}, {&value, "value"}}) {
    check_cuda(*item.first, item.second);
    check_same_device(key_cache, *item.first);
  }
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes(),
              "cache must have shape [blocks, block_size, kv_heads, head_dim]");
  TORCH_CHECK(key.dim() == 3 && value.sizes() == key.sizes(),
              "updates must have shape [batch, kv_heads, head_dim]");
  TORCH_CHECK(key.size(1) == key_cache.size(2) && key.size(2) == key_cache.size(3),
              "update and cache KV dimensions differ");
  TORCH_CHECK(block_tables.dim() == 2 && block_tables.size(0) == key.size(0),
              "block_tables must have shape [batch, max_blocks]");
  TORCH_CHECK(positions.dim() == 1 && positions.size(0) == key.size(0),
              "positions must have shape [batch]");
  TORCH_CHECK(block_tables.scalar_type() == torch::kLong && positions.scalar_type() == torch::kLong,
              "block_tables and positions must use int64");
  TORCH_CHECK(key.scalar_type() == value.scalar_type(), "key/value update dtypes differ");
  const bool quantized = key_cache.scalar_type() == torch::kInt8;
  if (quantized) {
    check_cuda(key_scales, "key_scales"); check_cuda(value_scales, "value_scales");
    check_same_device(key_cache, key_scales); check_same_device(key_cache, value_scales);
    TORCH_CHECK(key_scales.sizes() == key_cache.sizes().slice(0, 3) &&
                value_scales.sizes() == key_scales.sizes(),
                "quantized scale shape must be [blocks, block_size, kv_heads]");
    TORCH_CHECK(key_scales.scalar_type() == torch::kFloat && value_scales.scalar_type() == torch::kFloat,
                "quantized scales must use float32");
    check_floating(key, "key");
  } else {
    check_floating(key_cache, "key_cache");
    TORCH_CHECK(key.scalar_type() == key_cache.scalar_type() &&
                value_cache.scalar_type() == key_cache.scalar_type(),
                "floating cache/update dtypes must match");
  }
  paged_kv_append_cuda(key_cache, value_cache, key_scales, value_scales,
                       block_tables, positions, key, value);
}

void paged_decode_attention(
    const torch::Tensor& query, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, const torch::Tensor& key_scales,
    const torch::Tensor& value_scales, const torch::Tensor& block_tables,
    const torch::Tensor& sequence_lengths, torch::Tensor partial_output,
    torch::Tensor partial_max, torch::Tensor partial_sum, torch::Tensor output,
    int64_t num_splits, double scale) {
  for (const auto& item : std::vector<std::pair<const torch::Tensor*, const char*>>{
           {&query, "query"}, {&key_cache, "key_cache"}, {&value_cache, "value_cache"},
           {&block_tables, "block_tables"}, {&sequence_lengths, "sequence_lengths"},
           {&partial_output, "partial_output"}, {&partial_max, "partial_max"},
           {&partial_sum, "partial_sum"}, {&output, "output"}}) {
    check_cuda(*item.first, item.second); check_same_device(query, *item.first);
  }
  check_floating(query, "query");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, query_heads, head_dim]");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes(),
              "cache must have shape [blocks, block_size, kv_heads, head_dim]");
  TORCH_CHECK(query.size(2) == key_cache.size(3), "query/cache head_dim differs");
  TORCH_CHECK(query.size(1) % key_cache.size(2) == 0,
              "query_heads must be divisible by kv_heads");
  TORCH_CHECK(query.size(2) > 0 && query.size(2) <= 256,
              "head_dim must be in [1, 256]");
  TORCH_CHECK(block_tables.dim() == 2 && block_tables.size(0) == query.size(0),
              "block_tables must have shape [batch, max_blocks]");
  TORCH_CHECK(sequence_lengths.dim() == 1 && sequence_lengths.size(0) == query.size(0),
              "sequence_lengths must have shape [batch]");
  TORCH_CHECK(block_tables.scalar_type() == torch::kLong && sequence_lengths.scalar_type() == torch::kLong,
              "block_tables and sequence_lengths must use int64");
  TORCH_CHECK(num_splits > 0 && num_splits <= 64, "num_splits must be in [1, 64]");
  TORCH_CHECK(partial_output.sizes() == torch::IntArrayRef({query.size(0), query.size(1), num_splits, query.size(2)}),
              "partial_output has the wrong shape");
  TORCH_CHECK(partial_max.sizes() == torch::IntArrayRef({query.size(0), query.size(1), num_splits}) &&
              partial_sum.sizes() == partial_max.sizes(), "partial scalar workspace has the wrong shape");
  TORCH_CHECK(output.sizes() == query.sizes() && output.scalar_type() == query.scalar_type(),
              "output must match query");
  TORCH_CHECK(partial_output.scalar_type() == torch::kFloat && partial_max.scalar_type() == torch::kFloat &&
              partial_sum.scalar_type() == torch::kFloat, "partial workspace must use float32");
  const bool quantized = key_cache.scalar_type() == torch::kInt8;
  if (quantized) {
    check_cuda(key_scales, "key_scales"); check_cuda(value_scales, "value_scales");
    TORCH_CHECK(key_scales.sizes() == key_cache.sizes().slice(0, 3) && value_scales.sizes() == key_scales.sizes(),
                "quantized scale shape must be [blocks, block_size, kv_heads]");
  } else {
    TORCH_CHECK(key_cache.scalar_type() == query.scalar_type() && value_cache.scalar_type() == query.scalar_type(),
                "floating query/cache dtypes must match");
  }
  paged_decode_attention_cuda(query, key_cache, value_cache, key_scales, value_scales,
      block_tables, sequence_lengths, partial_output, partial_max, partial_sum,
      output, num_splits, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("residual_rms_norm", &residual_rms_norm);
  module.def("rms_norm_quantize", &rms_norm_quantize);
  module.def("rope_qk_norm", &rope_qk_norm);
  module.def("kv_cache_append", &kv_cache_append);
  module.def("bias_swiglu", &bias_swiglu);
  module.def("paged_kv_append", &paged_kv_append);
  module.def("paged_decode_attention", &paged_decode_attention);
}
