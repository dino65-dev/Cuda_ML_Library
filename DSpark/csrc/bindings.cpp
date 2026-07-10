#include <torch/extension.h>

#include "dspark_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "markov_logits",
        &dspark_markov_logits_cuda,
        "Fused DSpark low-rank Markov logit correction (CUDA)");
    module.def(
        "schedule",
        &dspark_schedule_cuda,
        "DSpark confidence-scheduled verification (CUDA)");
}
