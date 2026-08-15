#include <torch/extension.h>

torch::Tensor w2(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
                 torch::Tensor q, torch::Tensor hist, torch::Tensor offsets,
                 torch::Tensor gather, torch::Tensor scatter,
                 torch::Tensor gate, torch::Tensor schedule,
                 torch::Tensor schedule16, torch::Tensor scale, bool tiny);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &w2); }
