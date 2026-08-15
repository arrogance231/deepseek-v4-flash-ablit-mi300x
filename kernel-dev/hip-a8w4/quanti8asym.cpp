#include <torch/extension.h>

torch::Tensor run(torch::Tensor q, torch::Tensor zero, torch::Tensor scale,
                  torch::Tensor x);
void prepare(torch::Tensor q, torch::Tensor scale, torch::Tensor sums);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("prepare", &prepare);
}
