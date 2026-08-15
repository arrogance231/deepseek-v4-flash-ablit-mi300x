#include <torch/extension.h>

torch::Tensor fusedi8_w1(torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor, bool,
                         double);
torch::Tensor convert(torch::Tensor, torch::Tensor, torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &fusedi8_w1);
  m.def("convert", &convert);
}
