#include <torch/extension.h>
void schedule(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &schedule); }
