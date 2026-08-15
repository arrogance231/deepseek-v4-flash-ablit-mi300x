#include <torch/extension.h>
torch::Tensor run(torch::Tensor out, torch::Tensor x, double limit,
                  double alpha, double beta);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
