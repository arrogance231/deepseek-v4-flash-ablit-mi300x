#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float bf16_to_float(uint16_t x) {
  return __uint_as_float(static_cast<uint32_t>(x) << 16);
}
__device__ __forceinline__ uint16_t float_to_bf16(float x) {
  uint32_t bits = __float_as_uint(x);
  bits += 0x7fffu + ((bits >> 16) & 1);
  return bits >> 16;
}
__device__ __forceinline__ float bf16_round(float x) {
  return bf16_to_float(float_to_bf16(x));
}

__global__ void swiglu_clamp_kernel(uint16_t *__restrict__ out,
                                    const uint16_t *__restrict__ x, int rows,
                                    int width, float limit, float alpha,
                                    float beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = rows * width;
  if (i >= n) return;
  int row = i / width, col = i - row * width;
  float gate = bf16_to_float(x[row * width * 2 + col]);
  float up = bf16_to_float(x[row * width * 2 + width + col]);
  if (gate > limit) gate = limit;
  if (up < -limit) up = -limit;
  if (up > limit) up = limit;
  gate = bf16_round(gate);
  up = bf16_round(up);
  float scaled = bf16_round(alpha * gate);
  float sig = bf16_round(1.0f / (1.0f + expf(-scaled)));
  float gated = bf16_round(gate * sig);
  float shifted = bf16_round(up + beta);
  out[i] = float_to_bf16(gated * shifted);
}

torch::Tensor run(torch::Tensor out, torch::Tensor x, double limit,
                  double alpha, double beta) {
  int width = x.size(-1) / 2, rows = x.numel() / (width * 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  int n = rows * width;
  swiglu_clamp_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<uint16_t *>(out.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint16_t *>(x.data_ptr<at::BFloat16>()), rows,
      width, static_cast<float>(limit), static_cast<float>(alpha),
      static_cast<float>(beta));
  return out;
}
