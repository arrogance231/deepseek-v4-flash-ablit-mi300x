#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ int fp4(uint8_t v) {
  constexpr uint64_t lut = 0x0c08060403020100ull;
  int mag = (lut >> ((v & 7) * 8)) & 255;
  return (v & 8) && mag ? -mag : mag;
}

__global__ void prepare_kernel(uint8_t *q, uint8_t *scale,
                               int *sums, int k) {
  int row = blockIdx.x, lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  uint8_t *qr = q + static_cast<int64_t>(row) * k / 2;
  uint8_t *sr = scale + static_cast<int64_t>(row) * k / 32;
  int sum = 0;
  for (int i = threadIdx.x; i < k / 2; i += blockDim.x) {
    uint8_t v = qr[i], lo = v & 15, hi = v >> 4;
    if (lo == 8) lo = 0;
    if (hi == 8) hi = 0;
    qr[i] = lo | (hi << 4);
    int shift = min(max(static_cast<int>(sr[i / 16]) - 120, 0), 3);
    sum += (fp4(lo) + fp4(hi)) * (1 << shift);
  }
  for (int d = 32; d; d >>= 1) sum += __shfl_down(sum, d);
  __shared__ int partial[4];
  if (!lane) partial[wave] = sum;
  __syncthreads();
  if (!wave) {
    sum = lane < 4 ? partial[lane] : 0;
    for (int d = 32; d; d >>= 1) sum += __shfl_down(sum, d);
    if (!lane) sums[row] = sum;
  }
  __syncthreads();
  if (threadIdx.x < k / 32)
    sr[threadIdx.x] = min(max(static_cast<int>(sr[threadIdx.x]) - 120, 0), 3);
}

__global__ void quant_kernel(const uint16_t *__restrict__ x,
                             int8_t *__restrict__ q, int *__restrict__ zero,
                             float *__restrict__ scale, int k) {
  int row = blockIdx.x, lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  float lo = INFINITY, hi = -INFINITY;
  for (int pair = threadIdx.x; pair < k / 2; pair += blockDim.x) {
    uint32_t v = reinterpret_cast<const uint32_t *>(x + row * k)[pair];
    float a = __uint_as_float((v & 0xffffu) << 16);
    float b = __uint_as_float(v & 0xffff0000u);
    lo = fminf(lo, fminf(a, b));
    hi = fmaxf(hi, fmaxf(a, b));
  }
  for (int d = 32; d; d >>= 1) {
    lo = fminf(lo, __shfl_down(lo, d));
    hi = fmaxf(hi, __shfl_down(hi, d));
  }
  __shared__ float minima[4], maxima[4], inv;
  __shared__ int zp;
  if (!lane) { minima[wave] = lo; maxima[wave] = hi; }
  __syncthreads();
  if (!threadIdx.x) {
    lo = fminf(fminf(minima[0], minima[1]), fminf(minima[2], minima[3]));
    hi = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
    float s = fmaxf((hi - lo) / 255.0f, 1e-12f);
    zp = max(-128, min(127, __float2int_rn(-128.0f - lo / s)));
    scale[row] = s;
    zero[row] = zp;
    inv = 1.0f / s;
  }
  __syncthreads();
  for (int pair = threadIdx.x; pair < k / 2; pair += blockDim.x) {
    uint32_t v = reinterpret_cast<const uint32_t *>(x + row * k)[pair];
    float a = __uint_as_float((v & 0xffffu) << 16);
    float b = __uint_as_float(v & 0xffff0000u);
    int qa = max(-128, min(127, __float2int_rn(a * inv) + zp));
    int qb = max(-128, min(127, __float2int_rn(b * inv) + zp));
    reinterpret_cast<uint16_t *>(q + static_cast<int64_t>(row) * k)[pair] =
        (qa & 255) | ((qb & 255) << 8);
  }
}

torch::Tensor run(torch::Tensor q, torch::Tensor zero, torch::Tensor scale,
                  torch::Tensor x) {
  quant_kernel<<<x.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint16_t *>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<int8_t *>(q.data_ptr()),
      reinterpret_cast<int *>(zero.data_ptr()), scale.data_ptr<float>(),
      x.size(1));
  return scale;
}

void prepare(torch::Tensor q, torch::Tensor scale, torch::Tensor sums) {
  int rows = q.size(0) * q.size(1), k = q.size(2) * 2;
  prepare_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<uint8_t>(), scale.data_ptr<uint8_t>(), sums.data_ptr<int>(), k);
}
