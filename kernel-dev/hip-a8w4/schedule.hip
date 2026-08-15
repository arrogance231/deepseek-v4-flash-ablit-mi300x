#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>

template <int BM, bool DUAL>
__global__ void schedule_kernel(int *__restrict__ out,
                                int *__restrict__ out64,
                                const int *__restrict__ hist,
                                int capacity, int capacity64) {
  __shared__ int count, count64;
  if (!threadIdx.x) count = count64 = 0;
  for (int i = threadIdx.x; i < capacity; i += blockDim.x) out[i] = -1;
  if constexpr (DUAL)
    for (int i = threadIdx.x; i < capacity64; i += blockDim.x) out64[i] = -1;
  __syncthreads();
  if (threadIdx.x < 256) {
    int n = hist[threadIdx.x], blocks = (n + BM - 1) / BM;
    int pos = atomicAdd(&count, blocks);
    for (int block = 0; block < blocks; ++block)
      out[pos + block] = (block * BM << 16) | threadIdx.x;
    if constexpr (DUAL) {
      int blocks64 = (n + 63) / 64;
      int pos64 = atomicAdd(&count64, blocks64);
      for (int block = 0; block < blocks64; ++block)
        out64[pos64 + block] = (block * 64 << 16) | threadIdx.x;
    }
  }
}

void schedule(torch::Tensor out16, torch::Tensor out48,
              torch::Tensor out64, torch::Tensor hist, bool tiny) {
  auto stream = at::cuda::getCurrentCUDAStream();
  if (tiny)
    schedule_kernel<16, false><<<1, 512, 0, stream>>>(
        out16.data_ptr<int>(), nullptr, hist.data_ptr<int>(), out16.numel(), 0);
  else
    schedule_kernel<48, true><<<1, 512, 0, stream>>>(
        out48.data_ptr<int>(), out64.data_ptr<int>(), hist.data_ptr<int>(),
        out48.numel(), out64.numel());
}
