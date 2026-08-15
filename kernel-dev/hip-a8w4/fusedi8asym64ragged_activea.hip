#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>

using i32x4 = int __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void direct_load_lds(
    void *dst, __amdgpu_buffer_rsrc_t src, int offset) {
  auto m0 = __builtin_amdgcn_readfirstlane(reinterpret_cast<uintptr_t>(dst));
  asm volatile("s_mov_b32 m0, %0; \n\tbuffer_load_dword %1, %2, 0 offen lds"
               :: "s"(m0), "v"(offset), "s"(src) : "memory");
}

__device__ __forceinline__ __half bf16_to_half(float x) {
  uint32_t bits = __float_as_uint(x);
  bits += 0x7fffu + ((bits >> 16) & 1);
  return __float2half_rn(__uint_as_float(bits & 0xffff0000u));
}

__device__ __forceinline__ uint32_t fp4x4_to_i8(
    uint32_t q, uint32_t idx, uint32_t lut_lo, uint32_t lut_hi,
    int sign_shift) {
  uint32_t mag = __builtin_amdgcn_perm(lut_hi, lut_lo, idx);
  uint32_t negative = (q >> sign_shift) & 0x01010101u;
  uint32_t mask = negative * 255;
  return (mag ^ mask) + negative;
}

__device__ __forceinline__ uint64_t fp4x8_to_i8(uint32_t q, uint8_t s) {
  int d = static_cast<int>(s);
  uint64_t lut = 0x0c08060403020100ull << d;
  uint32_t lo = fp4x4_to_i8(q, q & 0x07070707u,
                             static_cast<uint32_t>(lut), lut >> 32, 3);
  uint32_t hi = fp4x4_to_i8(q, (q & 0x70707070u) >> 4,
                             static_cast<uint32_t>(lut), lut >> 32, 7);
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

__global__ void convert_kernel(uint64_t *out, const uint32_t *q,
                               const uint8_t *s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = fp4x8_to_i8(q[i], s[i]);
}

torch::Tensor convert(torch::Tensor out, torch::Tensor q, torch::Tensor s) {
  int n = q.numel();
  convert_kernel<<<(n + 255) / 256, 256, 0,
                   at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<uint64_t>(), q.data_ptr<uint32_t>(), s.data_ptr<uint8_t>(), n);
  return out;
}

template <int K, int BM, int NF, int WAVES = 8>
__global__ __launch_bounds__(WAVES * 64, 1) void kernel(
    const uint8_t *__restrict__ x, const float *__restrict__ xscale,
    const int *__restrict__ xzero, const uint8_t *__restrict__ q,
    const uint8_t *__restrict__ scale, const int *__restrict__ wsum,
    const int *__restrict__ hist, const int *__restrict__ offsets,
    const int *__restrict__ gather, const int *__restrict__ schedule,
    __half *__restrict__ y, int n, float limit) {
  constexpr int TILE_K = 128;
  int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  int task = schedule[blockIdx.y];
  if (task < 0)
    return;
  int expert = task & 0xffff, row0 = task >> 16;
  if constexpr (BM == 16) row0 *= 16;
  int rows = min(BM, hist[expert] - row0), start = offsets[expert] + row0;
  int nlane = lane & 15, kpack = lane >> 4;
  int n0 = (blockIdx.x * WAVES + wave) * (16 * NF);
  i32x4 acc[BM / 16][NF] = {};
  constexpr int WORDS = BM * TILE_K / 8;
  constexpr int LOADS = (WORDS + WAVES * 64 - 1) / (WAVES * 64);
  __shared__ __align__(16) uint64_t asmem[2 * WORDS];
  __shared__ int token[BM];
  if (threadIdx.x < BM)
    token[threadIdx.x] =
        threadIdx.x < rows ? gather[start + threadIdx.x] / 6 : 0;
  __syncthreads();

  __amdgpu_buffer_rsrc_t qr[NF];
  int64_t sb[NF];
#pragma unroll
  for (int j = 0; j < NF; ++j) {
    int ntile = (blockIdx.x * WAVES + __builtin_amdgcn_readfirstlane(wave)) * NF + j;
    const uint8_t *qbase = q +
        (static_cast<int64_t>(expert) * (n / 16) + ntile) * (K / 128) * 64 * 16;
    qr[j] = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<uint8_t *>(qbase), 0, 0xffffffff, 0x00020000);
    sb[j] = ((static_cast<int64_t>(expert) * (n / 16) + ntile) *
                 (K / 128) * 16 + nlane) * 4;
  }

  uint64_t a[BM / 16];
  auto xr = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<uint8_t *>(x), 0, 0xffffffff, 0x00020000);
#pragma unroll
  for (int p = 0; p < BM / (WAVES * 2); ++p) {
    int row = p * WAVES * 2 + wave * 2 + (lane >> 5);
    int k4 = ((lane & 31) ^ ((row & 7) * 2)) * 4;
    int off = static_cast<int64_t>(token[row]) * K + k4;
    auto dst = reinterpret_cast<uint32_t *>(asmem) + row * (TILE_K / 4);
    direct_load_lds(dst, xr, off);
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __syncthreads();
  __builtin_amdgcn_sched_barrier(0);
  for (int kb = 0; kb < K; kb += TILE_K) {
    uint4 qv[NF];
    uint32_t sv[NF];
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      auto raw = __builtin_amdgcn_raw_buffer_load_b128(qr[j], lane * 16, kb * 8, 2);
      qv[j] = *reinterpret_cast<uint4 *>(&raw);
      sv[j] = *reinterpret_cast<const uint32_t *>(scale + sb[j] + kb / 2);
    }
    if (kb + TILE_K < K) {
      int base = ((kb / TILE_K & 1) ^ 1) * WORDS * 2;
#pragma unroll
      for (int p = 0; p < BM / (WAVES * 2); ++p) {
        int row = p * WAVES * 2 + wave * 2 + (lane >> 5);
        int k4 = ((lane & 31) ^ ((row & 7) * 2)) * 4;
        int off = static_cast<int64_t>(token[row]) * K + kb + TILE_K + k4;
        auto dst = reinterpret_cast<uint32_t *>(asmem) + base + row * (TILE_K / 4);
        direct_load_lds(dst, xr, off);
      }
    }
#pragma unroll
    for (int ks = 0; ks < TILE_K; ks += 32) {
      uint64_t b[NF];
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        uint32_t v = ks == 0 ? qv[j].x : ks == 32 ? qv[j].y
                                                 : ks == 64 ? qv[j].z : qv[j].w;
        b[j] = fp4x8_to_i8(v, reinterpret_cast<uint8_t *>(&sv[j])[ks / 32]);
      }
#pragma unroll
      for (int i = 0; i < BM / 16; ++i) {
        if (rows > i * 16) {
          int row = i * 16 + nlane;
          a[i] = asmem[(kb / TILE_K & 1) * WORDS +
                       (row * TILE_K +
                        ((ks + kpack * 8) ^ ((row & 7) * 8))) / 8];
        }
      }
#pragma unroll
      for (int i = 0; i < BM / 16; ++i)
        if (rows > i * 16)
#pragma unroll
          for (int j = 0; j < NF; ++j)
            acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
                static_cast<long>(a[i]), static_cast<long>(b[j]),
                acc[i][j], 0, 0, 0);
    }
    if (kb + TILE_K < K) {
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      __syncthreads();
      __builtin_amdgcn_sched_barrier(0);
    }
  }

#pragma unroll
  for (int i = 0; i < BM / 16; ++i)
#pragma unroll
    for (int j = 0; j < NF / 2; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        int row = i * 16 + (lane >> 4) * 4 + v;
        if (row < rows) {
          int t = token[row], z = xzero[t];
          int gate_col = n0 + j * 32 + nlane;
          int up_col = gate_col + 16;
          float s = xscale[t] * 0.00390625f;
          float gate = static_cast<float>(acc[i][j * 2][v] -
              z * wsum[static_cast<int64_t>(expert) * n + gate_col]) * s;
          float up = static_cast<float>(acc[i][j * 2 + 1][v] -
              z * wsum[static_cast<int64_t>(expert) * n + up_col]) * s;
          gate = fminf(gate, limit);
          up = fminf(fmaxf(up, -limit), limit);
          int col = (blockIdx.x * WAVES + wave) * (NF / 2 * 16) +
                    j * 16 + nlane;
          y[static_cast<int64_t>(start + row) * (n / 2) + col] =
              bf16_to_half(gate / (1.0f + expf(-gate)) * up);
        }
      }
}

torch::Tensor fusedi8_w1(torch::Tensor y, torch::Tensor x,
                         torch::Tensor xscale, torch::Tensor xzero,
                         torch::Tensor q, torch::Tensor scale,
                         torch::Tensor wsum, torch::Tensor hist,
                         torch::Tensor offsets, torch::Tensor gather,
                         torch::Tensor schedule, torch::Tensor schedule16,
                         bool tiny, double limit) {
  int n = q.size(1), k = q.size(2) * 2;
  TORCH_CHECK(k == 4096, "unsupported K");
  auto stream = at::cuda::getCurrentCUDAStream();
#define RUN(BM, NF, W, S) kernel<4096, BM, NF, W><<<dim3(n / (16 * NF * W), S.numel()), W * 64, 0, stream>>>(\
      x.data_ptr<uint8_t>(), xscale.data_ptr<float>(), xzero.data_ptr<int>(), \
      q.data_ptr<uint8_t>(), scale.data_ptr<uint8_t>(), wsum.data_ptr<int>(), \
      hist.data_ptr<int>(), offsets.data_ptr<int>(), gather.data_ptr<int>(), \
      S.data_ptr<int>(), reinterpret_cast<__half *>(y.data_ptr<at::Half>()), n, limit)
#ifndef CJ_NF_SMALL_M
#define CJ_NF_SMALL_M 8
#endif
#ifndef CJ_W1_SMALL_WAVES
#define CJ_W1_SMALL_WAVES 4
#endif
  if (tiny) {
    if (x.size(0) <= CJ_NF_SMALL_M) RUN(16, 2, CJ_W1_SMALL_WAVES, schedule16);
    else RUN(16, 4, 8, schedule16);
  } else RUN(64, 4, 8, schedule);
#undef RUN
  return y;
}
