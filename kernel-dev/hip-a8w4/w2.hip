#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>

using f32x4 = float __attribute__((ext_vector_type(4)));
using i16x4 = short __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void direct_load_lds(
    void *dst, __amdgpu_buffer_rsrc_t src, int offset) {
  auto m0 = __builtin_amdgcn_readfirstlane(reinterpret_cast<uintptr_t>(dst));
  asm volatile("s_mov_b32 m0, %0; \n\tbuffer_load_dword %1, %2, 0 offen lds"
               :: "s"(m0), "v"(offset), "s"(src) : "memory");
}

__device__ __forceinline__ uint16_t float_to_bf16(float x) {
  uint32_t bits = __float_as_uint(x);
  bits += 0x7fffu + ((bits >> 16) & 1);
  return bits >> 16;
}

template <bool ODD>
__device__ __forceinline__ uint64_t fp4x4_to_f16(uint32_t q, uint8_t s) {
  uint32_t d = static_cast<uint32_t>(s) * 0x01010101u;
  uint32_t lo = (d + 0x3d3b3800u) & 0xffffff00u;
  uint32_t hi = d + 0x45434140u;
  uint32_t idx = ODD ? (q & 0x70707070u) >> 4 : q & 0x07070707u;
  uint32_t h = __builtin_amdgcn_perm(hi, lo, idx);
  h |= ODD ? q & 0x80808080u : (q & 0x08080808u) << 4;
  uint32_t a = __builtin_amdgcn_perm(h, 0, 0x05010400u);
  uint32_t b = __builtin_amdgcn_perm(h, 0, 0x07030602u);
  return static_cast<uint64_t>(a) | (static_cast<uint64_t>(b) << 32);
}

__global__ void convert_kernel(uint64_t *out, const uint32_t *q,
                               const uint8_t *s, bool odd) {
  out[threadIdx.x] = odd ? fp4x4_to_f16<true>(q[threadIdx.x], s[threadIdx.x])
                         : fp4x4_to_f16<false>(q[threadIdx.x], s[threadIdx.x]);
}

torch::Tensor convert(torch::Tensor out, torch::Tensor q, torch::Tensor s,
                      bool odd) {
  convert_kernel<<<1, q.numel(), 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<uint64_t>(), q.data_ptr<uint32_t>(), s.data_ptr<uint8_t>(), odd);
  return out;
}

template <int K, int BM, int NF, int WAVES = 8>
__global__ __launch_bounds__(WAVES * 64, 1) void kernel(
    const uint16_t *__restrict__ x, const uint8_t *__restrict__ q,
    const uint8_t *__restrict__ scale,
    const int *__restrict__ hist, const int *__restrict__ offsets,
    const int *__restrict__ schedule, const int *__restrict__ scatter,
    const uint16_t *__restrict__ gate,
    __half *__restrict__ y, int n) {
  constexpr int TK = 128;
  constexpr int WORDS = BM * TK / 4;
  constexpr int LOADS = (WORDS + WAVES * 64 - 1) / (WAVES * 64);
  int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  int task = schedule[blockIdx.y];
  if (task < 0)
    return;
  int expert = task & 0xffff, row0 = task >> 16;
  if constexpr (BM == 16) row0 *= 16;
  int rows = min(BM, hist[expert] - row0), start = offsets[expert] + row0;
  int nlane = lane & 15, kpack = lane >> 4;
  int n0 = (blockIdx.x * WAVES + wave) * (16 * NF);
  f32x4 acc[BM / 16][NF] = {};
  __shared__ __align__(16) uint64_t asmem[2 * WORDS];
  __shared__ int token[BM];
  if (threadIdx.x < BM)
    token[threadIdx.x] = threadIdx.x < rows ? start + threadIdx.x : 0;
  __syncthreads();

  __amdgpu_buffer_rsrc_t qr[NF];
  int64_t sb[NF];
#pragma unroll
  for (int j = 0; j < NF; ++j) {
    int ntile = (n0 + j * 16 + nlane) / 16;
    const uint8_t *qbase = q +
        (static_cast<int64_t>(expert) * (n / 16) + ntile) *
            (K / TK) * 64 * 16;
    qr[j] = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<uint8_t *>(qbase), 0, 0xffffffff, 0x00020000);
    sb[j] = ((static_cast<int64_t>(expert) * (n / 16) + ntile) *
                 (K / TK) * 16 + nlane) * 4;
  }

  auto xr = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<uint16_t *>(x), 0, 0xffffffff, 0x00020000);
#pragma unroll
  for (int p = 0; p < BM / WAVES; ++p) {
    int row = p * WAVES + wave;
    int k2 = (lane ^ ((row & 7) * 2)) * 2;
    int off = (static_cast<int64_t>(token[row]) * K + k2) * 2;
    auto dst = reinterpret_cast<uint32_t *>(asmem) + row * (TK / 2);
    direct_load_lds(dst, xr, off);
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  __builtin_amdgcn_sched_barrier(0);
  for (int kb = 0; kb < K; kb += TK) {
    uint4 qv[NF];
    uint32_t sv[NF];
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      auto raw =
          __builtin_amdgcn_raw_buffer_load_b128(qr[j], lane * 16, kb * 8, 2);
      qv[j] = *reinterpret_cast<uint4 *>(&raw);
      sv[j] = *reinterpret_cast<const uint32_t *>(scale + sb[j] + kb / 2);
    }
    if (kb + TK < K) {
      int base = ((kb / TK & 1) ^ 1) * WORDS * 2;
#pragma unroll
      for (int p = 0; p < BM / WAVES; ++p) {
        int row = p * WAVES + wave;
        int k2 = (lane ^ ((row & 7) * 2)) * 2;
        int off = (static_cast<int64_t>(token[row]) * K + kb + TK + k2) * 2;
        auto dst = reinterpret_cast<uint32_t *>(asmem) + base + row * (TK / 2);
        direct_load_lds(dst, xr, off);
      }
    }
#pragma unroll
    for (int ks = 0; ks < TK; ks += 16) {
      uint64_t a[BM / 16], b[NF];
#pragma unroll
      for (int i = 0; i < BM / 16; ++i) {
        int row = i * 16 + nlane;
        a[i] = asmem[(kb / TK & 1) * WORDS +
                     (row * TK + ((ks + kpack * 4) ^ ((row & 7) * 4))) / 4];
      }
#pragma unroll
      for (int j = 0; j < NF; ++j)
        b[j] = (ks & 16)
                   ? fp4x4_to_f16<true>(reinterpret_cast<uint32_t *>(&qv[j])[ks / 32],
                                        reinterpret_cast<uint8_t *>(&sv[j])[ks / 32])
                   : fp4x4_to_f16<false>(reinterpret_cast<uint32_t *>(&qv[j])[ks / 32],
                                         reinterpret_cast<uint8_t *>(&sv[j])[ks / 32]);
#pragma unroll
      for (int i = 0; i < BM / 16; ++i)
        if (i == 0 || rows > i * 16)
#pragma unroll
          for (int j = 0; j < NF; ++j)
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16f16(
                *reinterpret_cast<i16x4 *>(&a[i]),
                *reinterpret_cast<i16x4 *>(&b[j]), acc[i][j], 0, 0, 0);
    }
    if (kb + TK < K) {
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      __builtin_amdgcn_sched_barrier(0);
    }
  }
#pragma unroll
  for (int i = 0; i < BM / 16; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        int row = i * 16 + (lane >> 4) * 4 + v;
        if (row < rows) {
          int routed = start + row;
          y[static_cast<int64_t>(scatter[routed]) * n + n0 + j * 16 + nlane] =
              __float2half_rn(
                  acc[i][j][v] *
                  __uint_as_float(static_cast<uint32_t>(gate[routed]) << 16));
        }
      }
}

__global__ void reduce(const __half *__restrict__ routed,
                       uint16_t *__restrict__ out, int groups) {
  constexpr int n = 4096;
  int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= groups)
    return;
  int idx = group * 8, row = idx / n, col = idx - row * n;
  float sum[8] = {};
#pragma unroll
  for (int slot = 0; slot < 6; ++slot) {
    auto h = *reinterpret_cast<const uint4 *>(
        routed + (static_cast<int64_t>(row) * 6 + slot) * n + col);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      auto f = __half22float2(reinterpret_cast<__half2 *>(&h)[i]);
      sum[i * 2] += f.x; sum[i * 2 + 1] += f.y;
    }
  }
  uint4 packed;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    reinterpret_cast<uint32_t *>(&packed)[i] = float_to_bf16(sum[i * 2]) |
        (static_cast<uint32_t>(float_to_bf16(sum[i * 2 + 1])) << 16);
  reinterpret_cast<uint4 *>(out)[group] = packed;
}

torch::Tensor w2(torch::Tensor out, torch::Tensor routed, torch::Tensor x,
                 torch::Tensor q, torch::Tensor hist, torch::Tensor offsets,
                 torch::Tensor gather, torch::Tensor scatter,
                 torch::Tensor gate, torch::Tensor schedule,
                 torch::Tensor schedule16, torch::Tensor scale, bool tiny) {
  int n = q.size(1), k = q.size(2) * 2;
  auto stream = at::cuda::getCurrentCUDAStream();
#define RUN(K, BM, NF, W, S) kernel<K, BM, NF, W><<<dim3(n / (16 * NF * W), S.numel()), W * 64, 0, stream>>>(\
    reinterpret_cast<const uint16_t *>(x.data_ptr()), q.data_ptr<uint8_t>(),  \
    scale.data_ptr<uint8_t>(), hist.data_ptr<int>(), offsets.data_ptr<int>(), \
    S.data_ptr<int>(), scatter.data_ptr<int>(),                               \
    reinterpret_cast<const uint16_t *>(gate.data_ptr<at::BFloat16>()),        \
    reinterpret_cast<__half *>(routed.data_ptr<at::Half>()), n)
#ifndef CJ_NF_SMALL_M
#define CJ_NF_SMALL_M 8
#endif
#ifndef CJ_W2_SMALL_WAVES
#define CJ_W2_SMALL_WAVES 8
#endif
#define DISPATCH(K) if (tiny) { \
    if (x.size(0) <= CJ_NF_SMALL_M * 6) RUN(K, 16, 1, CJ_W2_SMALL_WAVES, schedule16); \
    else RUN(K, 16, 2, 8, schedule16); \
  } else RUN(K, 48, 2, 8, schedule)
  if (k == 4096) { DISPATCH(4096); } else { DISPATCH(2048); }
#undef DISPATCH
#undef RUN
  int groups = out.numel() / 8;
  reduce<<<(groups + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __half *>(routed.data_ptr<at::Half>()),
      reinterpret_cast<uint16_t *>(out.data_ptr<at::BFloat16>()), groups);
  return out;
}
