// Rotate producers across three LDS slices so other waves can run WMMA.
#pragma once
#include "exl3_gemm_tiled.h"

namespace crucible {
constexpr int kPipeSlots = 3;
constexpr int kPipeWaitLimit = 1 << 20;

__device__ __forceinline__ bool GemmPipeWait(int* flag, int value) {
  for (int spin = 0; spin < kPipeWaitLimit; ++spin) {
    const int seen = __hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    if (__builtin_amdgcn_readfirstlane((uint32_t)seen) == (uint32_t)value) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
      return true;
    }
    __builtin_amdgcn_s_sleep(1);
  }
  return false;
}

template <int K, int CHUNK>
__device__ __forceinline__ bool GemmPipeProduce(__amdgpu_buffer_rsrc_t wr,
    const LaneWindow<K>& win, int begin, int end, int seq, uint32_t* weights,
    int* ready, int* done) {
  const int lane = threadIdx.x & 31, slot = seq % kPipeSlots;
#pragma unroll
  for (int w = 0; w < 4; ++w)
    if (!GemmPipeWait(done + slot * 4 + w, seq - kPipeSlots)) return false;
  constexpr int T = 8 * K;
  const int a0 = begin + seq * CHUNK;
#pragma unroll
  for (int f = 0; f < CHUNK; ++f) {
    if (a0 + f >= end) break;
    const int so = (a0 + f - begin) * T * 4;
    const uint32_t w0 = __builtin_amdgcn_raw_buffer_load_b32(wr, win.d0 * 4 + so, 0, 0);
    const uint32_t w1 = __builtin_amdgcn_raw_buffer_load_b32(wr, ((win.d0 + 1) % T) * 4 + so, 0, 0);
    uint32_t w2 = 0;
    if constexpr (K == 5)
      w2 = __builtin_amdgcn_raw_buffer_load_b32(wr, ((win.d0 + 2) % T) * 4 + so, 0, 0);
    __half2 p[4];
    DecodeLane<K, kMulTrue16>(w0, w1, w2, win.shift, p);
    GemmTileWrite(p, weights + (slot * CHUNK + f) * kGemmWaveWords, lane);
  }
  // Lane zero must publish all lanes' stores, not only its own stores.
  GemmWaveSync();
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  if (lane == 0)
    __hip_atomic_store(ready + slot, seq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  return true;
}

template <int K, int RM, int CHUNK, bool FULL_M>
__device__ __forceinline__ void GemmPipeBody(const Exl3GemmJob& j,
    const Exl3GemmItem& it, uint32_t* weights, int* ready, int* done) {
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int b = it.b0, row0 = it.m0 + wave * RM * 16;
  const int A = j.I / 16, begin = A * it.s / j.S, end = A * (it.s + 1) / j.S;
  const int chunks = (end - begin + CHUNK - 1) / CHUNK;
  constexpr int T = 8 * K;
  const LaneWindow<K> win(lane);
  const auto wr = StreamRsrc(j.tiles + ((size_t)b * A + begin) * T, (end - begin) * T * 4);
  const auto xr = StreamRsrc(reinterpret_cast<const uint32_t*>(j.x), (uint32_t)j.Xrows * j.I * 2);
  uint32_t xo[RM];
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    const int row = row0 + r * 16 + (lane & 15);
    const int source = (FULL_M || row < j.M) ? (j.row_ids ? j.row_ids[row] : row) : j.Xrows;
    xo[r] = (uint32_t)source * j.I * 2;
  }
  GemmF8 acc[RM] = {};
  auto fail = [&]() {
    // A failed wait must produce a visible error rather than a plausible sum.
    if (row0 < j.M)
      j.partial[((size_t)it.s * j.M + row0) * j.O + b * 16 + (lane & 15)] = __builtin_nanf("");
  };
  if (threadIdx.x < kPipeSlots) ready[threadIdx.x] = -1;
  if (threadIdx.x < kPipeSlots * 4) done[threadIdx.x] = threadIdx.x / 4 - kPipeSlots;
  GemmGroupSync();
  if (wave == 0 && !GemmPipeProduce<K, CHUNK>(wr, win, begin, end, 0, weights, ready, done)) {
    fail(); return;
  }
  for (int seq = 0; seq < chunks; ++seq) {
    const int future = seq + 1, slot = seq % kPipeSlots;
    if (future < chunks && wave == (future & 3)) {
      if (!GemmPipeProduce<K, CHUNK>(wr, win, begin, end, future, weights, ready, done)) {
        fail(); return;
      }
    }
    if (!GemmPipeWait(ready + slot, seq)) { fail(); return; }
    const int a0 = begin + seq * CHUNK;
    if (FULL_M || row0 < j.M) {
#pragma unroll
      for (int f = 0; f < CHUNK; ++f) {
        if (a0 + f >= end) break;
        const GemmH16 bf = GemmTileRead(weights + (slot * CHUNK + f) * kGemmWaveWords, lane);
#pragma unroll
        for (int r = 0; r < RM; ++r) {
          if constexpr (!FULL_M) { if (row0 + r * 16 >= j.M) break; }
          const GemmH16 af = GemmAFragment<false>(xr, xo[r], a0 + f, nullptr, 0);
          acc[r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc[r]);
        }
      }
    }
    // Producers must not overwrite a slice while any lane still reads it.
    GemmWaveSync();
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    if (lane == 0)
      __hip_atomic_store(done + slot * 4 + wave, seq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  }
#pragma unroll
  for (int r = 0; r < RM; ++r)
#pragma unroll
    for (int z = 0; z < 8; ++z) {
      const int row = row0 + r * 16 + 2 * z + lane / 16;
      if (FULL_M || row < j.M)
        j.partial[((size_t)it.s * j.M + row) * j.O + b * 16 + (lane & 15)] = acc[r][z];
    }
}

template <int RM = 2, int CHUNK = 4, bool FULL_M = false>
__global__ void __launch_bounds__(128) Exl3GemmPipeKernel(const Exl3GemmJob* jobs,
    const Exl3GemmItem* items, int n) {
  static_assert(RM == 1 || RM == 2 || RM == 4 || RM == 8 || RM == 16);
  static_assert(CHUNK == 4 || CHUNK == 8 || CHUNK == 16);
  __shared__ uint32_t weights[kPipeSlots * CHUNK * kGemmWaveWords];
  __shared__ int ready[kPipeSlots], done[kPipeSlots * 4];
  if (blockIdx.x >= n) return;
  const Exl3GemmItem it = items[blockIdx.x];
  const Exl3GemmJob j = jobs[it.job];
  if (j.K == 3) GemmPipeBody<3, RM, CHUNK, FULL_M>(j, it, weights, ready, done);
  else if (j.K == 4) GemmPipeBody<4, RM, CHUNK, FULL_M>(j, it, weights, ready, done);
  else if (j.K == 5) GemmPipeBody<5, RM, CHUNK, FULL_M>(j, it, weights, ready, done);
}
}  // namespace crucible
