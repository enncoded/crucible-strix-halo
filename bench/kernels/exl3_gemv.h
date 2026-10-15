// EXL3 matrix-vector kernels that only benches launch: the 1-8 row GEMV and a
// debug tile decoder. The device helpers stay in engine/exl3/kernels/exl3_gemv.h.
#pragma once
#include "../../engine/exl3/kernels/exl3_gemv.h"

namespace crucible {

struct Exl3Job {
  const uint32_t* tiles;  // crucible order, 8K dwords per tile
  const __half* x;        // [R][I] activations, already input-transformed
  float* partial;         // [S][R][O] fp32 partial sums (pre output transform)
  int A, B, S;            // k tiles (I/16), n tiles (O/16), k splits
};

// One workgroup = (job, k split, up to 8 column blocks); one wave per block.
struct Exl3Item {
  int job, s, b0, nb;
};

// x staged in LDS as dwords [R][ktiles][8]; dword q of a tile holds k = 2q, 2q+1.
// Dynamic size R * span * 32 bytes, so short spans don't cap occupancy.
template <int K, int R, int MODE, int PF>
__global__ void __launch_bounds__(256) Exl3GemvKernel(const Exl3Job* __restrict__ jobs,
                                                      const Exl3Item* __restrict__ items,
                                                      int n_items) {
  extern __shared__ uint32_t xs[];
  const int lane = threadIdx.x & 31;
  // Wave-uniform in an SGPR, so tile addresses are scalar base + lane offset
  // (global_load with saddr) rather than per-lane 64-bit VALU math.
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const LaneWindow<K> win(lane);
  const int q = lane & 3;  // r0 / 2
  constexpr int T = 8 * K;  // dwords per tile

  for (int it = blockIdx.x; it < n_items; it += gridDim.x) {
    const Exl3Item item = items[it];
    const Exl3Job job = jobs[item.job];
    const int span = job.A / job.S;
    const int a0 = item.s * span;
    // Stage this k range of every activation row.
    const uint32_t* xg = reinterpret_cast<const uint32_t*>(job.x);
    const int row_dw = job.A * 8;
    for (int i = threadIdx.x; i < R * span * 8; i += blockDim.x) {
      int r = i / (span * 8), o = i % (span * 8);
      xs[i] = xg[r * row_dw + a0 * 8 + o];
    }
    __syncthreads();
    if (wave < item.nb) {
      const int b = item.b0 + wave;
      const __amdgpu_buffer_rsrc_t rs =
          StreamRsrc(job.tiles + ((size_t)b * job.A + a0) * T, (uint32_t)span * T * 4);
      const int o0 = 4 * win.d0, o1 = 4 * ((win.d0 + 1) % T), o2 = 4 * ((win.d0 + 2) % T);
      auto load_tile = [&](int tile, uint32_t* dst) {
        const int so = tile * T * 4;
        dst[0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0, so, 0);
        dst[1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1, so, 0);
        dst[2] = __builtin_amdgcn_raw_buffer_load_b32(rs, o2, so, 0);
      };
      float acc[R][2];
#pragma unroll
      for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.f;
      // Software pipeline: PF tiles of raw dwords in flight per lane.
      uint32_t ring[PF][3];
#pragma unroll
      for (int f = 0; f < PF; ++f) load_tile(f, ring[f]);
      for (int a = 0; a < span; a += PF) {
#pragma unroll
        for (int f = 0; f < PF; ++f) {
          if (a + f >= span) break;
          uint32_t w0 = ring[f][0], w1 = ring[f][1], w2 = ring[f][2];
          load_tile(a + f + PF, ring[f]);
          __half2 P[4];
          DecodeLane<K, MODE>(w0, w1, w2, win.shift, P);
#pragma unroll
          for (int r = 0; r < R; ++r) {
            const uint32_t* xr = xs + (r * span + a + f) * 8;
            uint32_t xa = xr[q], xb = xr[4 + q];
            __half2 ha = *reinterpret_cast<__half2*>(&xa);
            __half2 hb = *reinterpret_cast<__half2*>(&xb);
            acc[r][0] = Dot2(P[0], ha, acc[r][0]);
            acc[r][0] = Dot2(P[1], hb, acc[r][0]);
            acc[r][1] = Dot2(P[2], ha, acc[r][1]);
            acc[r][1] = Dot2(P[3], hb, acc[r][1]);
          }
        }
      }
      // Lanes 4c..4c+3 share columns; fixed-order butterfly keeps every lane's
      // sum bit-identical.
#pragma unroll
      for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          float v = acc[r][c];
          v = v + __shfl_xor(v, 1);
          v = v + __shfl_xor(v, 2);
          acc[r][c] = v;
        }
      }
      if (q == 0) {
        const int col = 16 * b + lane / 4;
        float* out = job.partial + (size_t)item.s * R * (job.B * 16);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          out[r * job.B * 16 + col] = acc[r][0];
          out[r * job.B * 16 + col + 8] = acc[r][1];
        }
      }
    }
    __syncthreads();
  }
}

// Debug: decode every tile of a crucible-order matrix to fp16 [I][O].
template <int K, int MODE>
__global__ void Exl3DecodeKernel(const uint32_t* tiles, int A, int B, __half* w) {
  const int lane = threadIdx.x & 31;
  const int tile = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (tile >= A * B) return;
  const int b = tile / A, a = tile % A;
  const LaneWindow<K> win(lane);
  constexpr int T = 8 * K;
  const uint32_t* t = tiles + (size_t)tile * T;
  __half2 P[4];
  DecodeLane<K, MODE>(t[win.d0], t[(win.d0 + 1) % T], t[(win.d0 + 2) % T], win.shift, P);
  const int r0 = 2 * (lane & 3), c0 = lane / 4;
  const int O = 16 * B;
  auto put = [&](int r, int c, __half v) { w[(size_t)(16 * a + r) * O + 16 * b + c] = v; };
  put(r0, c0, __low2half(P[0]));
  put(r0 + 1, c0, __high2half(P[0]));
  put(r0 + 8, c0, __low2half(P[1]));
  put(r0 + 9, c0, __high2half(P[1]));
  put(r0, c0 + 8, __low2half(P[2]));
  put(r0 + 1, c0 + 8, __high2half(P[2]));
  put(r0 + 8, c0 + 8, __low2half(P[3]));
  put(r0 + 9, c0 + 8, __high2half(P[3]));
}

}  // namespace crucible
