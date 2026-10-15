// Chunk-64 WY/UT Gated DeltaNet. Decode and speculative replay are untouched.
#pragma once
#include "../ops.h"
#include "../numerics.h"
#include "../../gfx1151/prefill_wmma.h"

namespace crucible::prefill {
constexpr int kGdnChunk = 64;

struct GdnPrefill {
  GdnArgs input;             // replay=0, commit=R only; pend unused
  const float* ab_logits;    // [R][96], a then b, produced by batched dense GEMM
  float *q, *k, *v, *w, *ut, *u, *y, *wk;  // [48][chunks][64][128]
  float *g, *beta;           // [48][chunks][64], cumulative log decay after prefix
  float *lower, *qk;         // [48][chunks][64][64]
  int chunks;
};

__device__ __forceinline__ size_t GdnBase(GdnPrefill a, int h, int c) {
  return ((size_t)h * a.chunks + c) * kGdnChunk;
}

__global__ void __launch_bounds__(256) GdnPrefillPrepKernel(GdnPrefill a) {
  __shared__ float red[8];
  const int h = blockIdx.x, r = blockIdx.y, j = threadIdx.x & 127, half = threadIdx.x >> 7;
  const size_t t = GdnBase(a, h, r / 64) + r % 64;
  if (r >= a.input.R) {
    if (half) a.k[t * 128 + j] = 0.f;
    else { a.q[t * 128 + j] = a.v[t * 128 + j] = 0.f; }
    if (threadIdx.x == 0) a.g[t] = a.beta[t] = 0.f;
    return;
  }
  const int kh = h / 3, cq = kh * 128 + j, ck = 2048 + kh * 128 + j, cv = 4096 + h * 128 + j;
  auto conv = [&](int ch) {
    float x = 0.f;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
      const int rr = r - 3 + tap, p = a.input.pos0 + rr;
      const float raw = rr >= 0 ? a.input.qkv[(size_t)rr * kGdnQkvWidth + ch]
          : p >= 0 ? a.input.hist[(size_t)(p & 15) * kGdnQkvWidth + ch] : 0.f;
      x += a.input.conv[ch * 4 + tap] * raw;
    }
    return Silu(x);
  };
  const float qk = conv(half ? ck : cq);
  const float sq = BlockSum<256>(half ? 0.f : qk * qk, red);
  const float sk = BlockSum<256>(half ? qk * qk : 0.f, red);
  if (half) a.k[t * 128 + j] = qk * rsqrtf(sk + kEps);
  else {
    a.q[t * 128 + j] = qk * rsqrtf(sq + kEps) * kHadNorm;
    a.v[t * 128 + j] = conv(cv);
  }
  if (threadIdx.x == 0) {
    const float x = a.ab_logits[(size_t)r * 96 + h] + a.input.dt_bias[h];
    const float softplus = x > 20.f ? x : log1pf(expf(x));
    a.g[t] = -__expf(a.input.A_log[h]) * softplus;
    a.beta[t] = Sigmoid(a.ab_logits[(size_t)r * 96 + 48 + h]);
  }
}

__global__ void GdnPrefillPrefixKernel(GdnPrefill a) {
  __shared__ float decay[64];
  const int h = blockIdx.x, c = blockIdx.y;
  const size_t base = GdnBase(a, h, c);
  if (threadIdx.x == 0) {
    float sum = 0.f;
    for (int t = 0; t < 64; ++t) {
      if (c * 64 + t < a.input.R) sum += a.g[base + t];
      a.g[base + t] = sum;
    }
  }
  __syncthreads();
  if (threadIdx.x<64) decay[threadIdx.x]=expf(a.g[base+threadIdx.x]);
  __syncthreads();
  for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
    const int row = i / 128;
    const float beta = a.beta[base + row];
    a.wk[base * 128 + i] = beta * decay[row] * a.k[base * 128 + i];
    a.v[base * 128 + i] *= beta;
  }
}

// The log-decay difference is nonpositive only for the causal lower triangle.
// Never form exp(-g), which overflows on strongly forgetful heads.
__global__ void __launch_bounds__(128) GdnPrefillGramKernel(GdnPrefill a) {
  const int m0 = blockIdx.x * 16, n0 = (threadIdx.x >> 5) * 16;
  const int c = blockIdx.y, h = blockIdx.z, lane = threadIdx.x & 31;
  const size_t b = GdnBase(a, h, c);
  const float* keys = a.k + b * 128;
  const float* queries = a.q + b * 128;
  const F8 kk = Product(128, [&](int m, int k) { return keys[(m0 + m) * 128 + k]; },
      [&](int k, int n) { return keys[(n0 + n) * 128 + k]; });
  const F8 qk = Product(128, [&](int m, int k) { return queries[(m0 + m) * 128 + k]; },
      [&](int k, int n) { return keys[(n0 + n) * 128 + k]; });
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const int t = m0 + FragmentRow(z), s = n0 + (lane & 15);
    const float decay = s <= t ? expf(a.g[b + t] - a.g[b + s]) : 0.f;
    const size_t dst = b * 64 + t * 64 + s;
    a.lower[dst] = s < t ? a.beta[b + t] * kk[z] * decay : 0.f;
    a.qk[dst] = s <= t ? qk[z] * decay : 0.f;
  }
}

// Build T=(I+L)^-1 with only 64 triangular right hand sides, rather than
// serially solving all 256 key/value features. The large WY/UT application
// is then a parallel WMMA product.
__global__ void __launch_bounds__(64) GdnPrefillInverseKernel(GdnPrefill a) {
  __shared__ float lower[64 * 64], inverse[64 * 64];
  const int h = blockIdx.x, c = blockIdx.y, j = threadIdx.x;
  const size_t b = GdnBase(a, h, c);
  for (int i = j; i < 64 * 64; i += 64) lower[i] = a.lower[b * 64 + i];
  __syncthreads();
  for (int r = 0; r < 64; ++r) {
    float value = r == j ? 1.f : 0.f;
    if (j < r) for (int s = j; s < r; ++s) value -= lower[r * 64 + s] * inverse[s * 64 + j];
    // Each thread reads only its own inverse column. No row barrier is needed.
    inverse[r * 64 + j] = value;
  }
  for (int i = j; i < 64 * 64; i += 64) a.lower[b * 64 + i] = inverse[i];
}

__global__ void __launch_bounds__(128) GdnPrefillWyKernel(GdnPrefill a) {
  const int hc = blockIdx.z / 2, kind = blockIdx.z & 1;
  const int h = hc / a.chunks, c = hc % a.chunks;
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 64 + (threadIdx.x >> 5) * 16;
  const int lane = threadIdx.x & 31;
  const size_t b = GdnBase(a, h, c);
  const float* src = kind == 0 ? a.wk : a.v;
  const F8 out = Product(64,
    [&](int m, int k) { return a.lower[b * 64 + (m0 + m) * 64 + k]; },
    [&](int k, int n) { return src[(b + k) * 128 + n0 + n]; });
  float* dst = kind == 0 ? a.w : a.ut;
#pragma unroll
  for (int z = 0; z < 8; ++z)
    dst[(b + m0 + FragmentRow(z)) * 128 + n0 + (lane & 15)] = out[z];
}

__global__ void __launch_bounds__(128) GdnPrefillEndKeysKernel(GdnPrefill a) {
  __shared__ float end_decay[64];
  const int h = blockIdx.x, c = blockIdx.y, j = threadIdx.x;
  const size_t b = GdnBase(a, h, c);
  const int end = min(64, a.input.R - c * 64) - 1;
  if (j<64) {
    end_decay[j]=j<=end ? expf(a.g[b+end]-a.g[b+j]) : 0.f;
    // Gram and WY no longer need beta. Reuse it for the carry decay.
    a.beta[b+j]=expf(a.g[b+j]);
  }
  __syncthreads();
  for (int t = 0; t < 64; ++t)
    a.v[(b + t) * 128 + j] = a.k[(b + t) * 128 + j] * end_decay[t];
}

__global__ void __launch_bounds__(128) GdnPrefillCarryKernel(GdnPrefill a, int c) {
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 64 + (threadIdx.x >> 5) * 16;
  const int h = blockIdx.z, lane = threadIdx.x & 31;
  const size_t b = GdnBase(a, h, c);
  const float* state = a.input.state + (size_t)h * 128 * 128;
  auto ls = [&](int k, int n) { return state[k * 128 + n0 + n]; };
  const F8 ws = Product(128, [&](int m, int k) { return a.w[(b + m0 + m) * 128 + k]; }, ls);
  const F8 qs = Product(128, [&](int m, int k) { return a.q[(b + m0 + m) * 128 + k]; }, ls);
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const int m = m0 + FragmentRow(z), n = n0 + (lane & 15);
    const size_t idx = (b + m) * 128 + n;
    a.u[idx] = a.ut[idx] - ws[z];
    a.y[idx] = a.beta[b + m] * qs[z];
  }
}

__global__ void __launch_bounds__(128) GdnPrefillOutputKernel(GdnPrefill a, int c) {
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 64 + (threadIdx.x >> 5) * 16;
  const int h = blockIdx.z, lane = threadIdx.x & 31;
  const size_t b = GdnBase(a, h, c);
  const F8 local = Product(64,
      [&](int m, int k) { return a.qk[b * 64 + (m0 + m) * 64 + k]; },
      [&](int k, int n) { return a.u[(b + k) * 128 + n0 + n]; });
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const size_t idx = (b + m0 + FragmentRow(z)) * 128 + n0 + (lane & 15);
    a.y[idx] += local[z];
  }
}

__global__ void __launch_bounds__(128) GdnPrefillStateKernel(GdnPrefill a, int c) {
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 64 + (threadIdx.x >> 5) * 16;
  const int h = blockIdx.z, lane = threadIdx.x & 31;
  const size_t b = GdnBase(a, h, c);
  const int end = min(64, a.input.R - c * 64) - 1;
  const float final_decay = a.beta[b + end];
  const F8 update = Product(64,
      [&](int m, int k) { return a.v[(b + k) * 128 + m0 + m]; },
      [&](int k, int n) { return a.u[(b + k) * 128 + n0 + n]; });
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const size_t idx = (size_t)h * 128 * 128 + (m0 + FragmentRow(z)) * 128 + n0 + (lane & 15);
    a.input.state[idx] = final_decay * a.input.state[idx] + update[z];
  }
}

__global__ void __launch_bounds__(256) GdnPrefillGateKernel(GdnPrefill a) {
  __shared__ float red[8], output[128];
  const int h = blockIdx.x, r = blockIdx.y, t = threadIdx.x, j = t & 127;
  const size_t b = GdnBase(a, h, r / 64) + r % 64;
  const float y = a.y[b * 128 + j];
  const float yy = BlockSum<256>(t < 128 ? y * y : 0.f, red);
  if (t < 128) {
    const float o = y * rsqrtf(yy / 128.f + kEps) * a.input.gamma[j]
        * Sigmoid(a.input.z[(size_t)r * kGdnValueWidth + h * 128 + j]);
    output[j] = RoundH(RoundH(o) * __half2float(a.input.suh_out[h * 128 + j]));
  }
  __syncthreads();
  if (t < 32) {
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = output[t * 4 + i];
    Wht128(v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      a.input.xh_out[(size_t)r * kGdnValueWidth + h * 128 + t * 4 + i] = __float2half(v[i] * kHadNorm);
  }
}

__global__ void GdnPrefillHistoryKernel(GdnPrefill a) {
  const int ch = blockIdx.x * 256 + threadIdx.x;
  if (ch >= kGdnQkvWidth) return;
  for (int r = max(0, a.input.R - 16); r < a.input.R; ++r)
    a.input.hist[(size_t)((a.input.pos0 + r) & 15) * kGdnQkvWidth + ch] = a.input.qkv[(size_t)r * kGdnQkvWidth + ch];
}

// Full-row workspaces let the exact recurrence keep state in registers.
__global__ void __launch_bounds__(128) GdnPersistentColumnKernel(GdnPrefill a) {
  __shared__ float keys[2][128],queries[2][128];
  const int h=blockIdx.x,j=threadIdx.x;
  float lo[64],hi[64];
  float* dst=a.input.state+(size_t)h*128*128+j;
#pragma unroll
  for (int i=0;i<64;++i) { lo[i]=dst[i*128];hi[i]=dst[(64+i)*128]; }
  for (int r=0;r<a.input.R;++r) {
    const size_t b=(size_t)h*a.chunks*64+r;
    const int slot=r&1;
    keys[slot][j]=a.k[b*128+j];queries[slot][j]=a.q[b*128+j];
    __syncthreads();
    const float decay=expf(a.g[b]);
    float m0=0,m1=0;
#pragma unroll
    for (int i=0;i<64;++i) {
      lo[i]*=decay;hi[i]*=decay;
      m0+=lo[i]*keys[slot][i];m1+=hi[i]*keys[slot][64+i];
    }
    const float delta=a.beta[b]*(a.v[b*128+j]-(m0+m1));
    float y0=0,y1=0;
#pragma unroll
    for (int i=0;i<64;++i) {
      lo[i]+=keys[slot][i]*delta;hi[i]+=keys[slot][64+i]*delta;
      y0+=lo[i]*queries[slot][i];y1+=hi[i]*queries[slot][64+i];
    }
    a.y[b*128+j]=y0+y1;
  }
#pragma unroll
  for (int i=0;i<64;++i) { dst[i*128]=lo[i];dst[(64+i)*128]=hi[i]; }
}

inline void ValidateGdnPrefill(GdnPrefill a) {
  if (a.input.R < 1 || a.input.R > 2048 || a.input.pos0 < 0 || a.input.replay != 0
      || a.input.commit != a.input.R || a.chunks != (a.input.R + 63) / 64) {
    std::fprintf(stderr, "GDN prefill requires 1..2048 committed rows and no replay\n");
    std::abort();
  }
}

// Lanes path layout: q and k are stored once per key head, [16][rows][128];
// v, y, g and beta per value head as before. Grid x: 16 q/k blocks, then 24
// blocks of 256 value channels; the first value block also writes g and beta.
// V=false (the vscan layout, bench only): grid x is 16+1, v is not written (the
// scan computes it), and block 16 writes only g and beta.
template <bool V = true>
__global__ void __launch_bounds__(256) GdnLanesPrepKernel(GdnPrefill a) {
  __shared__ float red[8];
  const int r = blockIdx.y, t = threadIdx.x;
  const size_t rows = (size_t)a.chunks * 64;
  auto conv = [&](int ch) {
    float x = 0.f;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
      const int rr = r - 3 + tap, p = a.input.pos0 + rr;
      const float raw = rr >= 0 ? a.input.qkv[(size_t)rr * kGdnQkvWidth + ch]
          : p >= 0 ? a.input.hist[(size_t)(p & 15) * kGdnQkvWidth + ch] : 0.f;
      x += a.input.conv[ch * 4 + tap] * raw;
    }
    return Silu(x);
  };
  if (blockIdx.x < 16) {
    const int kh = blockIdx.x, j = t & 127, half = t >> 7;
    const size_t at = ((size_t)kh * rows + r) * 128 + j;
    if (r >= a.input.R) { (half ? a.k : a.q)[at] = 0.f; return; }
    const float x = conv(half ? 2048 + kh * 128 + j : kh * 128 + j);
    const float sq = BlockSum<256>(half ? 0.f : x * x, red);
    const float sk = BlockSum<256>(half ? x * x : 0.f, red);
    if (half) a.k[at] = x * rsqrtf(sk + kEps);
    else a.q[at] = x * rsqrtf(sq + kEps) * kHadNorm;
    return;
  }
  if constexpr (V) {
    const int c = (blockIdx.x - 16) * 256 + t, h = c / 128, j = c % 128;
    const size_t b = (size_t)h * rows + r;
    a.v[b * 128 + j] = r < a.input.R ? conv(4096 + c) : 0.f;
  }
  if (blockIdx.x == 16 && t < 48) {
    const size_t bt = (size_t)t * rows + r;
    if (r >= a.input.R) { a.g[bt] = a.beta[bt] = 0.f; return; }
    const float x = a.ab_logits[(size_t)r * 96 + t] + a.input.dt_bias[t];
    const float softplus = x > 20.f ? x : log1pf(expf(x));
    a.g[bt] = -__expf(a.input.A_log[t]) * softplus;
    a.beta[bt] = Sigmoid(a.ab_logits[(size_t)r * 96 + 48 + t]);
  }
}

// vscan prep: GdnLanesPrepKernel<false> with 4 rows per block.
// The 7 raw taps of the 4 rows load once; each row keeps the conv expression
// and the two BlockSum calls of GdnLanesPrepKernel, so the values are equal.
// Grid x: 16 q/k blocks, then one block for g and beta of the 4 rows.
constexpr int kGdnPrepRows = 4;
// step and xoff: block (x, y) takes rows step*y .. step*y+3 as block kind x+xoff. The halo
// launch (step = the input GEMM's row tile, xoff 0) fills the q/k rows that GdnPrepEpi leaves
// out; the g/beta launch has xoff 16.
__global__ void __launch_bounds__(256) GdnLanesPrepQkKernel(GdnPrefill a, int step = kGdnPrepRows, int xoff = 0) {
  __shared__ float red[8];
  const int r0 = blockIdx.y * step, t = threadIdx.x, bx = blockIdx.x + xoff;
  const size_t rows = (size_t)a.chunks * 64;
  if (bx == 16) {
    if (t >= 48 * kGdnPrepRows) return;
    const int r = r0 + t / 48, hh = t % 48;
    const size_t bt = (size_t)hh * rows + r;
    if (r >= a.input.R) { a.g[bt] = a.beta[bt] = 0.f; return; }
    const float x = a.ab_logits[(size_t)r * 96 + hh] + a.input.dt_bias[hh];
    const float softplus = x > 20.f ? x : log1pf(expf(x));
    a.g[bt] = -__expf(a.input.A_log[hh]) * softplus;
    a.beta[bt] = Sigmoid(a.ab_logits[(size_t)r * 96 + 48 + hh]);
    return;
  }
  const int kh = bx, j = t & 127, half = t >> 7;
  const int ch = half ? 2048 + kh * 128 + j : kh * 128 + j;
  float cw[4], raw[kGdnPrepRows + 3];
#pragma unroll
  for (int tap = 0; tap < 4; ++tap) cw[tap] = a.input.conv[ch * 4 + tap];
#pragma unroll
  for (int i = 0; i < kGdnPrepRows + 3; ++i) {
    const int rr = r0 - 3 + i, p = a.input.pos0 + rr;
    raw[i] = rr >= a.input.R ? 0.f : rr >= 0 ? a.input.qkv[(size_t)rr * kGdnQkvWidth + ch]
        : p >= 0 ? a.input.hist[(size_t)(p & 15) * kGdnQkvWidth + ch] : 0.f;
  }
#pragma unroll
  for (int u = 0; u < kGdnPrepRows; ++u) {
    const int r = r0 + u;
    const size_t at = ((size_t)kh * rows + r) * 128 + j;
    if (r >= a.input.R) { (half ? a.k : a.q)[at] = 0.f; continue; }  // uniform in the block
    float x = 0.f;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) x += cw[tap] * raw[u + tap];
    x = Silu(x);
    const float sq = BlockSum<256>(half ? 0.f : x * x, red);
    const float sk = BlockSum<256>(half ? x * x : 0.f, red);
    if (half) a.k[at] = x * rsqrtf(sk + kEps);
    else a.q[at] = x * rsqrtf(sq + kEps) * kHadNorm;
  }
}

// The q/k prep of GdnLanesPrepQkKernel as the epilogue of the GDN input GEMM
// (PrefillGemmOutKernel EPI). A GEMM block owns 128 columns (one q or k
// head) and rows [m0, m1) of the launch; after it stored the raw rows, each of
// its 4 waves preps rows m0+3+w, m0+7+w, ... from those raw rows (L2). The
// first 3 rows of a block need raw rows of another block: the halo launch of
// GdnLanesPrepQkKernel (step = the row tile) does them. Lane t keeps columns
// 4t..4t+3, so the BlockSum tree of the prep kernel (pairs of columns, then
// lanes, then the 4 wave sums in order) is one wave's shuffle tree here, and
// every value is the same bit for bit. q and k are launch-wide, in parts of
// 2,048 rows: [part][16][rows of the part][128].
constexpr size_t kGdnPrepPartFloats = (size_t)16 * 2048 * 128;
struct GdnPrepEpi {
  const float* qkv;  // the q/k/v job's dst; other jobs of the group are skipped
  float *q, *k;
  const float* conv;
  int R;             // launch rows
  __device__ void Run(const float* dst, int ldd, int m0, int m1, int c0) const {
    if (dst != qkv || c0 >= 4096) return;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, waves = blockDim.x >> 5;
    const int c = c0 + lane * 4, kh = (c0 & 2047) >> 7;
    const bool is_k = c0 >= 2048;
    float cw[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int tap = 0; tap < 4; ++tap) cw[i][tap] = conv[(c + i) * 4 + tap];
    for (int row = m0 + 3 + wave; row < m1; row += waves) {
      float4 raw[4];
#pragma unroll
      for (int tap = 0; tap < 4; ++tap)
        raw[tap] = *reinterpret_cast<const float4*>(dst + (size_t)(row - 3 + tap) * ldd + c);
      float x[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        x[i] = 0.f;  // contracted to an FMA chain, as in the prep kernel
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) x[i] += cw[i][tap] * reinterpret_cast<const float*>(&raw[tap])[i];
        x[i] = Silu(x[i]);
      }
      float t;
      {
        // No FMA here: in the prep kernel a lane exchange separates each square from its first add.
        // (__fmul_rn and __fadd_rn do not stop the contraction in HIP.)
#pragma clang fp contract(off)
        // WaveSum over the 32 columns of a prep wave: column pairs, then lanes 1, 2, 4 apart.
        float v = (x[0] * x[0] + x[1] * x[1]) + (x[2] * x[2] + x[3] * x[3]);
#pragma unroll
        for (int m = 1; m < 8; m <<= 1) v += __shfl_xor(v, m);
        // BlockSum: the wave sums in order (the zeros of the other half do not change the value).
        t = __shfl(v, 0);
        t += __shfl(v, 8); t += __shfl(v, 16); t += __shfl(v, 24);
      }
      const int part = row >> 11, r = row & 2047, rows = min(2048, (R - (part << 11) + 63) & ~63);
      const float inv = rsqrtf(t + kEps);
      float4 o;
      float* op = reinterpret_cast<float*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) op[i] = is_k ? x[i] * inv : x[i] * inv * kHadNorm;
      *reinterpret_cast<float4*>((is_k ? k : q) + part * kGdnPrepPartFloats + ((size_t)kh * rows + r) * 128 + lane * 4) = o;
    }
  }
};

// Four lanes share one value column: each keeps 32 of its 128 key rows, so
// 48 heads x 128 columns give 768 waves (about 10 per SIMD) instead of 192,
// and every row's dependent chain is 32 FMAs plus two lane exchanges. k and q
// come straight from global memory; the 8 columns of a wave read the same
// addresses. The sums run in a different order from the decode kernel.
// LAZY keeps S_true = c * S with a per-head scalar c, so the decay costs one
// scalar multiply per row instead of 32 per lane (3 FMAs per state element,
// not 4). c is renormalized into S when it falls below 2^-20.
template <bool LAZY = false>
__global__ void __launch_bounds__(128) GdnScanLanesKernel(GdnPrefill a) {
  const int h=blockIdx.x>>2, lane=threadIdx.x&31, sub=lane&3;
  const int j=(blockIdx.x&3)*32+(threadIdx.x>>5)*8+(lane>>2);
  float S[32];
  float* dst=a.input.state+(size_t)h*128*128+(size_t)sub*32*128+j;
#pragma unroll
  for (int i=0;i<32;++i) S[i]=dst[i*128];
  const size_t rows=(size_t)a.chunks*64, base=(size_t)h*rows, kbase=(size_t)(h/3)*rows;
  float scale=1.f;
#pragma unroll 1
  for (int r=0;r<a.input.R;++r) {
    const size_t b=base+r;
    const float4* kp=reinterpret_cast<const float4*>(a.k+(kbase+r)*128+sub*32);
    const float4* qp=reinterpret_cast<const float4*>(a.q+(kbase+r)*128+sub*32);
    float k[32],q[32];
#pragma unroll
    for (int i=0;i<8;++i) {
      const float4 kv=kp[i],qv=qp[i];
      k[4*i]=kv.x;k[4*i+1]=kv.y;k[4*i+2]=kv.z;k[4*i+3]=kv.w;
      q[4*i]=qv.x;q[4*i+1]=qv.y;q[4*i+2]=qv.z;q[4*i+3]=qv.w;
    }
    const float decay=expf(a.g[b]),beta=a.beta[b],v=a.v[b*128+j];
    if constexpr (LAZY) {
      scale*=decay;
      if (scale<0x1p-20f) {  // uniform: every lane of the block has head h
#pragma unroll
        for (int i=0;i<32;++i) S[i]*=scale;
        scale=1.f;
      }
      float m0=0,m1=0;
#pragma unroll
      for (int i=0;i<32;i+=2) { m0+=S[i]*k[i];m1+=S[i+1]*k[i+1]; }
      float mem=m0+m1;
      mem+=__shfl_xor(mem,1);mem+=__shfl_xor(mem,2);
      const float delta=beta*(v-mem*scale), d=delta/scale;
      float y0=0,y1=0;
#pragma unroll
      for (int i=0;i<32;i+=2) {
        S[i]+=k[i]*d;S[i+1]+=k[i+1]*d;
        y0+=S[i]*q[i];y1+=S[i+1]*q[i+1];
      }
      float y=y0+y1;
      y+=__shfl_xor(y,1);y+=__shfl_xor(y,2);
      if (sub==0) a.y[b*128+j]=y*scale;
      continue;
    }
    float m0=0,m1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) {
      S[i]*=decay;S[i+1]*=decay;
      m0+=S[i]*k[i];m1+=S[i+1]*k[i+1];
    }
    float mem=m0+m1;
    mem+=__shfl_xor(mem,1);mem+=__shfl_xor(mem,2);
    const float delta=beta*(v-mem);
    float y0=0,y1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) {
      S[i]+=k[i]*delta;S[i+1]+=k[i+1]*delta;
      y0+=S[i]*q[i];y1+=S[i+1]*q[i+1];
    }
    float y=y0+y1;
    y+=__shfl_xor(y,1);y+=__shfl_xor(y,2);
    if (sub==0) a.y[b*128+j]=y;
  }
  if constexpr (LAZY) {
#pragma unroll
    for (int i=0;i<32;++i) S[i]*=scale;
  }
#pragma unroll
  for (int i=0;i<32;++i) dst[i*128]=S[i];
}

// The vscan scan: GdnScanLanesKernel<true> that also computes v from the
// raw qkv rows (the prep kernel then does not write v: 48 KB less DRAM traffic
// per row). The four lanes of a column (a DPP quad) share each 4-row block:
// lane sub computes v, the decay and beta of row r0+sub once, and row r0+u
// takes them from lane u of its quad. The conv is the prep kernel's expression
// (same taps, same order, same Silu) and the row body is that of the LAZY scan,
// so every value is the same bit for bit.
template <int U> __device__ __forceinline__ float GdnQuadBroadcast(float x) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), U | U << 2 | U << 4 | U << 6, 0xf, 0xf, false));
}

__global__ void __launch_bounds__(128) GdnScanLanesVKernel(GdnPrefill a) {
  const int h=blockIdx.x>>2, lane=threadIdx.x&31, sub=lane&3;
  const int j=(blockIdx.x&3)*32+(threadIdx.x>>5)*8+(lane>>2), ch=4096+h*128+j;
  float cw[4];
#pragma unroll
  for (int tap=0;tap<4;++tap) cw[tap]=a.input.conv[ch*4+tap];
  float S[32];
  float* dst=a.input.state+(size_t)h*128*128+(size_t)sub*32*128+j;
#pragma unroll
  for (int i=0;i<32;++i) S[i]=dst[i*128];
  const size_t rows=(size_t)a.chunks*64, base=(size_t)h*rows, kbase=(size_t)(h/3)*rows;
  float scale=1.f;
  auto step=[&](int r,float v,float decay,float beta) {
    const size_t b=base+r;
    const float4* kp=reinterpret_cast<const float4*>(a.k+(kbase+r)*128+sub*32);
    const float4* qp=reinterpret_cast<const float4*>(a.q+(kbase+r)*128+sub*32);
    float k[32],q[32];
#pragma unroll
    for (int i=0;i<8;++i) {
      const float4 kv=kp[i],qv=qp[i];
      k[4*i]=kv.x;k[4*i+1]=kv.y;k[4*i+2]=kv.z;k[4*i+3]=kv.w;
      q[4*i]=qv.x;q[4*i+1]=qv.y;q[4*i+2]=qv.z;q[4*i+3]=qv.w;
    }
    scale*=decay;
    if (scale<0x1p-20f) {  // uniform: every lane of the block has head h
#pragma unroll
      for (int i=0;i<32;++i) S[i]*=scale;
      scale=1.f;
    }
    float m0=0,m1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) { m0+=S[i]*k[i];m1+=S[i+1]*k[i+1]; }
    float mem=m0+m1;
    mem+=__shfl_xor(mem,1);mem+=__shfl_xor(mem,2);
    const float delta=beta*(v-mem*scale), d=delta/scale;
    float y0=0,y1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) {
      S[i]+=k[i]*d;S[i+1]+=k[i+1]*d;
      y0+=S[i]*q[i];y1+=S[i+1]*q[i+1];
    }
    float y=y0+y1;
    y+=__shfl_xor(y,1);y+=__shfl_xor(y,2);
    if (sub==0) a.y[b*128+j]=y*scale;
  };
  // The raw taps, g and beta of the next block load while this block runs.
  float raw[4]={0.f,0.f,0.f,0.f},gl=0.f,bl=0.f;
  auto load=[&](int r0) {
    const int rs=r0+sub;
    if (rs>=a.input.R) return;
#pragma unroll
    for (int tap=0;tap<4;++tap) {
      const int rr=rs-3+tap,p=a.input.pos0+rr;
      raw[tap]=rr>=0 ? a.input.qkv[(size_t)rr*kGdnQkvWidth+ch]
          : p>=0 ? a.input.hist[(size_t)(p&15)*kGdnQkvWidth+ch] : 0.f;
    }
    gl=a.g[base+rs]; bl=a.beta[base+rs];
  };
  load(0);
#pragma unroll 1
  for (int r0=0;r0<a.input.R;r0+=4) {
    float vq=0.f,dq=1.f,bq=0.f;
    if (r0+sub<a.input.R) {
      float x=0.f;
#pragma unroll
      for (int i=0;i<4;++i) {
#ifndef CRUCIBLE_GDN_VSCAN_MUTANT
        const int tap=i;
#else  // bench mutant: the taps in reverse order, so the bitwise check must fail
        const int tap=3-i;
#endif
        x+=cw[tap]*raw[tap];
      }
      vq=Silu(x); dq=expf(gl); bq=bl;
    }
    load(r0+4);
    step(r0,GdnQuadBroadcast<0>(vq),GdnQuadBroadcast<0>(dq),GdnQuadBroadcast<0>(bq));
    if (r0+1<a.input.R) step(r0+1,GdnQuadBroadcast<1>(vq),GdnQuadBroadcast<1>(dq),GdnQuadBroadcast<1>(bq));
    if (r0+2<a.input.R) step(r0+2,GdnQuadBroadcast<2>(vq),GdnQuadBroadcast<2>(dq),GdnQuadBroadcast<2>(bq));
    if (r0+3<a.input.R) step(r0+3,GdnQuadBroadcast<3>(vq),GdnQuadBroadcast<3>(dq),GdnQuadBroadcast<3>(bq));
  }
#pragma unroll
  for (int i=0;i<32;++i) S[i]*=scale;
#pragma unroll
  for (int i=0;i<32;++i) dst[i*128]=S[i];
}

// GdnScanLanesVKernel with k in SGPRs. In the lanes kernels the 8
// columns of a wave load the same k and q (16 float4 loads per lane and row),
// and the load latency of each row is on the row's dependent chain. Here wave
// w of a block keeps key rows 32w..32w+31 and lane t keeps column t, so k and q
// of a row are wave-uniform. k of row r+1 loads into the other of two SGPR
// sets and q of row r+1 into the other of two VGPR sets while row r runs. The
// 4 partial sums of a column meet in LDS with the pairing of the quad shuffles,
// (P0+P1)+(P2+P3), so every value is the same bit for bit. Wave w computes the
// conv of row r0+w of each 4-row block. One LDS barrier per row; wave 0 stores
// y of row r-1 after the barrier of row r.
__global__ void __launch_bounds__(128) GdnScanSgprKernel(GdnPrefill a, const float* __restrict__ kv) {
  __shared__ float4 memx[2][32], yx[2][32];
  __shared__ float vx[2][4][32];
  const int h=blockIdx.x>>2, lane=threadIdx.x&31, sub=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int j=(blockIdx.x&3)*32+lane, ch=4096+h*128+j, R=a.input.R;
  float cw[4];
#pragma unroll
  for (int tap=0;tap<4;++tap) cw[tap]=a.input.conv[ch*4+tap];
  float S[32];
  float* dst=a.input.state+(size_t)h*128*128+(size_t)sub*32*128+j;
#pragma unroll
  for (int i=0;i<32;++i) S[i]=dst[i*128];
  const size_t rows=(size_t)a.chunks*64, base=(size_t)h*rows, kbase=(size_t)(h/3)*rows;
  float scale=1.f, pscale=1.f, raw[4], kb[2][32], qb[2][32];
  auto load=[&](int r0) {
    const int rs=r0+sub;
#pragma unroll
    for (int tap=0;tap<4;++tap) {
      const int rr=rs-3+tap,p=a.input.pos0+rr;
      raw[tap]=rs>=R ? 0.f : rr>=0 ? a.input.qkv[(size_t)rr*kGdnQkvWidth+ch]
          : p>=0 ? a.input.hist[(size_t)(p&15)*kGdnQkvWidth+ch] : 0.f;
    }
  };
  auto conv=[&](int r0,int slot) {
    if (r0+sub>=R) return;
    float x=0.f;
#pragma unroll
    for (int i=0;i<4;++i) {
#ifndef CRUCIBLE_GDN_VSCAN_MUTANT
      const int tap=i;
#else
      const int tap=3-i;
#endif
      x+=cw[tap]*raw[tap];
    }
    vx[slot][sub][lane]=Silu(x);
  };
  auto kl=[&](float (&kd)[32],int r) {
    const float* kp=kv+(kbase+r)*128+sub*32;
#pragma unroll
    for (int i=0;i<32;++i) kd[i]=kp[i];
  };
  auto ql=[&](float (&qd)[32],int r) {
    const float4* qp=reinterpret_cast<const float4*>(a.q+(kbase+r)*128+sub*32);
#pragma unroll
    for (int i=0;i<8;++i) { const float4 t=qp[i]; qd[4*i]=t.x; qd[4*i+1]=t.y; qd[4*i+2]=t.z; qd[4*i+3]=t.w; }
  };
  // The LDS barrier waits only for LDS (lgkmcnt), not for the k, q, g and beta
  // loads in flight or the y stores (__syncthreads waits for all of them).
  auto barrier=[] { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); };
  load(0); conv(0,0); load(4); kl(kb[0],0);
  float gn=R>(lane&3) ? a.g[base+(lane&3)] : 0.f, bn=R>(lane&3) ? a.beta[base+(lane&3)] : 0.f;
  __syncthreads();
  auto step=[&](auto U,int r0,int slot,float dq,float bq) {
    constexpr int u=decltype(U)::value, P=u&1;
    const int r=r0+u;
    // Lane 4m+u holds row r0+u: the quad broadcast keeps decay and scale in VGPRs
    // (VALU arithmetic, as in GdnScanLanesVKernel; no SALU float).
    const float decay=GdnQuadBroadcast<u>(dq), beta=GdnQuadBroadcast<u>(bq);
    float (&k)[32]=kb[P]; float (&q)[32]=qb[P];
    scale*=decay;
    if (scale<0x1p-20f) {
#pragma unroll
      for (int i=0;i<32;++i) S[i]*=scale;
      scale=1.f;
    }
    float m0=0,m1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) { m0+=S[i]*k[i];m1+=S[i+1]*k[i+1]; }
    reinterpret_cast<float*>(&memx[r&1][lane])[sub]=m0+m1;
    barrier();
    if (sub==0 && r>0) {
      const float4 t=yx[(r-1)&1][lane];
      a.y[(base+r-1)*128+j]=((t.x+t.y)+(t.z+t.w))*pscale;
    }
    if (u==0) { conv(r0+4,slot^1); load(r0+8); }
    const float4 m4=memx[r&1][lane];
    const float mem=(m4.x+m4.y)+(m4.z+m4.w), v=vx[slot][u][lane];
    const float delta=beta*(v-mem*scale), d=delta/scale;
    // Issue the loads of row r+1 after the LDS reads are waited for (the SGPR
    // loads share lgkmcnt with LDS): the zero z depends on d. The loads are
    // unconditional (the last row loads row R-1 again), so no branch merges the
    // wait counters and the update below waits only for the loads of row r.
    int z;
    asm volatile("s_mov_b32 %0, 0" : "=s"(z) : "v"(d));
    // q of the next block's first row loads at the top of that block: a vector
    // load in flight at the loop latch makes the compiler wait for it there.
    kl(kb[P^1],min(r+1,R-1)+z);
    if constexpr (u<3) ql(qb[P^1],min(r+1,R-1)+z);
    float y0=0,y1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) {
      S[i]+=k[i]*d;S[i+1]+=k[i+1]*d;
      y0+=S[i]*q[i];y1+=S[i+1]*q[i+1];
    }
    reinterpret_cast<float*>(&yx[r&1][lane])[sub]=y0+y1;
    pscale=scale;
  };
#pragma unroll 1
  for (int r0=0;r0<R;r0+=4) {
    const int slot=(r0>>2)&1, rl=r0+(lane&3), rn=rl+4;
    const float dq=rl<R ? expf(gn) : 1.f, bq=rl<R ? bn : 0.f;
    gn=rn<R ? a.g[base+rn] : 0.f; bn=rn<R ? a.beta[base+rn] : 0.f;
    ql(qb[0],r0);
    step(std::integral_constant<int,0>{},r0,slot,dq,bq);
    if (r0+1<R) step(std::integral_constant<int,1>{},r0,slot,dq,bq);
    if (r0+2<R) step(std::integral_constant<int,2>{},r0,slot,dq,bq);
    if (r0+3<R) step(std::integral_constant<int,3>{},r0,slot,dq,bq);
  }
  __syncthreads();
  if (sub==0 && R>0) {
    const float4 t=yx[(R-1)&1][lane];
    a.y[(base+R-1)*128+j]=((t.x+t.y)+(t.z+t.w))*pscale;
  }
#pragma unroll
  for (int i=0;i<32;++i) S[i]*=scale;
#pragma unroll
  for (int i=0;i<32;++i) dst[i*128]=S[i];
}

// One wave per (row, head): lane t holds y[4t..4t+3], the layout Wht128 needs.
// HEAD_RUNS (the vscan path): a block takes 8 consecutive rows of one
// head instead of 8 heads of one row, so its y reads are one 4 KB run and its
// packed xh writes are 256-byte runs. Same arithmetic per element.
template <bool HEAD_RUNS = false>
__global__ void __launch_bounds__(256) GdnLanesGateKernel(GdnPrefill a) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  int r, h;
  if constexpr (HEAD_RUNS) {
    h = blockIdx.x % 48; r = blockIdx.x / 48 * 8 + (threadIdx.x >> 5);
    if (r >= a.input.R) return;
  } else {
    if (w >= a.input.R * 48) return;
    r = w / 48; h = w % 48;
  }
  const int j = lane * 4;
  const size_t b = (size_t)h * a.chunks * 64 + r;
  const float4 yv = *reinterpret_cast<const float4*>(a.y + b * 128 + j);
  const float y[4] = {yv.x, yv.y, yv.z, yv.w};
  const float inv = rsqrtf(WaveSum(y[0]*y[0] + y[1]*y[1] + y[2]*y[2] + y[3]*y[3]) / 128.f + kEps);
  float v[4];
  if constexpr (HEAD_RUNS) {
    // One 16-byte z load and one 8-byte xh store per lane: the 4 columns of a
    // lane are adjacent in both xh layouts (j % 4 == 0). Same values.
    const float4 zv = *reinterpret_cast<const float4*>(a.input.z + (size_t)r * kGdnValueWidth + h * 128 + j);
    const float z[4] = {zv.x, zv.y, zv.z, zv.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float o = y[i] * inv * a.input.gamma[j + i] * Sigmoid(z[i]);
      v[i] = RoundH(RoundH(o) * __half2float(a.input.suh_out[h * 128 + j + i]));
    }
    Wht128(v);
    union { __half h[4]; uint2 u; } out;
#pragma unroll
    for (int i = 0; i < 4; ++i) out.h[i] = __float2half(v[i] * kHadNorm);
    *reinterpret_cast<uint2*>(a.input.xh_out + OutProjIndex(r, h * 128 + j, a.input.pack_rows, a.input.pack_off)) = out.u;
    return;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float o = y[i] * inv * a.input.gamma[j + i] * Sigmoid(a.input.z[(size_t)r * kGdnValueWidth + h * 128 + j + i]);
    v[i] = RoundH(RoundH(o) * __half2float(a.input.suh_out[h * 128 + j + i]));
  }
  Wht128(v);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    a.input.xh_out[OutProjIndex(r, h * 128 + j + i, a.input.pack_rows, a.input.pack_off)] =
        __float2half(v[i] * kHadNorm);
}

// vscan: the v conv in the scan, the only lazy path since train 20. Same values bit for bit as
// the older lanes path, which bench/kernels/prefill_gdn_lanes.h keeps as the bench reference.
// sgpr=false keeps GdnScanLanesVKernel as the bench reference (bench/gdn_vscan.hip).
// CRUCIBLE_PF_GDN_EAGER selects the eager-decay scan for A/B tests (other arithmetic).
inline bool GdnEager() { static const bool eager=NumericsOn("CRUCIBLE_PF_GDN_EAGER"); return eager; }

// halo > 0: GdnPrepEpi wrote the q/k rows of this part except the first 3 rows of each
// halo-row tile of the input GEMM; only those rows and g/beta are prepped here.
inline void LaunchGdnLanes(GdnPrefill a,hipStream_t stream=nullptr,bool sgpr=true,int halo=0) {
  ValidateGdnPrefill(a);
  const bool eager=GdnEager();
  if (halo>0 && eager) {
    std::fprintf(stderr,"GDN prep epilogue needs the vscan path, not the eager scan\n"); std::abort();
  }
  // GdnPrepEpi parts are 2,048 rows: each part must start at a GEMM tile, so that rows 0..2 of
  // each part are halo rows (prepped here from the history).
  if (halo>0 && 2048%halo) {
    std::fprintf(stderr,"GDN prep epilogue: row tile %d does not divide the 2,048-row part\n",halo);
    std::abort();
  }
  if (!eager) {
    if (halo>0) {
#ifndef GDN_PREP_EPI_BENCH_MUTANT
      GdnLanesPrepQkKernel<<<dim3(16,(a.input.R+halo-1)/halo),256,0,stream>>>(a,halo,0);
#endif  // the bench mutant drops the halo rows: the check must fail
      GdnLanesPrepQkKernel<<<dim3(1,a.chunks*64/kGdnPrepRows),256,0,stream>>>(a,kGdnPrepRows,16);
    } else {
      GdnLanesPrepQkKernel<<<dim3(16+1,a.chunks*64/kGdnPrepRows),256,0,stream>>>(a);
    }
    if (sgpr) GdnScanSgprKernel<<<48*4,128,0,stream>>>(a,a.k);
    else GdnScanLanesVKernel<<<48*4,128,0,stream>>>(a);
    GdnLanesGateKernel<true><<<(a.input.R+7)/8*48,256,0,stream>>>(a);
    GdnPrefillHistoryKernel<<<40,256,0,stream>>>(a);
    return;
  }
  GdnLanesPrepKernel<<<dim3(16+24,a.chunks*64),256,0,stream>>>(a);
  GdnScanLanesKernel<false><<<48*4,128,0,stream>>>(a);
  GdnLanesGateKernel<<<(a.input.R*48+7)/8,256,0,stream>>>(a);
  GdnPrefillHistoryKernel<<<40,256,0,stream>>>(a);
}

// q/k/v/y/ut require 48*ceil(R/64)*64*128 floats each, not one chunk.
inline void LaunchGdnPersistent(GdnPrefill a,hipStream_t stream=nullptr) {
  ValidateGdnPrefill(a);
  GdnPrefillPrepKernel<<<dim3(48,a.chunks*64),256,0,stream>>>(a);
  GdnPersistentColumnKernel<<<48,128,0,stream>>>(a);
  GdnPrefillGateKernel<<<dim3(48,a.input.R),256,0,stream>>>(a);
  GdnPrefillHistoryKernel<<<40,256,0,stream>>>(a);
}

inline void LaunchGdnPrefill(GdnPrefill a, hipStream_t stream = nullptr) {
  ValidateGdnPrefill(a);
  GdnPrefillPrepKernel<<<dim3(48, a.chunks * 64), 256, 0, stream>>>(a);
  GdnPrefillPrefixKernel<<<dim3(48, a.chunks), 128, 0, stream>>>(a);
  GdnPrefillGramKernel<<<dim3(4, a.chunks, 48), 128, 0, stream>>>(a);
  GdnPrefillInverseKernel<<<dim3(48, a.chunks), 64, 0, stream>>>(a);
  GdnPrefillWyKernel<<<dim3(2, 4, 48 * a.chunks * 2), 128, 0, stream>>>(a);
  GdnPrefillEndKeysKernel<<<dim3(48, a.chunks), 128, 0, stream>>>(a);
  for (int c = 0; c < a.chunks; ++c) {
    GdnPrefillCarryKernel<<<dim3(2, 4, 48), 128, 0, stream>>>(a, c);
    GdnPrefillOutputKernel<<<dim3(2, 4, 48), 128, 0, stream>>>(a, c);
    GdnPrefillStateKernel<<<dim3(2, 8, 48), 128, 0, stream>>>(a, c);
  }
  GdnPrefillGateKernel<<<dim3(48, a.input.R), 256, 0, stream>>>(a);
  GdnPrefillHistoryKernel<<<40, 256, 0, stream>>>(a);
}
__global__ void GdnCopyRawKernel(GdnPrefill full, const float* chunk, int start, int rows) {
  const int h = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
  if (i >= rows * 128) return;
  const size_t dst = ((size_t)h * full.chunks * 64 + start) * 128 + i;
  full.y[dst] = chunk[(size_t)h * 64 * 128 + i];
}

// Process each 64-row working set before the next one. This avoids streaming
// every prompt's Q/K/W/UT arrays through DRAM before the state scan starts.
inline void LaunchGdnMicro(GdnPrefill a, float* chunk_y, bool save_raw = false,
                           hipStream_t stream = nullptr) {
  ValidateGdnPrefill(a);
  if (!chunk_y || (save_raw && (!a.y || a.y == chunk_y))) {
    std::fprintf(stderr, "GDN microchunks require a separate raw-output workspace\n");
    std::abort();
  }
  for (int start = 0; start < a.input.R; start += 64) {
    GdnPrefill small = a;
    const int rows = a.input.R - start < 64 ? a.input.R - start : 64;
    small.input.qkv += (size_t)start * kGdnQkvWidth;
    small.input.z += (size_t)start * kGdnValueWidth;
    if (small.input.x) small.input.x += (size_t)start * kHidden;
    small.input.xh_out += (size_t)start * kGdnValueWidth;
    small.input.pos0 += start;
    small.input.R = small.input.commit = rows;
    small.ab_logits += (size_t)start * 96;
    small.chunks = 1;
    small.y = chunk_y;
    LaunchGdnPrefill(small, stream);
    if (save_raw) GdnCopyRawKernel<<<dim3((rows * 128 + 255) / 256, 48), 256, 0, stream>>>(a, chunk_y, start, rows);
  }
}
}  // namespace crucible::prefill
