// Vision tower kernels (HF qwen4_exp Qwen4ExpVisionModel). EXL3 linears run through PrefillLinear;
// these kernels cover patch embedding, position embedding, LayerNorm, bias epilogues, 2D RoPE, and
// non-causal attention. Patches use merge-block order (four consecutive patches form one 2x2 cell).
#pragma once
#include "../../gfx1151/common.h"

namespace crucible::vision {
using H16 = _Float16 __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));

constexpr int kHidden = 1152, kHeads = 16, kHeadDim = 72, kHeadPad = 80, kPatchIn = 1536;
constexpr int kInter = 4352, kMergeIn = 4608, kOut = crucible::kHidden, kPosSide = 48, kDepth = 27;
constexpr float kLnEps = 1e-6f;
constexpr float kTheta = 10000.f;

// Patch n in merge-block order -> its patch row and column.
__device__ __forceinline__ void PatchRowCol(int n, int gw, int& row, int& col) {
  const int cells = gw / 2, cell = n / 4, in = n % 4;
  row = (cell / cells) * 2 + in / 2;
  col = (cell % cells) * 2 + in % 2;
}

// fp16 patches [N][1536]: channel, temporal copy, 16x16 pixels; (x / 255 - 0.5) / 0.5 in fp32.
__global__ void PatchifyKernel(const uint8_t* __restrict__ rgb, int width, int gw, int n_patches,
                               __half* __restrict__ out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n_patches * kPatchIn) return;
  const int n = int(i / kPatchIn), e = int(i % kPatchIn);
  const int c = e / 512, py = (e / 16) % 16, px = e % 16;
  int row, col;
  PatchRowCol(n, gw, row, col);
  const float v = float(rgb[((size_t)(row * 16 + py) * width + col * 16 + px) * 3 + c]);
  out[i] = __float2half((v * (1.f / 255.f) - 0.5f) / 0.5f);
}

// x[n] = bilinear (align_corners) resample of the 48x48 table, fp32 taps and weights as HF.
__global__ void __launch_bounds__(128) PosEmbedKernel(const float* __restrict__ table, int gh, int gw,
                                                      float* __restrict__ x) {
  const int n = blockIdx.x;
  int row, col;
  PatchRowCol(n, gw, row, col);
  auto taps = [](int index, int size, int t[2], float w[2]) {
    const float src = float(index) * float(kPosSide - 1) / float(max(size - 1, 1));
    const float lo = floorf(src);
    for (int k = 0; k < 2; ++k) {
      w[k] = fmaxf(1.f - fabsf(src - lo - float(k)), 0.f);
      t[k] = min(max(int(lo) + k, 0), kPosSide - 1);
    }
  };
  int ht[2], wt[2];
  float hw[2], ww[2];
  taps(row, gh, ht, hw);
  taps(col, gw, wt, ww);
  for (int d = threadIdx.x; d < kHidden; d += blockDim.x) {
    float s = 0.f;
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int b = 0; b < 2; ++b) s += table[(size_t)(ht[a] * kPosSide + wt[b]) * kHidden + d] * (hw[a] * ww[b]);
    x[(size_t)n * kHidden + d] = s;
  }
}

// x[n][o] += P[n] . W[o] + bias[o]. One wave per 16x16 tile, WMMA over K = 1536.
__global__ void __launch_bounds__(128) PatchEmbedKernel(const __half* __restrict__ patches, int n_patches,
                                                        const __half* __restrict__ w, const __half* __restrict__ bias,
                                                        float* __restrict__ x) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int r0 = blockIdx.x * 16, o0 = (blockIdx.y * 4 + wave) * 16;
  const int ar = min(r0 + (lane & 15), n_patches - 1), bo = o0 + (lane & 15);
  F8 acc = {};
  for (int k = 0; k < kPatchIn; k += 16) {
    H16 a, b;
    const uint4* ap = reinterpret_cast<const uint4*>(patches + (size_t)ar * kPatchIn + k);
    const uint4* bp = reinterpret_cast<const uint4*>(w + (size_t)bo * kPatchIn + k);
    reinterpret_cast<uint4*>(&a)[0] = ap[0]; reinterpret_cast<uint4*>(&a)[1] = ap[1];
    reinterpret_cast<uint4*>(&b)[0] = bp[0]; reinterpret_cast<uint4*>(&b)[1] = bp[1];
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = r0 + 2 * i + (lane >> 4);
    if (r < n_patches) x[(size_t)r * kHidden + bo] += acc[i] + __half2float(bias[bo]);
  }
}

// LayerNorm over 1152 channels, fp32 two-pass statistics. Row r of src writes row r of dst.
__global__ void __launch_bounds__(128) LayerNormKernel(const float* __restrict__ src, int rows,
                                                       const float* __restrict__ w, const float* __restrict__ b,
                                                       float* __restrict__ dst) {
  __shared__ float red[4];
  const int r = blockIdx.x, t = threadIdx.x;
  if (r >= rows) return;
  const float* x = src + (size_t)r * kHidden;
  float v[9], s = 0.f;
#pragma unroll
  for (int i = 0; i < 9; ++i) { v[i] = x[t + 128 * i]; s += v[i]; }
  auto sum = [&](float value) {
    for (int o = 16; o; o >>= 1) value += __shfl_xor(value, o);
    if ((t & 31) == 0) red[t >> 5] = value;
    __syncthreads();
    const float total = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    return total;
  };
  const float mean = sum(s) / kHidden;
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < 9; ++i) { v[i] -= mean; q += v[i] * v[i]; }
  const float inv = rsqrtf(sum(q) / kHidden + kLnEps);
  float* y = dst + (size_t)r * kHidden;
#pragma unroll
  for (int i = 0; i < 9; ++i) y[t + 128 * i] = v[i] * inv * w[t + 128 * i] + b[t + 128 * i];
}

// q, k, v chunk rows [R][1152] fp32 (+ bias) -> fp16 Q, K [head][Npad][80] and V^T [head][80][Npad],
// rows r0.. of the image. q and k get the axial 2D RoPE (row on 0..17 and 36..53, column on the rest).
__global__ void __launch_bounds__(128) QkvPostKernel(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const __half* __restrict__ bq, const __half* __restrict__ bk,
    const __half* __restrict__ bv, int r0, int R, int gw, int npad,
    __half* __restrict__ Q, __half* __restrict__ K, __half* __restrict__ Vt) {
  const int r = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
  if (r >= R || j >= kHeadDim) return;
  const int n = r0 + r, c = h * kHeadDim + j;
  int row, col;
  PatchRowCol(n, gw, row, col);
  const int half = kHeadDim / 2, f = j % half;
  const int pos = f < half / 2 ? row : col;
  const float inv = 1.f / powf(kTheta, float(2 * (f % (half / 2))) / float(half));
  const float angle = float(pos) * inv, cs = cosf(angle), sn = sinf(angle);
  const int pair = j < half ? c + half : c - half;
  const float sign = j < half ? -1.f : 1.f;
  const size_t at = (size_t)r * kHidden;
  const float qx = q[at + c] + __half2float(bq[c]), qp = q[at + pair] + __half2float(bq[pair]);
  const float kx = k[at + c] + __half2float(bk[c]), kp = k[at + pair] + __half2float(bk[pair]);
  const size_t qk = ((size_t)h * npad + n) * kHeadPad + j;
  Q[qk] = __float2half(qx * cs + sign * qp * sn);
  K[qk] = __float2half(kx * cs + sign * kp * sn);
  Vt[((size_t)h * kHeadPad + j) * npad + n] = __float2half(v[at + c] + __half2float(bv[c]));
}

// Non-causal attention, one head per blockIdx.y, 64 queries per block (16 per wave), 32 keys per step.
// Scores and the softmax are fp32; P and V enter WMMA as fp16. Output fp32 [N][1152].
__global__ void __launch_bounds__(128) AttentionKernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                                       const __half* __restrict__ Vt, int n, int npad,
                                                       float* __restrict__ out) {
  __shared__ _Float16 pl[4][16][32];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, h = blockIdx.y;
  const int q0 = blockIdx.x * 64 + wave * 16;
  const __half* qh = Q + (size_t)h * npad * kHeadPad;
  const __half* kh = K + (size_t)h * npad * kHeadPad;
  const __half* vh = Vt + (size_t)h * kHeadPad * npad;
  auto load16 = [](const __half* p) {
    H16 f;
    reinterpret_cast<uint4*>(&f)[0] = reinterpret_cast<const uint4*>(p)[0];
    reinterpret_cast<uint4*>(&f)[1] = reinterpret_cast<const uint4*>(p)[1];
    return f;
  };
  H16 qf[5];
  const int qrow = min(q0 + (lane & 15), npad - 1);
#pragma unroll
  for (int d = 0; d < 5; ++d) qf[d] = load16(qh + (size_t)qrow * kHeadPad + d * 16);
  constexpr float kScale = 0.11785113019775792f * 1.4426950408889634f;  // 72^-0.5 * log2(e)
  F8 o[5] = {};
  float m[8], l[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { m[i] = -INFINITY; l[i] = 0.f; }
  for (int k0 = 0; k0 < n; k0 += 32) {
    F8 s[2] = {};
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int key = k0 + 16 * t + (lane & 15);
#pragma unroll
      for (int d = 0; d < 5; ++d)
        s[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[d], load16(kh + (size_t)key * kHeadPad + d * 16), s[t]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float a = k0 + (lane & 15) < n ? s[0][i] * kScale : -INFINITY;
      float b = k0 + 16 + (lane & 15) < n ? s[1][i] * kScale : -INFINITY;
      float mx = fmaxf(a, b);
      for (int x = 8; x; x >>= 1) mx = fmaxf(mx, __shfl_xor(mx, x));
      const float next = fmaxf(m[i], mx), alpha = exp2f(m[i] - next);
      m[i] = next;
      a = exp2f(a - next);
      b = exp2f(b - next);
      l[i] = l[i] * alpha + a + b;
#pragma unroll
      for (int d = 0; d < 5; ++d) o[d][i] *= alpha;
      const int prow = 2 * i + (lane >> 4);
      pl[wave][prow][lane & 15] = (_Float16)a;
      pl[wave][prow][16 + (lane & 15)] = (_Float16)b;
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      H16 pf;
#pragma unroll
      for (int e = 0; e < 16; ++e) pf[e] = pl[wave][lane & 15][16 * t + e];
      const int key = k0 + 16 * t;
#pragma unroll
      for (int d = 0; d < 5; ++d)
        o[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pf, load16(vh + (size_t)(d * 16 + (lane & 15)) * npad + key), o[d]);
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float total = l[i];
    for (int x = 8; x; x >>= 1) total += __shfl_xor(total, x);
    const int row = q0 + 2 * i + (lane >> 4);
    if (row >= n) continue;
#pragma unroll
    for (int d = 0; d < 5; ++d) {
      const int dim = d * 16 + (lane & 15);
      if (dim < kHeadDim) out[(size_t)row * kHidden + h * kHeadDim + dim] = o[d][i] / total;
    }
  }
}

// x[r] += y[r] + bias, rows r0.. of x.
__global__ void ResidualKernel(float* __restrict__ x, const float* __restrict__ y, const __half* __restrict__ bias,
                               int rows) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)rows * kHidden) return;
  x[i] += y[i] + __half2float(bias[i % kHidden]);
}

// In place: y = act(y + bias). TANH selects gelu_pytorch_tanh (blocks); otherwise erf GELU (merger).
template <bool TANH>
__global__ void GeluKernel(float* __restrict__ y, const __half* __restrict__ bias, int rows, int cols) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)rows * cols) return;
  const float x = y[i] + __half2float(bias[i % cols]);
  if constexpr (TANH) y[i] = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
  else y[i] = 0.5f * x * (1.f + erff(x * 0.7071067811865476f));
}

// Merger output + bias, rounded to bf16 like a token embedding row.
__global__ void OutputKernel(const float* __restrict__ y, const __half* __restrict__ bias, int rows,
                             uint16_t* __restrict__ out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)rows * kOut) return;
  const float x = y[i] + __half2float(bias[i % kOut]);
  uint32_t u;
  __builtin_memcpy(&u, &x, 4);
  u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even; outputs are finite
  out[i] = uint16_t(u >> 16);
}

// Embedding rows of image tokens replace the token-table rows in all four streams.
__global__ void OverlayKernel(const uint16_t* __restrict__ embeds, float* __restrict__ H, int ldh) {
  const int r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t u = uint32_t(embeds[(size_t)r * kOut + d]) << 16;
  float e;
  __builtin_memcpy(&e, &u, 4);
#pragma unroll
  for (int s = 0; s < 4; ++s) H[(size_t)r * ldh + s * kOut + d] = e;
}
}  // namespace crucible::vision
