#pragma once
#include <cmath>
#include <random>
#include <limits>
#include <map>
#include "kernels/exl3_pack_x.h"
#include "../engine/exl3/kernels/exl3_gemm.h"
#include "kernels/exl3_gemm_tiled.h"
#include "kernels/exl3_gemm_pipe.h"
#include "common.h"
#include "../engine/exl3/safetensors.h"

namespace gm {
using namespace crucible;
struct Matrix {
  std::string name;
  int I, O, K;
  std::vector<uint32_t> host;
  uint32_t* dev = nullptr;
};
inline void Require(bool ok, const std::string& why) {
  if (!ok) { std::fprintf(stderr, "%s\n", why.c_str()); std::exit(2); }
}
inline Matrix Load(st::Checkpoint& ck, const std::string& name) {
  const auto& t = ck.Info(name + ".trellis");
  Require(t.dtype == "I16" && t.shape.size() == 3 && t.shape[2] >= 48 && t.shape[2] <= 80 && t.shape[2] % 16 == 0,
          "unsupported trellis: " + name);
  const int64_t limit = std::numeric_limits<int>::max() / 16;
  Require(t.shape[0] > 0 && t.shape[0] <= limit && t.shape[1] > 0 && t.shape[1] <= limit,
          "invalid trellis dimensions: " + name);
  const int I = t.shape[0] * 16, O = t.shape[1] * 16, K = t.shape[2] / 16;
  Require(I % 128 == 0 && O % 128 == 0 && K >= 3 && K <= 5,
          "unsupported dimensions/rate: " + name);
  Require(ck.Has(name + ".mul1") && !ck.Has(name + ".mcg"), "unsupported codebook: " + name);
  Require(t.end - t.begin == (size_t)I * O * K / 8, "unexpected trellis byte count: " + name);
  size_t marker_bytes;
  const auto* marker_data = ck.Data(name + ".mul1", &marker_bytes);
  Require(marker_bytes == 4, "mul1 must be a scalar int32: " + name);
  uint32_t marker;
  std::memcpy(&marker, marker_data, 4);
  Require(marker == kMul1, "unknown mul1: " + name);
  Matrix m{name, I, O, K, std::vector<uint32_t>((size_t)I * O * K / 32)};
  const auto* src = reinterpret_cast<const uint32_t*>(ck.Data(name + ".trellis"));
  const int A = I / 16, B = O / 16, T = 8 * K;
  for (int a = 0; a < A; ++a)
    for (int b = 0; b < B; ++b)
      std::memcpy(m.host.data() + ((size_t)b * A + a) * T,
                  src + ((size_t)a * B + b) * T, T * 4);
  return m;
}
inline void Upload(Matrix& m) {
  HIP_CHECK(hipMalloc(&m.dev, m.host.size() * 4));
  HIP_CHECK(hipMemcpy(m.dev, m.host.data(), m.host.size() * 4, hipMemcpyHostToDevice));
}
// Literal bit-ring oracle deliberately does not share LaneWindow or DecodeLane.
inline float Weight(const Matrix& m, int i, int o) {
  const int r = i & 15, c = o & 15;
  const int lane = 4 * (c & 7) + (r & 7) / 2;
  const int pos = 8 * lane + (r & 1) + ((r & 8) ? 2 : 0) + ((c & 8) ? 4 : 0);
  const int bits = 256 * m.K;
  const auto* t = m.host.data() + ((size_t)(o / 16) * (m.I / 16) + i / 16) * 8 * m.K;
  uint32_t state = 0;
  for (int z = 0; z < 16; ++z) {
    const int bit = ((pos + 1) * m.K - 16 + z + bits) % bits;
    state = (state << 1) | ((t[bit / 32] >> (31 - bit % 32)) & 1);
  }
  const uint32_t p = state * kMul1;
  const int sum = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24);
  return __half2float(__float2half((float)((1024.0 + sum) * 0.00676727294921875 - 10.3828125)));
}
struct CpuOracle { std::vector<double> y, mag; };
inline CpuOracle FullOracle(const Matrix& m, const std::vector<__half>& x) {
  constexpr int rows = 17;
  CpuOracle out{std::vector<double>((size_t)rows * m.O),
                std::vector<double>((size_t)rows * m.O)};
  std::vector<double> xd((size_t)rows * m.I), w(m.I);
  for (int r = 0; r < rows; ++r) {
    const int source = r < 8 ? r : (r * 37) % 160;
    for (int i = 0; i < m.I; ++i) xd[(size_t)r * m.I + i] = __half2float(x[(size_t)source * m.I + i]);
  }
  for (int o = 0; o < m.O; ++o) {
    for (int i = 0; i < m.I; ++i) w[i] = Weight(m, i, o);
    for (int r = 0; r < rows; ++r) {
      double sum = 0, mag = 0;
      for (int i = 0; i < m.I; ++i) {
        const double v = xd[(size_t)r * m.I + i] * w[i];
        sum += v; mag += std::fabs(v);
      }
      out.y[(size_t)r * m.O + o] = sum;
      out.mag[(size_t)r * m.O + o] = mag;
    }
  }
  return out;
}
inline std::vector<__half> RandomX(size_t n) {
  std::mt19937 rng(17);
  std::normal_distribution<float> norm;
  std::vector<__half> x(n);
  for (auto& h : x) h = __float2half(norm(rng));
  return x;
}
inline bool SharedPackedLayout(int layout) { return layout >= 10 && layout <= 13; }
inline bool CompactLayout(int layout) { return layout >= 14 && layout <= 16; }
inline bool PackedLayout(int layout) { return SharedPackedLayout(layout) || CompactLayout(layout); }
struct PackedInputs {
  std::map<int, const __half*> by_I;
  std::vector<__half*> owned;
  double ms = 0;
  void Free() { for (auto* p : owned) HIP_CHECK(hipFree(p)); }
};
inline PackedInputs MakePackedInputs(const std::vector<Matrix>& ms, const __half* x,
                                    int rows, bool enabled) {
  PackedInputs p;
  if (!enabled) return p;
  Require(rows > 0 && x != nullptr, "invalid packed input pool");
  for (const auto& m : ms) {
    if (p.by_I.contains(m.I)) continue;
    const size_t bytes = (size_t)rows * m.I * 2;
    Require(bytes <= (1ull << 31), "packed input pool exceeds 2 GiB");
    __half* out; HIP_CHECK(hipMalloc(&out, bytes));
    p.by_I[m.I] = out; p.owned.push_back(out);
  }
  bench::EventTimer timer;
  timer.Begin();
  for (const auto& [I, out] : p.by_I)
    hipLaunchKernelGGL(PackX16Kernel, dim3(I / 16, (rows + 31) / 32), dim3(256),
                       0, nullptr, x, const_cast<__half*>(out), rows, I);
  HIP_CHECK(hipGetLastError());
  p.ms = timer.EndMs();
  return p;
}
inline void CheckPackedInputs(const PackedInputs& p, const std::vector<__half>& original, int rows) {
  for (const auto& [I, ptr] : p.by_I) {
    std::vector<uint16_t> got((size_t)rows * I);
    HIP_CHECK(hipMemcpy(got.data(), ptr, got.size() * 2, hipMemcpyDeviceToHost));
    for (int a = 0; a < I / 16; ++a)
      for (int r = 0; r < rows; ++r)
        for (int k = 0; k < 16; ++k) {
          uint16_t expected;
          std::memcpy(&expected, &original[(size_t)r * I + a * 16 + k], 2);
          Require(got[((size_t)a * rows + r) * 16 + k] == expected, "packed input changed an fp16 value");
        }
  }
}
struct Plan {
  std::vector<Exl3GemmJob> jobs;
  std::vector<Exl3GemmItem> items;
  Exl3GemmJob* dj = nullptr;
  Exl3GemmItem* di = nullptr;
  float* partial = nullptr;
  int* rows = nullptr;
  __half* inputs = nullptr;
  double input_pack_ms = 0;
  double flops = 0;
  size_t output = 0;
  void Free() {
    HIP_CHECK(hipFree(dj)); HIP_CHECK(hipFree(di)); HIP_CHECK(hipFree(partial));
    if (rows) HIP_CHECK(hipFree(rows));
    if (inputs) HIP_CHECK(hipFree(inputs));
  }
};
inline void CompactInputs(Plan& p) {
  size_t count = 0;
  for (const auto& j : p.jobs) count += (size_t)j.M * j.I;
  HIP_CHECK(hipMalloc(&p.inputs, std::max((size_t)4, count * 2)));
  std::vector<PackXJob> jobs;
  std::vector<PackXItem> items;
  size_t off = 0;
  for (size_t n = 0; n < p.jobs.size(); ++n) {
    auto& j = p.jobs[n];
    auto* dst = p.inputs + off;
    jobs.push_back({j.x, dst, j.row_ids, j.I, j.M});
    for (int a = 0; a < j.I / 16; ++a)
      for (int r = 0; r < j.M; r += 32) items.push_back({(int)n, a, r});
    off += (size_t)j.M * j.I;
    j.x = dst; j.Xrows = j.M; j.row_ids = nullptr;
  }
  if (items.empty()) return;
  PackXJob* dj; PackXItem* di;
  HIP_CHECK(hipMalloc(&dj, jobs.size() * sizeof(PackXJob)));
  HIP_CHECK(hipMalloc(&di, items.size() * sizeof(PackXItem)));
  HIP_CHECK(hipMemcpy(dj, jobs.data(), jobs.size() * sizeof(PackXJob), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(di, items.data(), items.size() * sizeof(PackXItem), hipMemcpyHostToDevice));
  bench::EventTimer timer; timer.Begin();
  hipLaunchKernelGGL(PackCompactX16Kernel, dim3(items.size()), dim3(256), 0, nullptr,
                     dj, di, (int)items.size());
  HIP_CHECK(hipGetLastError()); p.input_pack_ms = timer.EndMs();
  HIP_CHECK(hipFree(dj)); HIP_CHECK(hipFree(di));
}
inline Plan MakePlan(const std::vector<Matrix>& ms, int M, int RM, int S,
                     const __half* x, const std::vector<std::vector<int>>& lists = {}, int source_rows = 0,
                     int layout = 1, const std::map<int, const __half*>& packed = {}) {
  Require(RM == 1 || RM == 2 || RM == 4 || RM == 8 || RM == 16, "RM must be 1,2,4,8,16");
  Plan p;
  const int Xrows = source_rows ? source_rows : M;
  Require(M >= 0 && Xrows > 0 && !ms.empty(), "invalid row pool or empty matrix set");
  for (const auto& m : ms)
    Require((uint64_t)Xrows * m.I * 2 <= UINT32_MAX, "activation pool exceeds raw-buffer limit");
  Require(lists.empty() || lists.size() == ms.size(), "one row list required per job");
  for (const auto& list : lists)
    for (int row : list) Require(row >= 0 && row < Xrows, "gather row outside source pool");
  std::vector<int> ids;
  for (size_t n = 0; n < ms.size(); ++n) {
    const auto& m = ms[n];
    const int rows = lists.empty() ? M : lists[n].size();
    Require(S > 0 && S <= m.I / 16, "split count exceeds input tiles");
    Require((layout != 5 && layout != 7) || rows % (RM * 64) == 0,
            "full-row tiled layout needs complete M tiles");
    Require(!PackedLayout(layout) || (uint64_t)Xrows * m.I * 2 <= (1ull << 31), "packed input exceeds 2 GiB");
    Require(!SharedPackedLayout(layout) || packed.contains(m.I), "packed input pool is missing a matrix width");
    const auto* input = SharedPackedLayout(layout) ? packed.at(m.I) : x;
    p.jobs.push_back({m.dev, input, nullptr, nullptr, m.I, m.O, rows, S, m.K, Xrows});
    p.output += (size_t)S * rows * m.O;
    p.flops += 2.0 * rows * m.I * m.O;
    for (int s = 0; s < S; ++s)
      for (int r = 0; r < rows; r += RM * (layout >= 4 && layout <= 7 ? 64 : 16))
        for (int b = 0; b < m.O / 16; b += layout >= 4 && layout <= 7 ? 1 : 4)
          p.items.push_back({(int)n, s, r, b});
    if (!lists.empty()) ids.insert(ids.end(), lists[n].begin(), lists[n].end());
  }
  HIP_CHECK(hipMalloc(&p.partial, std::max((size_t)4, p.output * 4)));
  if (!ids.empty()) {
    HIP_CHECK(hipMalloc(&p.rows, ids.size() * 4));
    HIP_CHECK(hipMemcpy(p.rows, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
  }
  size_t off = 0, roff = 0;
  for (auto& j : p.jobs) {
    j.partial = p.partial + off;
    j.row_ids = p.rows ? p.rows + roff : nullptr;
    off += (size_t)S * j.M * j.O; roff += j.M;
  }
  if (CompactLayout(layout)) CompactInputs(p);
  HIP_CHECK(hipMalloc(&p.dj, p.jobs.size() * sizeof(Exl3GemmJob)));
  HIP_CHECK(hipMalloc(&p.di, std::max((size_t)1, p.items.size()) * sizeof(Exl3GemmItem)));
  HIP_CHECK(hipMemcpy(p.dj, p.jobs.data(), p.jobs.size() * sizeof(Exl3GemmJob), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(p.di, p.items.data(), p.items.size() * sizeof(Exl3GemmItem), hipMemcpyHostToDevice));
  return p;
}
using GemmFn = void (*)(const Exl3GemmJob*, const Exl3GemmItem*, int);
template <int RM>
inline GemmFn PickPf(int lds, int pf) {
#if (!defined(EXL3_LEGACY_ONLY) || defined(EXL3_PACKED_ONLY)) && !defined(EXL3_PIPE_ONLY)
  if (CompactLayout(lds)) {
    if constexpr (RM <= 4) {
#define PICK_COMPACT(P) case P: return lds == 16 ? Exl3GemmKernel<RM, true, false, P, true, true, true, true> : \
                                  lds == 15 ? Exl3GemmKernel<RM, true, false, P, true, true, true> : \
                                              Exl3GemmKernel<RM, true, false, P, false, true, true>
      switch (pf) { PICK_COMPACT(1); PICK_COMPACT(4); PICK_COMPACT(8); }
#undef PICK_COMPACT
    }
    Require(false, "compact mode needs RM=1,2,4 and PF=1,4,8"); return nullptr;
  }
#ifndef EXL3_COMPACT_ONLY
  if (SharedPackedLayout(lds)) {
#define PICK_PACK(P) case P: return lds == 11 ? Exl3GemmKernel<RM, true, false, P, true, true> : \
                                  lds == 12 ? Exl3GemmKernel<RM, true, true, P, false, true> : \
                                  lds == 13 ? Exl3GemmKernel<RM, false, false, P, false, true> : \
                                              Exl3GemmKernel<RM, true, false, P, false, true>
    switch (pf) { PICK_PACK(1); PICK_PACK(4); PICK_PACK(8); }
#undef PICK_PACK
    Require(false, "packed PF must be 1,4,8"); return nullptr;
  }
#endif
#endif
#ifndef EXL3_LEGACY_ONLY
  if (lds == 6 || lds == 7) {
    if constexpr (RM <= 4) {
#define PICK_PIPE(C) case C: return lds == 7 ? Exl3GemmPipeKernel<RM, C, true> : Exl3GemmPipeKernel<RM, C, false>
      switch (pf) {
        case 1: return lds == 7 ? Exl3GemmPipeKernel<RM, 4, true> : Exl3GemmPipeKernel<RM, 4, false>;
        PICK_PIPE(4); PICK_PIPE(8);
      }
#undef PICK_PIPE
    }
    Require(false, "pipeline needs RM=1,2,4 and CHUNK=4,8"); return nullptr;
  }
#ifndef EXL3_PIPE_ONLY
  if (lds == 5) {
    switch (pf) {
      case 4: return Exl3GemmTiledKernel<RM, 4, true>;
      case 1:
      case 8: return Exl3GemmTiledKernel<RM, 8, true>;
      case 16: return Exl3GemmTiledKernel<RM, 16, true>;
    }
    Require(false, "tiled CHUNK must be 4,8,16"); return nullptr;
  }
  if (lds == 4) {
    switch (pf) {
      case 4: return Exl3GemmTiledKernel<RM, 4>;
      case 1:
      case 8: return Exl3GemmTiledKernel<RM, 8>;
      case 16: return Exl3GemmTiledKernel<RM, 16>;
    }
    Require(false, "tiled CHUNK must be 4,8,16"); return nullptr;
  }
#endif
#endif
#if !defined(EXL3_PIPE_ONLY) && !defined(EXL3_PACKED_ONLY)
  Require(lds >= 0 && lds <= 3, "layout is not supported by this binary");
#define PICK_PF(P) case P: return lds == 3 ? Exl3GemmKernel<RM, true, false, P, true> : \
                                 lds == 2 ? Exl3GemmKernel<RM, true, true, P> : \
                                 lds == 1 ? Exl3GemmKernel<RM, true, false, P> : Exl3GemmKernel<RM, false, false, P>
  switch (pf) { PICK_PF(1); PICK_PF(4); PICK_PF(8); }
#undef PICK_PF
#endif
  Require(false, "unsupported layout or PF in this binary"); return nullptr;
}
inline void Launch(const Plan& p, int rm, int lds, int pf = 1) {
  Require((lds >= 0 && lds <= 7) || PackedLayout(lds), "LDS must be in [0,7] or [10,16]");
  if (p.items.empty()) return;
  GemmFn fn;
#define PICK(R) case R: fn = PickPf<R>(lds, pf); break
  switch (rm) { PICK(1); PICK(2); PICK(4); PICK(8); PICK(16);
    default: Require(false, "RM must be 1,2,4,8,16"); return; }
#undef PICK
  hipLaunchKernelGGL(fn, dim3(p.items.size()), dim3(128), 0, nullptr,
                     p.dj, p.di, (int)p.items.size());
  HIP_CHECK(hipGetLastError());
}
inline std::vector<float> Reduce(const Plan& p, int job, uint64_t* hash = nullptr) {
  const auto& j = p.jobs[job];
  std::vector<float> partial((size_t)j.S * j.M * j.O), y((size_t)j.M * j.O);
  HIP_CHECK(hipMemcpy(partial.data(), j.partial, partial.size() * 4, hipMemcpyDeviceToHost));
  if (hash) {
    uint64_t h = 14695981039346656037ull;
    for (float value : partial) {
      uint32_t bits; std::memcpy(&bits, &value, 4);
      h = (h ^ bits) * 1099511628211ull;
    }
    *hash = h;
  }
  for (int s = 0; s < j.S; ++s)
    for (size_t i = 0; i < y.size(); ++i) y[i] += partial[(size_t)s * y.size() + i];
  return y;
}
__global__ void FiniteKernel(const float* src, size_t n, unsigned* bad) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    if ((__float_as_uint(src[i]) & 0x7f800000u) == 0x7f800000u) {
      atomicExch(bad, 1u); break;
    }
  }
}
inline void ValidateFinite(const Plan& p) {
  unsigned* bad;
  HIP_CHECK(hipMalloc(&bad, 4)); HIP_CHECK(hipMemset(bad, 0, 4));
  hipLaunchKernelGGL(FiniteKernel, dim3(320), dim3(256), 0, nullptr, p.partial, p.output, bad);
  unsigned host;
  HIP_CHECK(hipMemcpy(&host, bad, 4, hipMemcpyDeviceToHost)); HIP_CHECK(hipFree(bad));
  Require(host == 0, "nonfinite kernel output; discard the timing result");
}
struct ColdCache {
  uint32_t* data;
  uint32_t* sum;
  size_t words;
  void Free() { HIP_CHECK(hipFree(data)); HIP_CHECK(hipFree(sum)); }
};
__global__ void EvictKernel(const uint32_t* src, uint32_t* dst, size_t words) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t sum = 0;
  for (size_t i = tid; i < words / 4; i += gridDim.x * blockDim.x) {
    const uint4 v = reinterpret_cast<const uint4*>(src)[i];
    sum ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  dst[tid] = sum;
}
inline ColdCache MakeColdCache() {
  ColdCache c{nullptr, nullptr, 64u * 1024 * 1024};
  HIP_CHECK(hipMalloc(&c.data, c.words * 4));
  HIP_CHECK(hipMalloc(&c.sum, 320 * 256 * 4));
  HIP_CHECK(hipMemset(c.data, 0x51, c.words * 4));
  return c;
}
inline void Flush(const ColdCache* c) {
  if (!c) return;
  hipLaunchKernelGGL(EvictKernel, dim3(320), dim3(256), 0, nullptr, c->data, c->sum, c->words);
  HIP_CHECK(hipGetLastError());
}
inline double Time(const Plan& p, int rm, int lds, int pf = 1, const ColdCache* cold = nullptr) {
  Flush(cold);
  Launch(p, rm, lds, pf);
  HIP_CHECK(hipDeviceSynchronize());
  bench::EventTimer t;
  std::vector<double> times;
  for (int rep = 0; rep < 3; ++rep) {
    Flush(cold);
    t.Begin(); Launch(p, rm, lds, pf); times.push_back(t.EndMs());
  }
  return bench::Median(times);
}
}  // namespace gm
