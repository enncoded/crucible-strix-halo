// Go/no-go harness for the EXL3 matvec kernel on real Flash-Next tensors.
//
//   exl3_gemv decode CKPT PREFIX OUT.bin   GPU-decode one matrix to fp16 [I][O]
//                                          (all multiply modes must agree)
//   exl3_gemv check CKPT                   numerics vs CPU, exactness across R
//   exl3_gemv bench CKPT [span]            bandwidth on a decode token's routed
//                                          experts (1,440 matrices) and dense set
#include <cmath>
#include <random>

#include "kernels/exl3_gemv.h"
#include "common.h"
#include "../engine/exl3/safetensors.h"

using namespace crucible;

struct Matrix {
  std::string name;
  int A = 0, B = 0, K = 0;
  std::vector<uint32_t> host;  // crucible order
  uint32_t* dev = nullptr;
  size_t bytes() const { return host.size() * 4; }
};

static Matrix Load(st::Checkpoint& ck, const std::string& prefix) {
  Matrix m;
  m.name = prefix;
  const st::TensorInfo& ti = ck.Info(prefix + ".trellis");
  if (ti.dtype != "I16" || ti.shape.size() != 3) {
    std::fprintf(stderr, "%s: unexpected trellis %s\n", prefix.c_str(), ti.dtype.c_str());
    std::exit(1);
  }
  m.A = (int)ti.shape[0];
  m.B = (int)ti.shape[1];
  m.K = (int)ti.shape[2] / 16;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(ck.Data(prefix + ".trellis"));
  const int T = 8 * m.K;
  m.host.resize((size_t)m.A * m.B * T);
  for (int a = 0; a < m.A; ++a)
    for (int b = 0; b < m.B; ++b)
      std::memcpy(&m.host[((size_t)b * m.A + a) * T], &src[((size_t)a * m.B + b) * T], T * 4);
  return m;
}

static void Upload(std::vector<Matrix>& ms) {
  size_t total = 0;
  for (auto& m : ms) total += m.bytes();
  uint8_t* base;
  HIP_CHECK(hipMalloc(&base, total));
  size_t off = 0;
  for (auto& m : ms) {
    m.dev = reinterpret_cast<uint32_t*>(base + off);
    HIP_CHECK(hipMemcpy(m.dev, m.host.data(), m.bytes(), hipMemcpyHostToDevice));
    off += m.bytes();
  }
}

// ---- kernel dispatch -----------------------------------------------------

using GemvFn = void (*)(const Exl3Job*, const Exl3Item*, int);

template <int K, int R, int MODE>
static GemvFn PickPf(int pf) {
  switch (pf) {
    case 2: return Exl3GemvKernel<K, R, MODE, 2>;
    case 4: return Exl3GemvKernel<K, R, MODE, 4>;
    case 8: return Exl3GemvKernel<K, R, MODE, 8>;
    default: return Exl3GemvKernel<K, R, MODE, 16>;
  }
}
template <int K, int R>
static GemvFn PickMode(int mode, int pf) {
  switch (mode) {
    case 0: return PickPf<K, R, 0>(pf);
    case 1: return PickPf<K, R, 1>(pf);
    default: return PickPf<K, R, 2>(pf);
  }
}
template <int K>
static GemvFn PickR(int r, int mode, int pf) {
  switch (r) {
    case 1: return PickMode<K, 1>(mode, pf);
    case 2: return PickMode<K, 2>(mode, pf);
    case 3: return PickMode<K, 3>(mode, pf);
    case 4: return PickMode<K, 4>(mode, pf);
    default: return PickMode<K, 8>(mode, pf);
  }
}
static GemvFn Pick(int k, int r, int mode, int pf) {
  if (k == 3) return PickR<3>(r, mode, pf);
  if (k == 4) return PickR<4>(r, mode, pf);
  if (k == 5) return PickR<5>(r, mode, pf);
  if (k == 6) return PickR<6>(r, mode, pf);
  std::fprintf(stderr, "unsupported K=%d\n", k);
  std::exit(1);
}

// Builds the job and item lists for a set of same-K matrices.
struct Plan {
  std::vector<Exl3Job> jobs;
  std::vector<Exl3Item> items;
  Exl3Job* d_jobs = nullptr;
  Exl3Item* d_items = nullptr;
  size_t weight_bytes = 0;
};

static int PickS(int A, int span) {
  int best = 1;
  for (int s = 1; s <= A; ++s)
    if (A % s == 0 && std::abs(A / s - span) < std::abs(A / best - span)) best = s;
  return best;
}

static Plan MakePlan(const std::vector<Matrix*>& ms, int R, int span, __half* x_pool,
                     float* partial_pool) {
  Plan p;
  size_t poff = 0;
  for (size_t j = 0; j < ms.size(); ++j) {
    const Matrix& m = *ms[j];
    Exl3Job job;
    job.tiles = m.dev;
    job.x = x_pool;  // all jobs read the same activation rows (bandwidth test)
    job.A = m.A;
    job.B = m.B;
    job.S = PickS(m.A, span);
    job.partial = partial_pool + poff;
    poff += (size_t)job.S * R * m.B * 16;
    p.jobs.push_back(job);
    for (int s = 0; s < job.S; ++s)
      for (int b0 = 0; b0 < m.B; b0 += 8)
        p.items.push_back({(int)j, s, b0, std::min(8, m.B - b0)});
    p.weight_bytes += m.bytes();
  }
  HIP_CHECK(hipMalloc(&p.d_jobs, p.jobs.size() * sizeof(Exl3Job)));
  HIP_CHECK(hipMalloc(&p.d_items, p.items.size() * sizeof(Exl3Item)));
  HIP_CHECK(hipMemcpy(p.d_jobs, p.jobs.data(), p.jobs.size() * sizeof(Exl3Job),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(p.d_items, p.items.data(), p.items.size() * sizeof(Exl3Item),
                      hipMemcpyHostToDevice));
  return p;
}

static void Launch(const Plan& p, int K, int R, int mode, int pf, int grid = 0) {
  int n = (int)p.items.size();
  int span = 0;
  for (const auto& j : p.jobs) span = std::max(span, j.A / j.S);
  hipLaunchKernelGGL(Pick(K, R, mode, pf), dim3(grid > 0 ? std::min(grid, n) : n), dim3(256),
                     R * span * 32, nullptr, p.d_jobs, p.d_items, n);
}

// ---- helpers ---------------------------------------------------------------

static std::vector<__half> RandomHalves(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<__half> v(n);
  for (auto& h : v) h = __float2half(nd(rng));
  return v;
}

static std::vector<__half> DecodeOnGpu(const Matrix& m, int mode) {
  __half* w;
  size_t n = (size_t)m.A * 16 * m.B * 16;
  HIP_CHECK(hipMalloc(&w, n * 2));
  int tiles = m.A * m.B;
  auto launch = [&](auto kern) {
    hipLaunchKernelGGL(kern, dim3((tiles + 7) / 8), dim3(256), 0, nullptr, m.dev, m.A, m.B, w);
  };
  if (m.K == 3) {
    if (mode == 0) launch(Exl3DecodeKernel<3, 0>);
    else if (mode == 1) launch(Exl3DecodeKernel<3, 1>);
    else launch(Exl3DecodeKernel<3, 2>);
  } else if (m.K == 4) {
    if (mode == 0) launch(Exl3DecodeKernel<4, 0>);
    else if (mode == 1) launch(Exl3DecodeKernel<4, 1>);
    else launch(Exl3DecodeKernel<4, 2>);
  } else if (m.K == 5) {
    if (mode == 0) launch(Exl3DecodeKernel<5, 0>);
    else if (mode == 1) launch(Exl3DecodeKernel<5, 1>);
    else launch(Exl3DecodeKernel<5, 2>);
  } else if (m.K == 6) {
    if (mode == 0) launch(Exl3DecodeKernel<6, 0>);
    else if (mode == 1) launch(Exl3DecodeKernel<6, 1>);
    else launch(Exl3DecodeKernel<6, 2>);
  } else std::abort();
  std::vector<__half> h(n);
  HIP_CHECK(hipMemcpy(h.data(), w, n * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(w));
  return h;
}

static int Decode(const std::string& ckpt, const std::string& prefix, const std::string& out) {
  st::Checkpoint ck(ckpt);
  std::vector<Matrix> ms{Load(ck, prefix)};
  Upload(ms);
  auto w0 = DecodeOnGpu(ms[0], 0), w1 = DecodeOnGpu(ms[0], 1), w2 = DecodeOnGpu(ms[0], 2);
  bool same = std::memcmp(w0.data(), w1.data(), w0.size() * 2) == 0 &&
              std::memcmp(w0.data(), w2.data(), w0.size() * 2) == 0;
  std::printf("%s: I=%d O=%d K=%d; multiply modes agree bitwise: %s\n", prefix.c_str(),
              ms[0].A * 16, ms[0].B * 16, ms[0].K, same ? "yes" : "NO");
  FILE* f = std::fopen(out.c_str(), "wb");
  std::fwrite(w0.data(), 2, w0.size(), f);
  std::fclose(f);
  return same ? 0 : 1;
}

// Sums partials [S][R][O] in split order (the fixed reduction order).
static std::vector<float> Reduce(const Plan& p, int R) {
  const Exl3Job& j = p.jobs[0];
  size_t n = (size_t)j.S * R * j.B * 16;
  std::vector<float> h(n);
  HIP_CHECK(hipMemcpy(h.data(), j.partial, n * 4, hipMemcpyDeviceToHost));
  std::vector<float> y((size_t)R * j.B * 16, 0.f);
  for (int s = 0; s < j.S; ++s)
    for (size_t i = 0; i < y.size(); ++i) y[i] += h[(size_t)s * y.size() + i];
  return y;
}

static int Check(const std::string& ckpt) {
  st::Checkpoint ck(ckpt);
  std::vector<Matrix> ms{Load(ck, "model.language_model.layers.0.linear_attn.in_proj_qkv"),
                         Load(ck, "model.language_model.layers.0.mlp.experts.0.down_proj")};
  Upload(ms);
  int fails = 0;
  for (auto& m : ms) {
    const int I = m.A * 16, O = m.B * 16;
    auto w = DecodeOnGpu(m, 1);
    auto xh = RandomHalves((size_t)8 * I, 7);
    __half* x;
    float* part;
    HIP_CHECK(hipMalloc(&x, xh.size() * 2));
    HIP_CHECK(hipMemcpy(x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&part, (size_t)m.A * 8 * O * 4));
    std::vector<Matrix*> one{&m};
    for (int mode : {0, 1, 2}) {
      std::vector<std::vector<float>> ys;
      for (int R : {1, 2, 4, 8}) {
        Plan p = MakePlan(one, R, 20, x, part);
        Launch(p, m.K, R, mode, 4);
        HIP_CHECK(hipDeviceSynchronize());
        ys.push_back(Reduce(p, R));
      }
      // Exactness: row r of every R must match bitwise.
      bool exact = true;
      for (int k = 1; k < 4; ++k)
        for (int r = 0; r < (1 << k) && r < 1; ++r)
          exact &= std::memcmp(&ys[k][(size_t)r * O], &ys[0][0], O * 4) == 0;
      for (int r = 0; r < 4; ++r)
        exact &= std::memcmp(&ys[3][(size_t)r * O], &ys[2][(size_t)r * O], O * 4) == 0;
      // Numerics: R=8 against a double-precision CPU product of the decoded W.
      double worst = 0;
      for (int r = 0; r < 8; ++r)
        for (int o = 0; o < O; ++o) {
          double ref = 0, mag = 0;
          for (int i = 0; i < I; ++i) {
            double prod = (double)__half2float(xh[(size_t)r * I + i]) *
                          (double)__half2float(w[(size_t)i * O + o]);
            ref += prod;
            mag += std::fabs(prod);
          }
          double err = std::fabs(ys[3][(size_t)r * O + o] - ref) / (mag + 1e-30);
          worst = std::max(worst, err);
        }
      bool ok = exact && worst < 1e-5;
      fails += !ok;
      std::printf("%-58s K=%d mode=%d exact-across-R=%s max|err|/sum|xw|=%.2e %s\n",
                  m.name.c_str(), m.K, mode, exact ? "yes" : "NO", worst, ok ? "OK" : "FAIL");
    }
    HIP_CHECK(hipFree(x));
    HIP_CHECK(hipFree(part));
  }
  return fails ? 1 : 0;
}

static int Bench(const std::string& ckpt, int span) {
  st::Checkpoint ck(ckpt);
  std::map<std::string, std::vector<Matrix>> sets;
  for (int l = 0; l < 48; ++l) {
    const std::string L = "model.language_model.layers." + std::to_string(l);
    for (int e = 0; e < 10; ++e)
      for (const char* p : {"gate_proj", "up_proj", "down_proj"})
        sets["experts"].push_back(Load(ck, L + ".mlp.experts." + std::to_string(e) + "." + p));
    if ((l + 1) % 4 != 0) {
      for (const char* p : {"in_proj_qkv", "in_proj_z", "out_proj"})
        sets["attention"].push_back(Load(ck, L + ".linear_attn." + p));
    } else {
      for (const char* p : {"q_proj", "k_proj", "v_proj", "o_proj"})
        sets["attention"].push_back(Load(ck, L + ".self_attn." + p));
      sets["indexer"].push_back(Load(ck, L + ".self_attn.indexer.index_qk_proj"));
    }
    for (const char* p : {"gate_proj", "up_proj", "down_proj"})
      sets["shared"].push_back(Load(ck, L + ".mlp.shared_expert." + p));
  }
  sets["head"].push_back(Load(ck, "lm_head"));
  for (auto& [name, matrices] : sets) Upload(matrices);
  __half* x;
  float* part;
  auto xh = RandomHalves((size_t)8 * 10240, 3);
  HIP_CHECK(hipMalloc(&x, xh.size() * 2));
  HIP_CHECK(hipMemcpy(x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&part, (size_t)1 << 30));
  std::printf("# one launch per set; GB/s = packed trellis bytes / kernel time, median of 7\n");
  std::printf("%-10s %2s %2s %4s %2s %4s %8s %8s %8s\n", "set", "K", "R", "mode", "pf", "span", "MB", "ms", "GBps");
  for (auto& [name, matrices] : sets) {
    const int K = matrices[0].K;
    std::vector<Matrix*> set;
    for (auto& m : matrices) {
      if (m.K != K) std::abort();
      set.push_back(&m);
    }
    for (int R : {1, 3, 8})
      for (int mode : {0, 1})
        for (int pf : {8, 16}) {
          Plan p = MakePlan(set, R, span, x, part);
          bench::EventTimer t;
          std::vector<double> ms;
          for (int i = 0; i < 8; ++i) {
            t.Begin();
            Launch(p, K, R, mode, pf);
            double v = t.EndMs();
            if (i) ms.push_back(v);
          }
          double med = bench::Median(ms);
          std::printf("%-10s %2d %2d %4d %2d %4d %8.1f %8.3f %8.1f\n", name.c_str(), K, R, mode, pf, span,
                      p.weight_bytes / 1e6, med, p.weight_bytes / med / 1e6);
          HIP_CHECK(hipFree(p.d_jobs));
          HIP_CHECK(hipFree(p.d_items));
        }
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: exl3_gemv decode|check|bench CKPT ...\n");
    return 2;
  }
  std::string mode = argv[1];
  if (mode == "decode" && argc >= 5) return Decode(argv[2], argv[3], argv[4]);
  if (mode == "check") return Check(argv[2]);
  if (mode == "bench") return Bench(argv[2], argc > 3 ? std::atoi(argv[3]) : 20);
  return 2;
}
