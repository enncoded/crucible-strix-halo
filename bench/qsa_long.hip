// Build: hipcc -O3 -std=c++20 --offload-arch=gfx1151 --rocm-path=/opt/rocm bench/qsa_long.hip -o build/qsa_long
// Run under the shared GPU lock. Numerical reference uses the stored fp16 KV.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "../engine/layers/qsa_long.h"
#include "kernels/qsa_decode.h"

using namespace crucible;

template<class T> struct Device {
  T* p;
  size_t n;
  explicit Device(size_t count) : n(count) { HIP_OK(hipMalloc(&p, n * sizeof(T))); }
  ~Device() { HIP_OK(hipFree(p)); }
  void put(const std::vector<T>& v) { HIP_OK(hipMemcpy(p, v.data(), v.size()*sizeof(T), hipMemcpyHostToDevice)); }
  std::vector<T> get(size_t count = 0) const {
    std::vector<T> v(count ? count : n);
    HIP_OK(hipMemcpy(v.data(), p, v.size()*sizeof(T), hipMemcpyDeviceToHost));
    return v;
  }
};

void Require(bool ok, const char* msg) {
  if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}

double Time(const std::function<void()>& f, int repeats = 10) {
  f();
  hipEvent_t start, stop;
  HIP_OK(hipEventCreate(&start)); HIP_OK(hipEventCreate(&stop));
  std::vector<double> samples;
  for (int trial = 0; trial < 5; ++trial) {
    HIP_OK(hipEventRecord(start));
    for (int i = 0; i < repeats; ++i) f();
    HIP_OK(hipEventRecord(stop)); HIP_OK(hipEventSynchronize(stop));
    float ms;
    HIP_OK(hipEventElapsedTime(&ms, start, stop));
    samples.push_back(ms*1000./repeats);
  }
  HIP_OK(hipEventDestroy(start)); HIP_OK(hipEventDestroy(stop));
  HIP_OK(hipGetLastError());
  std::sort(samples.begin(),samples.end());
  return samples[samples.size()/2];
}

__global__ void EvictKernel(const uint4* input, unsigned* output, size_t n) {
  const int t = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned acc = 0;
  for (size_t i = t; i < n; i += gridDim.x*blockDim.x) {
    const uint4 x = input[i]; acc ^= x.x ^ x.y ^ x.z ^ x.w;
  }
  output[t] = acc;
}

double ColdTime(const std::function<void()>& f) {
  constexpr size_t bytes = 64u<<20;
  Device<uint4> evict(bytes/16);
  Device<unsigned> sink(256*80);
  HIP_OK(hipMemset(evict.p,1,bytes));
  hipEvent_t start, stop;
  HIP_OK(hipEventCreate(&start)); HIP_OK(hipEventCreate(&stop));
  std::vector<double> samples;
  for (int i = 0; i < 20; ++i) {
    EvictKernel<<<80,256>>>(evict.p,sink.p,bytes/16);
    HIP_OK(hipEventRecord(start)); f(); HIP_OK(hipEventRecord(stop));
    HIP_OK(hipEventSynchronize(stop));
    float ms; HIP_OK(hipEventElapsedTime(&ms,start,stop)); samples.push_back(ms*1000.);
  }
  HIP_OK(hipEventDestroy(start)); HIP_OK(hipEventDestroy(stop));
  std::sort(samples.begin(),samples.end());
  return (samples[9]+samples[10])/2;
}

std::vector<int> ReferenceSelection(const float* iq, const std::vector<float>& pool, int n) {
  const int nb = n / 4;
  if (nb <= 512) return {};
  std::vector<float> score(nb);
  std::vector<int> order(nb);
  std::iota(order.begin(), order.end(), 0);
  for (int b = 0; b < nb; ++b) {
    float s = 0;
    for (int h = 0; h < 4; ++h) {
      float d = 0;
      for (int i = 0; i < 128; ++i) d = std::fma(iq[h*128+i], pool[b*128+i], d);
      s += std::max(d, 0.f);
    }
    score[b] = s * 0.08838834764831845f;
  }
  std::partial_sort(order.begin(), order.begin()+512, order.end(), [&](int x, int y) {
    return score[x] != score[y] ? score[x] > score[y] : x < y;
  });
  order.resize(512);
  std::sort(order.begin(), order.end());
  return order;
}

std::vector<int> Positions(const std::vector<int>& sel, int n) {
  std::vector<int> p;
  if (sel.empty()) { p.resize(n); std::iota(p.begin(), p.end(), 0); return p; }
  for (int b : sel) for (int t = 0; t < 4; ++t) p.push_back(b*4+t);
  for (int t = n/4*4; t < n; ++t) p.push_back(t);
  return p;
}

double AttentionError(const float* q, const std::vector<__half>& k, const std::vector<__half>& v,
                      const std::vector<int>& pos, const float* actual) {
  double error = 0, norm = 0;
  for (int h = 0; h < 24; ++h) {
    std::vector<double> logits(pos.size());
    for (size_t t = 0; t < pos.size(); ++t) {
      const size_t base = (size_t)pos[t]*512 + (h/12)*256;
      double d = 0;
      for (int i = 0; i < 256; ++i) d += (double)q[h*512+i] * __half2float(k[base+i]);
      logits[t] = d / 16;
    }
    const double mx = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    for (double& x : logits) { x = std::exp(x-mx); sum += x; }
    for (int i = 0; i < 256; ++i) {
      double o = 0;
      for (size_t t = 0; t < pos.size(); ++t)
        o += logits[t] * __half2float(v[(size_t)pos[t]*512 + (h/12)*256+i]);
      o /= sum;
      error += std::pow(o-actual[h*256+i], 2);
      norm += o*o;
    }
  }
  return std::sqrt(error / std::max(norm, 1e-100));
}

template<class T> std::vector<T> ReadFile(const std::string& path) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  Require(bool(f),"open capture file");
  const size_t bytes = (size_t)f.tellg();
  Require(bytes%sizeof(T) == 0,"capture size");
  std::vector<T> result(bytes/sizeof(T));
  f.seekg(0);
  Require(bool(f.read(reinterpret_cast<char*>(result.data()),bytes)),"read capture");
  return result;
}

void Run(int n, bool ties, int rows, const std::string& prefix = "", bool cold = false) {
  std::mt19937 rng(42);
  std::normal_distribution<float> normal(0, 1);
  const int ctx = n+8, nb = ctx/4;
  std::vector<float> q(rows*12288), iq(rows*640), pool(nb*128);
  std::vector<__half> k((size_t)ctx*512), v(k.size()), suh(6144, __float2half(1));
  if (prefix.empty()) {
    for (float& x : q) x = normal(rng);
    for (float& x : iq) x = ties ? 0.f : normal(rng);
    for (float& x : pool) x = normal(rng);
    for (__half& x : k) x = __float2half(normal(rng));
    for (__half& x : v) x = __float2half(normal(rng));
  }
  std::vector<int> captured;
  if (!prefix.empty()) {
    q = ReadFile<float>(prefix+".q.f32"); iq = ReadFile<float>(prefix+".idx.f32");
    k = ReadFile<__half>(prefix+".k.f16"); v = ReadFile<__half>(prefix+".v.f16");
    suh = ReadFile<__half>(prefix+".suh.f16");
    captured = ReadFile<int>(prefix+".sel.i32");
    const auto tile = ReadFile<float>(prefix+".pool.f32");
    Require(q.size() == (size_t)rows*12288 && iq.size() == (size_t)rows*640,"capture row count");
    Require(captured.size() == (size_t)rows*512 && suh.size() == 6144,"capture selection/scale size");
    Require(k.size() == (size_t)(n+rows-1)*512 && k.size() == v.size(),"capture KV size");
    Require(tile.size() == QsaPoolFloats(n+rows-1),"capture pool size");
    for (int b = 0; b < nb; ++b) for (int i = 0; i < 128; ++i)
      pool[b*128+i] = b < (n+rows-1)/4 ? tile[QsaPoolOffset(b,i)] : 0;
  }
  Device<float> dq(q.size()), di(iq.size()), dp(pool.size()), scores(rows*nb), context(rows*6144);
  Device<__half> dk(k.size()), dv(v.size()), ds(suh.size()), out(rows*6144);
  Device<int> sel(rows*512), nsel(rows);
  dq.put(q); di.put(iq); dp.put(pool); dk.put(k); dv.put(v); ds.put(suh);
  QsaArgs a{};
  a.q = dq.p; a.idx = di.p; a.c = {dk.p, dv.p, nullptr, dp.p};
  a.scores = scores.p; a.sstride = nb; a.sel = sel.p; a.nsel = nsel.p;
  a.suh_o = ds.p; a.xh_o = out.p; a.pos0 = n-1; a.R = rows; a.context = context.p;
  const double select_us = Time([&] { QsaSelectKernel<<<rows,1024>>>(a); }, 3);
  const double attn_us = Time([&] { QsaAttnKernel<<<dim3(24,rows),256>>>(a); });
  std::vector<float> tiled(QsaPoolFloats(ctx));
  for (int b = 0; b < nb; ++b) for (int i = 0; i < 128; ++i)
    tiled[QsaPoolOffset(b,i)] = pool[b*128+i];
  Device<float> dt(tiled.size()), scratch(QsaScratchFloats(rows));
  dt.put(tiled); a.c.ik_pool = dt.p;
  const double score_us = (n+rows-1)/4 <= 512 ? 0 : Time([&] { QsaScoreKernel<<<dim3((nb+255)/256,rows),256>>>(a); });
  const double top_us = Time([&] { LaunchQsaTop(a,scratch.p); });
  const double fast_select = Time([&] { LaunchQsaSelect(a,scratch.p); });
  const double split_us = Time([&] { QsaGqaSplitKernel<<<dim3(2,rows,kQsaSplits),384>>>(a,scratch.p); });
  const double merge_us = Time([&] { QsaMergeKernel<<<dim3(24,rows),256>>>(a,scratch.p); });
  const double fast_attn = Time([&] { LaunchQsaAttn(a,scratch.p); });
  const auto selected = sel.get(), counts = nsel.get();
  const auto actual = context.get();
  double max_error = 0;
  for (int r = 0; r < rows; ++r) {
    const auto ref = ReferenceSelection(iq.data()+r*640, pool, n+r);
    Require(counts[r] == (ref.empty() ? -1 : 512), "selection count");
    if (!ref.empty()) {
      Require(std::equal(ref.begin(),ref.end(),selected.begin()+r*512), "selection list");
      if (!captured.empty()) Require(std::equal(ref.begin(),ref.end(),captured.begin()+r*512), "captured engine selection");
    }
    const double error = AttentionError(q.data()+r*12288, k, v, Positions(ref,n+r), actual.data()+r*6144);
    max_error = std::max(max_error, error);
    Require(error <= 1e-5, "attention relative L2 error");
  }
  for (int batch = 1; batch <= rows; ++batch) {
    QsaArgs one = a;
    one.R = batch;
    LaunchQsaSelect(one,scratch.p); LaunchQsaAttn(one,scratch.p);
    const auto s = sel.get(batch*512);
    const auto c = context.get(batch*6144);
    for (int r = 0; r < batch; ++r)
      if ((n+r)/4 > 512)
        Require(std::equal(s.begin()+r*512,s.begin()+(r+1)*512,selected.begin()+r*512),"selection row invariance");
    Require(std::memcmp(c.data(),actual.data(),c.size()*sizeof(float)) == 0,"attention row invariance");
  }
  std::printf("n=%d R=%d ties=%d baseline_select=%.3f baseline_attn=%.3f score=%.3f top=%.3f fast_select=%.3f split=%.3f merge=%.3f fast_attn=%.3f us error=%.3g PASS\n",
              n, rows, ties, select_us, attn_us, score_us, top_us, fast_select, split_us, merge_us, fast_attn, max_error);
  std::fflush(stdout);
  if (cold) {
    QsaArgs base = a; base.c.ik_pool = dp.p;
    const double old_us = ColdTime([&] { QsaSelectKernel<<<rows,1024>>>(base); QsaAttnKernel<<<dim3(24,rows),256>>>(base); });
    const double new_us = ColdTime([&] { LaunchQsaSelect(a,scratch.p); LaunchQsaAttn(a,scratch.p); });
    const double score_cold = ColdTime([&] { QsaScoreKernel<<<dim3((nb+255)/256,rows),256>>>(a); });
    std::printf("COLD n=%d R=%d score=%.3f us pool_bw=%.1f GB/s baseline_qsa=%.3f us fast_qsa=%.3f us trunk12=%.3f->%.3f ms/token\n",
                n,rows,score_cold,(n/4)*512./score_cold/1000,old_us,new_us,old_us*12/1000,new_us*12/1000);
  }
}

void TestTop() {
  int cases = 0;
  for (int nb : {513,2048,8192,8193,8704,8705,16385,32768,32769,65535,65536}) {
    Device<float> scores(nb), scratch(QsaSelectScratchFloats(1));
    Device<int> sel(512), nsel(1);
    QsaArgs a{};
    a.scores = scores.p; a.sstride = nb; a.sel = sel.p; a.nsel = nsel.p;
    a.pos0 = nb*4-1; a.R = 1;
    for (int mode = 0; mode < 5; ++mode) {
      std::vector<float> values(nb);
      std::vector<int> order(nb);
      std::iota(order.begin(),order.end(),0);
      for (int b = 0; b < nb; ++b) {
        const unsigned p = (unsigned)((uint64_t)b*65521)%nb;
        if (mode == 1) values[b] = p < 511 ? 2.f : 1.f;
        if (mode == 2) values[b] = std::bit_cast<float>(0x3f800000u+p);
        if (mode == 3) values[b] = std::bit_cast<float>(p+1);
        if (mode == 4) values[b] = p%7 == 0 ? INFINITY : float(p%97);
      }
      std::partial_sort(order.begin(),order.begin()+512,order.end(),[&](int x,int y) {
        return values[x] != values[y] ? values[x] > values[y] : x < y;
      });
      order.resize(512); std::sort(order.begin(),order.end());
      scores.put(values);
      LaunchQsaTop(a,scratch.p);
      Require(sel.get() == order,"adversarial score-only top-k");
      ++cases;
    }
  }
  std::printf("ADVERSARIAL_TOP %d cases PASS\n",cases);
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--top-tests") { TestTop(); return 0; }
  if (argc == 3 && std::string(argv[1]) == "--data") {
    int pos, rows;
    std::ifstream f(std::string(argv[2])+".meta");
    Require(bool(f >> pos >> rows) && pos >= 0 && rows >= 1 && rows <= 8 &&
            (int64_t)pos+rows <= 262144,"capture metadata");
    Run(pos+1,false,rows,argv[2]);
    return 0;
  }
  const int n = argc > 1 ? std::stoi(argv[1]) : 8192;
  const int rows = argc > 2 ? std::stoi(argv[2]) : 1;
  Require(n > 0 && rows >= 1 && rows <= 8 && (int64_t)n+rows-1 <= 262144, "arguments");
  Run(n, false, rows,"",argc > 3 && std::string(argv[3]) == "cold");
  if (argc > 3 && std::string(argv[3]) != "cold") Run(n, true, rows);
}
