// Multi-row QSA indexer scores for verify windows: bitwise check and timing.
// Build: hipcc -O3 -std=c++20 --offload-arch=gfx1151 bench/decode_long/score_rows.hip -o build/score_rows
// Run:   score_rows CAPTURE_PREFIX   (qsa_L.{meta,idx.f32,pool.f32}, as CRUCIBLE_QSA_CAPTURE_DIR writes)
//        score_rows random N         (random data, context N)
//        score_rows sweep            (random data, pos0 2036..2060: windows that cross the 512-block
//                                     threshold, kSelBlocks, between their rows; bitwise only, no timing)
// Prints PASS only if QsaScoreRowsKernel<R> equals QsaScoreKernel row by row for every R.
#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "../../engine/layers/qsa_long.h"

using namespace crucible;

static void Require(bool ok, const char* msg) {
  if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}

template<class T> static std::vector<T> ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  Require(bool(f), "open capture file");
  const size_t bytes = (size_t)f.tellg();
  std::vector<T> v(bytes / sizeof(T));
  f.seekg(0);
  Require(bool(f.read(reinterpret_cast<char*>(v.data()), bytes)), "read capture");
  return v;
}

template<class T> static T* Upload(const std::vector<T>& v) {
  T* p; HIP_OK(hipMalloc(&p, v.size() * sizeof(T)));
  HIP_OK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}

__global__ void EvictKernel(const uint4* in, unsigned* out, size_t n) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned acc = 0;
  for (size_t i = t; i < n; i += gridDim.x * blockDim.x) { const uint4 x = in[i]; acc ^= x.x ^ x.y ^ x.z ^ x.w; }
  out[t] = acc;
}

// Median of 21 samples; cold evicts the MALL (64 MiB read) before each sample.
static double Time(const std::function<void()>& f, bool cold) {
  static uint4* ev = nullptr; static unsigned* sink = nullptr;
  constexpr size_t evb = 64u << 20;
  if (!ev) { HIP_OK(hipMalloc(&ev, evb)); HIP_OK(hipMemset(ev, 1, evb)); HIP_OK(hipMalloc(&sink, 80 * 256 * 4)); }
  hipEvent_t s, e; HIP_OK(hipEventCreate(&s)); HIP_OK(hipEventCreate(&e));
  f(); HIP_OK(hipDeviceSynchronize());
  std::vector<double> t;
  for (int i = 0; i < 21; ++i) {
    if (cold) EvictKernel<<<80, 256>>>(ev, sink, evb / 16);
    HIP_OK(hipEventRecord(s)); f(); HIP_OK(hipEventRecord(e)); HIP_OK(hipEventSynchronize(e));
    float ms; HIP_OK(hipEventElapsedTime(&ms, s, e)); t.push_back(ms * 1000.);
  }
  std::sort(t.begin(), t.end());
  return t[10];
}


// Compares QsaScoreRowsKernel<R> with QsaScoreKernel for R = 1..8 at one pos0. Scores that the
// kernels skip (blocks at or below kSelBlocks) stay at the NaN fill, so the skip logic is compared too.
static void CheckPos(int pos, const std::vector<float>& idx, const std::vector<float>& tile, bool timed) {
  const int ctx = pos + 8, sstride = ctx / 4 + 4;
  float* didx = Upload(idx);
  float* dpool = Upload(tile);
  const std::vector<float> fill((size_t)8 * sstride, std::bit_cast<float>(0x7fc00001u));
  float* s_old = Upload(fill);
  float* s_new = Upload(fill);
  for (int R = 1; R <= 8; ++R) {
    QsaArgs a{};
    a.idx = didx; a.c.ik_pool = dpool; a.sstride = sstride; a.pos0 = pos; a.R = R;
    const int nb = (pos + R) / 4, grid = (nb + 255) / 256;
    a.scores = s_old;
    HIP_OK(hipMemcpy(s_old, fill.data(), fill.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(s_new, fill.data(), fill.size() * 4, hipMemcpyHostToDevice));
    const auto old_launch = [&] { QsaScoreKernel<<<dim3(grid, R), 256>>>(a); };
    old_launch();
    QsaArgs b = a; b.scores = s_new;
    const auto new_launch = [&] { LaunchQsaScore(b); };
    new_launch();
    HIP_OK(hipDeviceSynchronize());
    std::vector<float> x((size_t)R * sstride), y(x.size());
    HIP_OK(hipMemcpy(x.data(), s_old, x.size() * 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(y.data(), s_new, y.size() * 4, hipMemcpyDeviceToHost));
    const bool equal = std::memcmp(x.data(), y.data(), x.size() * 4) == 0;
    if (timed) {
      const double tow = Time(old_launch, false), tnw = Time(new_launch, false);
      const double toc = Time(old_launch, true), tnc = Time(new_launch, true);
      std::printf("pos=%d R=%d nb=%d bitwise=%s warm old %.1f new %.1f us | cold old %.1f new %.1f us (%.2fx), "
                  "new cold pool read %.1f GB/s\n", pos, R, nb, equal ? "EQUAL" : "DIFFER", tow, tnw, toc, tnc,
                  toc / tnc, (double)nb * 512 / tnc / 1e3);
    } else {
      int written = 0;
      for (int r = 0; r < R; ++r)
        for (int c = 0; c < sstride; ++c) written += x[(size_t)r * sstride + c] == x[(size_t)r * sstride + c];
      std::printf("pos=%d R=%d nb=%d scored=%d bitwise=%s\n", pos, R, nb, written, equal ? "EQUAL" : "DIFFER");
    }
    std::fflush(stdout);
    Require(equal, "multi-row scores differ from the per-row kernel");
  }
  HIP_OK(hipFree(didx)); HIP_OK(hipFree(dpool)); HIP_OK(hipFree(s_old)); HIP_OK(hipFree(s_new));
}

int main(int argc, char** argv) {
  Require(argc == 2 || argc == 3, "usage: score_rows PREFIX | score_rows random N | score_rows sweep");
  std::mt19937 rng(7); std::normal_distribution<float> nd(0, 1);
  if (argc == 2 && std::string(argv[1]) == "sweep") {
    // One pool for all positions: block b holds the same data at every pos0, as in a real cache.
    const int lo = 2036, hi = 2060;
    std::vector<float> idx(8 * 640), tile(QsaPoolFloats(hi + 8));
    for (float& x : idx) x = nd(rng);
    for (float& x : tile) x = nd(rng);
    for (int pos = lo; pos <= hi; ++pos) CheckPos(pos, idx, tile, false);
    std::printf("PASS\n");
    return 0;
  }
  int pos = 0, rows = 8;
  std::vector<float> idx, tile;
  if (argc == 2) {
    std::ifstream m(std::string(argv[1]) + ".meta");
    Require(bool(m >> pos >> rows) && rows == 8, "capture metadata with 8 rows");
    idx = ReadFile<float>(std::string(argv[1]) + ".idx.f32");
    tile = ReadFile<float>(std::string(argv[1]) + ".pool.f32");
    Require(idx.size() == 8 * 640 && tile.size() == QsaPoolFloats(pos + 8), "capture sizes");
  } else {
    const int n = std::stoi(argv[2]);
    pos = n - 1;
    idx.resize(8 * 640); tile.resize(QsaPoolFloats(pos + 8));
    for (float& x : idx) x = nd(rng);
    for (float& x : tile) x = nd(rng);
  }
  CheckPos(pos, idx, tile, true);
  std::printf("PASS\n");
}
