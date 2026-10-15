// Prefill QSA top-512 selection: QsaTopPassKernel with 1024 threads (train 10)
// against a smaller block. sel and nsel must match bit for bit; then ABAB
// event timing of one part. Scores are random with many exact ties.
// "reg" is the engine's QsaTopRegKernel (qsa_long.h) against the tile pass
// and merge it replaces for prefill rows.
// Usage: qsa_top_threads [POS0 ...] (default 0 6144 30720 63488 129024), R = 2048
#include "prefill_common.h"
#include "../engine/layers/qsa_long.h"
#include <cstring>

using namespace crucible;
using namespace prefill_bench;

namespace {
constexpr int kR = 2048;

template <int T> void Launch(QsaArgs a, float* scratch) {
  const int nb = (a.pos0 + a.R) / 4;
  if (nb > kSelBlocks && nb <= kQsaCandidates) {
    QsaTopPassKernel<0, QsaArgs, T><<<a.R, T>>>(a, nullptr, nullptr);
  } else {
    int* ids = reinterpret_cast<int*>(scratch + (size_t)a.R * kQsaCandidates);
    QsaTopPassKernel<1, QsaArgs, T><<<dim3((nb + kQsaTopTile - 1) / kQsaTopTile, a.R), T>>>(a, scratch, ids);
    QsaTopPassKernel<2, QsaArgs, T><<<a.R, T>>>(a, scratch, ids);
  }
}

template <class T> bool Same(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}


template <int T, int PT> bool RunReg(QsaArgs a, Buffer<int>& sel, Buffer<int>& nsel, float* scratch,
                                     const std::vector<int>& s0, const std::vector<int>& n0, int nb) {
  if ((nb + T - 1) / T > PT) return true;
  HIP_OK(hipMemset(sel.p, 0x7f, sel.n * sizeof(int))); HIP_OK(hipMemset(nsel.p, 0x7f, nsel.n * sizeof(int)));
  QsaTopRegKernel<T, PT><<<a.R, T>>>(a); CheckLaunch();
  const bool same = Same(s0, sel.Get()) && Same(n0, nsel.Get());
  std::vector<double> tb, tt;
  for (int rep = 0; rep < 2; ++rep) {
    tb.push_back(Time([&] { Launch<1024>(a, scratch); }));
    tt.push_back(Time([&] { QsaTopRegKernel<T, PT><<<a.R, T>>>(a); }));
  }
  const double b = bench::Median(tb), t = bench::Median(tt);
  std::printf("pos0=%d nb=%d reg threads=%d pt=%d bitwise=%s base_us=%.1f us=%.1f ratio=%.3f\n", a.pos0, nb, T, PT,
              same ? "EQUAL" : "DIFFER", b, t, t / b);
  std::fflush(stdout);
  return same;
}

template <int T> bool Run(int pos0) {
  const int nb = (pos0 + kR) / 4, stride = nb;
  // Scores rounded to 1/64 so equal values (ties) are common, plus negatives
  // would not occur after relu: keep them nonnegative as the engine does.
  std::vector<float> host = Random((size_t)kR * stride, 1.f, 401 + pos0);
  for (auto& v : host) v = std::floor(std::fabs(v) * 64.f) / 64.f;
  Buffer<float> scores(host), scratch(QsaSelectScratchFloats(kR));
  Buffer<int> sel((size_t)kR * kSelBlocks), nsel(kR);
  QsaArgs a{};
  a.pos0 = pos0; a.R = kR; a.scores = scores.p; a.sstride = stride; a.sel = sel.p; a.nsel = nsel.p;
  sel.Zero(); nsel.Zero(); Launch<1024>(a, scratch.p); CheckLaunch();
  const auto s0 = sel.Get(), n0 = nsel.Get();
  HIP_OK(hipMemset(sel.p, 0x7f, sel.n * sizeof(int))); HIP_OK(hipMemset(nsel.p, 0x7f, nsel.n * sizeof(int)));
  scratch.Zero(); Launch<T>(a, scratch.p); CheckLaunch();
  const bool same = Same(s0, sel.Get()) && Same(n0, nsel.Get());
  std::vector<double> tb, tt;
  for (int rep = 0; rep < 2; ++rep) {
    tb.push_back(Time([&] { Launch<1024>(a, scratch.p); }));
    tt.push_back(Time([&] { Launch<T>(a, scratch.p); }));
  }
  const double b = bench::Median(tb), t = bench::Median(tt);
  std::printf("pos0=%d nb=%d threads=%d bitwise=%s base_us=%.1f us=%.1f ratio=%.3f\n", pos0, nb, T,
              same ? "EQUAL" : "DIFFER", b, t, t / b);
  std::fflush(stdout);
  bool reg = true;
  if (T == 256) {  // once per pos0
    reg = RunReg<1024, 32>(a, sel, nsel, scratch.p, s0, n0, nb) && reg;
    reg = RunReg<1024, 64>(a, sel, nsel, scratch.p, s0, n0, nb) && reg;
    reg = RunReg<512, 64>(a, sel, nsel, scratch.p, s0, n0, nb) && reg;
  }
  return same && reg;
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<int> pos;
  for (int i = 1; i < argc; ++i) pos.push_back(std::stoi(argv[i]));
  if (pos.empty()) pos = {0, 6144, 30720, 63488, 129024};
  bool ok = true;
  for (int p : pos) {
    if (p < 0 || p + kR > 262144 || p % 4) return 2;
    ok = Run<256>(p) && ok;
    ok = Run<512>(p) && ok;
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
