// GDN lanes path with the v conv in the scan (LaunchGdnLanes, the engine path) against
// the older lanes path (LaunchGdnLanesRef), in the engine's 2048-row parts. The state, the
// history ring, the raw output y and the gated out-projection input must match
// bit for bit at every row count, pos0 and layout. Then ABAB event timing.
// Usage: gdn_vscan [--pairs N] [--layers L] [rows ...]
// A build with -DCRUCIBLE_GDN_VSCAN_MUTANT changes the conv tap order in the
// scan: it must FAIL. the vscan path also runs with the SGPR scan
// (GdnScanSgprKernel, the default) and with the lanes V scan (the reference).
#include "prefill_common.h"
#include "kernels/prefill_gdn_lanes.h"
#include <cstring>
#include <string>
using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;
namespace {
constexpr int kCore = 2048;  // engine workspace rows (kPrefillCoreRows)

struct Inputs {
  int R;
  Buffer<float> qkv, z, ab, conv, A_log, dt_bias, gamma, hist, hist0, state, state0;
  Buffer<__half> suh, xh;
  Buffer<float> q, k, v, y, g, beta;
  explicit Inputs(int rows)
      : R(rows), qkv(Random((size_t)rows * 10240, 1.f, 1)), z(Random((size_t)rows * 6144, 1.f, 2)),
        ab(Random((size_t)rows * 96, 1.f, 3)), conv(Random(10240 * 4, .5f, 4)), A_log(Random(48, .5f, 5)),
        dt_bias(Random(48, .5f, 6)), gamma(Random(128, .1f, 7)), hist(16 * 10240),
        hist0(Random(16 * 10240, 1.f, 8)), state(48 * 128 * 128), state0(Random(48 * 128 * 128, .01f, 9)),
        suh(Half(Random(6144, 1.f, 10))), xh((size_t)rows * 6144), q(48 * (size_t)kCore * 128),
        k(48 * (size_t)kCore * 128), v(48 * (size_t)kCore * 128), y(48 * (size_t)kCore * 128),
        g(48 * kCore), beta(48 * kCore) {}
  void Reset() {
    HIP_OK(hipMemcpy(state.p, state0.p, state.n * 4, hipMemcpyDeviceToDevice));
    HIP_OK(hipMemcpy(hist.p, hist0.p, hist.n * 4, hipMemcpyDeviceToDevice));
    HIP_OK(hipMemset(xh.p, 0xff, xh.n * 2));
    HIP_OK(hipMemset(y.p, 0, y.n * 4));
  }
  GdnPrefill Full(bool packed, int pos0) {
    GdnPrefill a{};
    a.input = {qkv.p, z.p, nullptr, nullptr, conv.p, A_log.p, dt_bias.p, gamma.p, hist.p, state.p, suh.p,
               xh.p, nullptr, pos0, R, 0, R};
    if (packed) a.input.pack_rows = R;
    a.ab_logits = ab.p;
    a.q = q.p; a.k = k.p; a.v = v.p; a.y = y.p; a.g = g.p; a.beta = beta.p;
    return a;
  }
};

// engine/runtime/prefill_engine.inc: 2048-row parts on one stream.
void Parts(GdnPrefill full, bool vscan, bool sgpr = false) {
  for (int o = 0; o < full.input.R; o += kCore) {
    const int n = std::min(kCore, full.input.R - o);
    GdnPrefill a = full;
    a.input.qkv += (size_t)o * 10240; a.input.z += (size_t)o * 6144;
    if (a.input.pack_rows) a.input.pack_off = o; else a.input.xh_out += (size_t)o * 6144;
    a.input.pos0 += o; a.input.R = a.input.commit = n;
    a.ab_logits += (size_t)o * 96; a.chunks = (n + 63) / 64;
    if (vscan) LaunchGdnLanes(a, nullptr, sgpr);
    else LaunchGdnLanesRef(a);
  }
}

struct Result { std::vector<float> state, hist, y; std::vector<__half> xh; };
Result Run(Inputs& in, bool packed, int pos0, bool vscan, bool sgpr = false) {
  in.Reset();
  Parts(in.Full(packed, pos0), vscan, sgpr);
  CheckLaunch();
  return {in.state.Get(), in.hist.Get(), in.y.Get(), in.xh.Get()};
}
template <class T> bool Same(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
}  // namespace

int main(int argc, char** argv) {
  int pairs = 7, layers = 4;
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--pairs" && i + 1 < argc) pairs = std::stoi(argv[++i]);
    else if (s == "--layers" && i + 1 < argc) layers = std::stoi(argv[++i]);
    else sizes.push_back(std::stoi(s));
  }
  if (sizes.empty()) sizes = {1, 2, 3, 17, 64, 65, 600, 2047, 2048, 2049, 4096};
  bool ok = true;
  for (int R : sizes) {
    if (R < 1 || R > 16384) return 2;
    Inputs in(R);
    bool same = true;
    for (int packed = 0; packed < 2; ++packed)
      for (int pos0 : {0, 2, 5, 100}) {
        const Result a = Run(in, packed, pos0, false), b = Run(in, packed, pos0, true),
                     c = Run(in, packed, pos0, true, true);
        same = same && Same(a.state, b.state) && Same(a.hist, b.hist) && Same(a.y, b.y) && Same(a.xh, b.xh);
        same = same && Same(a.state, c.state) && Same(a.hist, c.hist) && Same(a.y, c.y) && Same(a.xh, c.xh);
      }
    ok = ok && same;
    std::printf("R=%d bitwise=%s", R, same ? "EQUAL" : "DIFFER");
    if (R >= 2048) {  // ABAB; the state keeps evolving, which does not change the work
      const GdnPrefill full = in.Full(true, 0);
      std::vector<double> t0, t1, t2, s2, p0, p1, p2, s0, s1, g0, g1;
      const GdnPrefill part = [&] { GdnPrefill a = full; a.input.R = a.input.commit = kCore; a.chunks = kCore / 64; return a; }();
      for (int rep = 0; rep < pairs; ++rep) {
        t0.push_back(Time([&] { for (int l = 0; l < layers; ++l) Parts(full, false); }, 3) / layers);
        t1.push_back(Time([&] { for (int l = 0; l < layers; ++l) Parts(full, true); }, 3) / layers);
        t2.push_back(Time([&] { for (int l = 0; l < layers; ++l) Parts(full, true, true); }, 3) / layers);
        p0.push_back(Time([&] { GdnLanesPrepKernel<true><<<dim3(16 + 24, kCore), 256>>>(part); }, 3));
        p1.push_back(Time([&] { GdnLanesPrepKernel<false><<<dim3(16 + 1, kCore), 256>>>(part); }, 3));
        p2.push_back(Time([&] { GdnLanesPrepQkKernel<<<dim3(16 + 1, kCore / kGdnPrepRows), 256>>>(part); }, 3));
        s0.push_back(Time([&] { GdnScanLanesKernel<true><<<48 * 4, 128>>>(part); }, 3));
        s1.push_back(Time([&] { GdnScanLanesVKernel<<<48 * 4, 128>>>(part); }, 3));
        s2.push_back(Time([&] { GdnScanSgprKernel<<<48 * 4, 128>>>(part, part.k); }, 3));
        g0.push_back(Time([&] { GdnLanesGateKernel<<<kCore * 48 / 8, 256>>>(part); }, 3));
        g1.push_back(Time([&] { GdnLanesGateKernel<true><<<kCore / 8 * 48, 256>>>(part); }, 3));
      }
      const double a = bench::Median(t0), b = bench::Median(t1);
      std::printf(" base_us=%.1f vscan_us=%.1f ratio=%.3f ms_per4096_36L=%.1f | prep2048 %.1f->%.1f (rows4 %.1f) us scan2048 %.1f->%.1f us gate2048 %.1f->%.1f us",
                  a, b, b / a, (a - b) * 36 / 1000. * 4096. / R, bench::Median(p0), bench::Median(p1), bench::Median(p2),
                  bench::Median(s0), bench::Median(s1), bench::Median(g0), bench::Median(g1));
      const double c = bench::Median(t2);  // vscan with the SGPR scan against vscan
      std::printf(" | sgpr_us=%.1f sgpr/vscan=%.3f ms_per4096_36L=%.1f scan2048 sgpr %.1f us", c, c / b,
                  (b - c) * 36 / 1000. * 4096. / R, bench::Median(s2));
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
