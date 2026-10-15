// Long-context QSA union flash on real engine selections (index-only captures).
// q, K, and V are random: the selections set the work and the memory pattern.
#include "prefill_common.h"
#include "../engine/layers/kernels/prefill_qsa_union.h"
#include <algorithm>
#include <fstream>
#include <cstring>
#include <random>
#include <string>

using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;

namespace {
template <class T> std::vector<T> Read(const std::string& path, size_t count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in || in.tellg() != (std::streamoff)(count * sizeof(T))) {
    std::fprintf(stderr, "wrong capture size: %s\n", path.c_str());
    std::exit(2);
  }
  std::vector<T> x(count);
  in.seekg(0);
  if (!in.read(reinterpret_cast<char*>(x.data()), count * sizeof(T))) std::exit(2);
  return x;
}

// Work counts of one union list: steps, live M-tile steps, and visible pairs.
struct Work { double steps = 0, live_tiles = 0, visible = 0, keys = 0; };

Work Count(const std::vector<int>& list, const std::vector<int>& count, int R) {
  Work w;
  for (size_t tile = 0; tile < count.size(); ++tile) {
    const int n = count[tile] * 4, rows = std::min(kUnionRows, R - (int)tile * kUnionRows);
    w.keys += n;
    for (int start = 0; start < n; start += kUnionKeys) {
      int sm = 0;
      for (int i = start; i < std::min(n, start + kUnionKeys); ++i) {
        const int m = list[tile * kUnionCap + i / 4] >> 24 & 0xff;
        sm |= m;
        for (int r = 0; r < rows; ++r) w.visible += m >> r & 1;
      }
      w.steps += 1;
      w.live_tiles += (sm & 3 ? 1 : 0) + (sm & 6 ? 1 : 0) + (sm & 12 ? 1 : 0);
    }
  }
  return w;
}

// Random sorted selections of 512 completed blocks; rows with at most 512
// completed blocks are dense (count -1), as in the engine.
void Synthetic(int pos0, int R, std::vector<int>& sel, std::vector<int>& nsel) {
  std::mt19937 rng(41 + pos0);
  sel.assign((size_t)R * 512, 0);
  nsel.assign(R, -1);
  std::vector<int> all;
  for (int r = 0; r < R; ++r) {
    const int nb = (pos0 + r + 1) / 4;
    if (nb <= 512) continue;
    all.resize(nb);
    for (int b = 0; b < nb; ++b) all[b] = b;
    for (int i = 0; i < 512; ++i) std::swap(all[i], all[i + rng() % (nb - i)]);
    std::sort(all.begin(), all.begin() + 512);
    std::copy(all.begin(), all.begin() + 512, sel.begin() + (size_t)r * 512);
    nsel[r] = 512;
  }
}

void Run(const std::string& capture, int local_blocks) {
  int pos0 = 0, R = 0;
  std::vector<int> hsel, hnsel;
  if (capture.rfind("synthetic:", 0) == 0) {
    pos0 = std::stoi(capture.substr(10)); R = 2048;
    if (pos0 < 0 || pos0 > 262144 - R) std::exit(2);
    Synthetic(pos0, R, hsel, hnsel);
  } else {
    std::ifstream meta(capture + ".meta");
    if (!(meta >> pos0 >> R) || pos0 < 0 || R < 1 || R > 2048) std::exit(2);
    hsel = Read<int>(capture + ".sel.i32", (size_t)R * 512);
    hnsel = Read<int>(capture + ".nsel.i32", R);
  }
  const int ctx = pos0 + R;
  Buffer<int> sel(hsel), nsel(hnsel);
  Buffer<__half> ck(Half(Random((size_t)ctx * 512, 1.f, 35)));
  Buffer<__half> cv(Half(Random((size_t)ctx * 512, .4f, 36)));
  Buffer<__half> qhi(Half(Random((size_t)R * 6144, .4f, 31))), qlo(Half(Random((size_t)R * 6144, .4f, 32)));
  Buffer<float> context((size_t)R * 6144);
  Buffer<float> gates(Random((size_t)R * 24 * 512, 1.f, 33));
  std::vector<float> suh_values = Random(6144, .3f, 34);
  for (auto& v : suh_values) v += 1.f;
  Buffer<__half> suh(Half(suh_values)), xh((size_t)R * 6144);
  QsaArgs a{};
  a.c.k = ck.p; a.c.v = cv.p; a.sel = sel.p; a.nsel = nsel.p; a.pos0 = pos0; a.R = R;
  a.q = gates.p; a.suh_o = suh.p; a.xh_o = xh.p;
  QsaPrefill f{a, context.p, qhi.p, nullptr};
  Buffer<int> ulist(QsaUnionListInts(R)), ucount((R + kUnionRows - 1) / kUnionRows);
  QsaUnion u{ulist.p, ucount.p};
  const int tiles = (R + kUnionRows - 1) / kUnionRows, blocks = (pos0 + R - 1) / 4 + 1;
  auto build = [&] { QsaUnionBuildKernel<<<tiles, 256, blocks>>>(a, u); };
  build(); CheckLaunch();
  const auto list = ulist.Get(), counts = ucount.Get();
  const Work w = Count(list, counts, R);
  auto nolo = [&] { QsaUnionFlashKernel<false><<<dim3(tiles, 2), 256>>>(f, u); };
  QsaPrefill fl = f; fl.q_lo = qlo.p;
  auto lo = [&] { QsaUnionFlashKernel<true><<<dim3(tiles, 2), 256>>>(fl, u); };
  // Bitwise check and ABBA timing of the half-q variant against production.
  context.Zero(); nolo(); CheckLaunch();
  const auto c0 = context.Get();
  auto halfq = [&] { QsaUnionHalfQKernel<false><<<dim3(2, tiles), 256>>>(f, u); };
  context.Set(std::vector<float>(c0.size(), 7.f)); halfq(); CheckLaunch();
  const auto c1 = context.Get();
  size_t diff = 0;
  for (size_t i = 0; i < c0.size(); ++i)
    diff += std::memcmp(&c0[i], &c1[i], sizeof(float)) != 0;
  const double a0 = Time(nolo), b0 = Time(halfq), b1 = Time(halfq), a1 = Time(nolo);
  std::printf("halfq source=%s pos0=%d rows=%d prod_us=%.1f halfq_us=%.1f ratio=%.3f bitwise=%d diff=%zu\n",
      capture.c_str(), pos0, R, (a0 + a1) / 2, (b0 + b1) / 2, (b0 + b1) / (a0 + a1), diff == 0, diff);
  std::fflush(stdout);
  if (diff) std::exit(9);
  // Fused gate epilogue against flash followed by QsaPrefillGateKernel.
  auto separate = [&] { nolo(); QsaPrefillGateKernel<<<dim3(24, R), 256>>>(f); };
  auto fused = [&] { QsaUnionHalfQKernel<true><<<dim3(2, tiles), 256>>>(f, u); };
  xh.Zero(); separate(); CheckLaunch();
  const auto x0 = xh.Get();
  HIP_OK(hipMemset(xh.p, 0x7f, xh.n * sizeof(__half)));
  fused(); CheckLaunch();
  const auto x1 = xh.Get();
  const bool gate_same = std::memcmp(x0.data(), x1.data(), x0.size() * sizeof(__half)) == 0;
  const double s0 = Time(separate), g0 = Time(fused), g1 = Time(fused), s1 = Time(separate);
  const double gate_only = Time([&] { QsaPrefillGateKernel<<<dim3(24, R), 256>>>(f); });
  std::printf("gate source=%s pos0=%d rows=%d separate_us=%.1f fused_us=%.1f ratio=%.3f gate_kernel_us=%.1f bitwise=%d\n",
      capture.c_str(), pos0, R, (s0 + s1) / 2, (g0 + g1) / 2, (g0 + g1) / (s0 + s1), gate_only, (int)gate_same);
  std::fflush(stdout);
  if (!gate_same) std::exit(10);
  const double build_us = Time(build), nolo_us = Time(nolo), lo_us = Time(lo);
  // Same steps and masks, but every block maps into the first local_blocks
  // blocks, so K and V stay in cache. Outputs are not compared.
  std::vector<int> local = list;
  for (auto& e : local) e = (e & ~0xffffff) | ((e & 0xffffff) % local_blocks);
  ulist.Set(local);
  const double local_us = Time(nolo);
  ulist.Set(list);
  const double rows_keys = (double)R * 4 * 512;  // visible keys if no tail
  std::printf("flash source=%s pos0=%d rows=%d union_blocks_per_tile=%.1f steps_per_tile=%.1f "
      "live_mtiles_per_step=%.2f visible_frac=%.3f build_us=%.1f nolo_us=%.1f lo_us=%.1f local%d_us=%.1f "
      "kv_gbps=%.1f\n",
      capture.c_str(), pos0, R, w.keys / 4 / tiles, w.steps / tiles, w.live_tiles / w.steps,
      w.visible / (w.live_tiles * 16 / 12.0 * kUnionKeys), build_us, nolo_us, lo_us, local_blocks, local_us,
      w.keys * 2 * 1024 / nolo_us / 1e3);
  (void)rows_keys;
  std::fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0) return 2;
  if (argc != 3) {
    std::fprintf(stderr, "usage: qsa_flash_long CAPTURE_PREFIX LOCAL_BLOCKS\n");
    return 2;
  }
  const int local = std::stoi(argv[2]);
  if (local < 512 || local > 65536) return 2;
  Run(argv[1], local);
}
