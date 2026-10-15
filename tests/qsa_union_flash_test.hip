// The default QSA union flash (LaunchQsaUnionFlash, LaunchQsaUnionFlashGate)
// must keep the bits of the legacy path: QsaUnionFlashKernel<false> for the
// context, then QsaPrefillGateKernel for xh_o. Synthetic selections, dense and
// sparse rows, partial tiles, absent keys past the call end.
// Build: hipcc -O3 -std=c++20 --offload-arch=gfx1151 -o build/qsa_union_flash_test tests/qsa_union_flash_test.hip
#include "../bench/prefill_common.h"
#include "../engine/layers/kernels/prefill_qsa_union.h"
#include <algorithm>
#include <cstring>

using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;

namespace {
void Selections(int pos0, int R, std::vector<int>& sel, std::vector<int>& nsel) {
  std::mt19937 rng(7 + pos0 + R);
  sel.assign((size_t)R * kSelBlocks, 0);
  nsel.assign(R, -1);
  std::vector<int> all;
  for (int r = 0; r < R; ++r) {
    const int nb = (pos0 + r + 1) / 4;
    if (nb <= kSelBlocks) continue;
    all.resize(nb);
    for (int b = 0; b < nb; ++b) all[b] = b;
    // Neighbouring rows share most blocks, as in real selections.
    std::mt19937 row_rng(rng() % 3 == 0 ? 11 + r : 11 + r / 8 * 8);
    for (int i = 0; i < kSelBlocks; ++i) std::swap(all[i], all[i + row_rng() % (nb - i)]);
    std::sort(all.begin(), all.begin() + kSelBlocks);
    std::copy(all.begin(), all.begin() + kSelBlocks, sel.begin() + (size_t)r * kSelBlocks);
    nsel[r] = kSelBlocks;
  }
}

template <class T> bool Same(const std::vector<T>& x, const std::vector<T>& y) {
  return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}

void Check(int pos0, int R) {
  std::vector<int> hsel, hnsel;
  Selections(pos0, R, hsel, hnsel);
  // The cache holds 3 positions past the call: their K and V must stay unused.
  const int ctx = pos0 + R + 3;
  Buffer<int> sel(hsel), nsel(hnsel);
  Buffer<__half> ck(Half(Random((size_t)ctx * 512, 1.f, 35)));
  Buffer<__half> cv(Half(Random((size_t)ctx * 512, .4f, 36)));
  Buffer<__half> qhi(Half(Random((size_t)R * 6144, .4f, 31)));
  Buffer<float> context((size_t)R * 6144), gates(Random((size_t)R * 24 * 512, 1.f, 33));
  std::vector<float> suh_values = Random(6144, .3f, 34);
  for (auto& v : suh_values) v += 1.f;
  Buffer<__half> suh(Half(suh_values)), xh((size_t)R * 6144);
  QsaArgs a{};
  a.c.k = ck.p; a.c.v = cv.p; a.sel = sel.p; a.nsel = nsel.p; a.pos0 = pos0; a.R = R;
  a.q = gates.p; a.suh_o = suh.p; a.xh_o = xh.p;
  QsaPrefill f{a, context.p, qhi.p, nullptr};
  const int tiles = (R + kUnionRows - 1) / kUnionRows, blocks = (pos0 + R - 1) / 4 + 1;
  Buffer<int> ulist(QsaUnionListInts(R)), ucount(tiles);
  QsaUnion u{ulist.p, ucount.p};

  context.Zero(); xh.Zero();
  QsaUnionBuildKernel<<<tiles, 256, blocks>>>(a, u);
  QsaUnionFlashKernel<false><<<dim3(tiles, 2), 256>>>(f, u);
  QsaPrefillGateKernel<<<dim3(24, R), 256>>>(f);
  CheckLaunch();
  const auto c0 = context.Get();
  const auto x0 = xh.Get();

  HIP_OK(hipMemset(context.p, 0x7f, context.n * sizeof(float)));
  LaunchQsaUnionFlash(f, u);
  CheckLaunch();
  const bool context_same = Same(context.Get(), c0);
  HIP_OK(hipMemset(xh.p, 0x7f, xh.n * sizeof(__half)));
  LaunchQsaUnionFlashGate(f, u);
  CheckLaunch();
  const bool xh_same = Same(xh.Get(), x0);
  if (!context_same || !xh_same) {
    std::fprintf(stderr, "FAIL: pos0=%d R=%d context=%d xh=%d\n", pos0, R, context_same, xh_same);
    std::exit(1);
  }
  std::printf("PASS: union flash pos0=%d R=%d\n", pos0, R);
}
}  // namespace

int main() {
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0) return 2;
  Check(0, 1);
  Check(0, 2048);
  Check(1, 5);
  Check(2040, 37);
  Check(4093, 2047);
  Check(30720, 2048);
  Check(65531, 1031);
  std::printf("PASS: qsa_union_flash_test\n");
}
