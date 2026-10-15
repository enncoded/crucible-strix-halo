// Pipelined QSA union flash variants against QsaUnionHalfQKernel on real engine
// selections (index-only captures) or synthetic ones. q, K, V are random.
// Usage: qsa_flash_pipe LOCAL_BLOCKS CAPTURE_PREFIX|synthetic:POS0 ...
#include "prefill_common.h"
#include "qsa_flash_pipe.h"
#include <cstring>
#include <fstream>
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

template <class T> bool Same(const std::vector<T>& x, const std::vector<T>& y) {
  return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
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
  Buffer<__half> qhi(Half(Random((size_t)R * 6144, .4f, 31)));
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
  QsaUnionBuildKernel<<<tiles, 256, blocks>>>(a, u);
  CheckLaunch();
  const auto list = ulist.Get();
  const dim3 grid(tiles, 2);

  auto base_ctx = [&] { QsaUnionHalfQKernel<false><<<dim3(2, tiles), 256>>>(f, u); };
  auto base_gate = [&] { QsaUnionHalfQKernel<true><<<dim3(2, tiles), 256>>>(f, u); };
  context.Zero(); base_ctx(); CheckLaunch();
  const auto c0 = context.Get();
  xh.Zero(); base_gate(); CheckLaunch();
  const auto x0 = xh.Get();
  bool all_same = true;
  auto check = [&](const char* name, auto ctx_launch, auto gate_launch) {
    HIP_OK(hipMemset(context.p, 0x7f, context.n * sizeof(float)));
    ctx_launch(); CheckLaunch();
    const bool cs = Same(context.Get(), c0);
    HIP_OK(hipMemset(xh.p, 0x7f, xh.n * sizeof(__half)));
    gate_launch(); CheckLaunch();
    const bool gs = Same(xh.Get(), x0);
    if (std::strncmp(name, "x_", 2) != 0) all_same = all_same && cs && gs;
    const double b0 = Time(base_gate), v0 = Time(gate_launch), v1 = Time(gate_launch), b1 = Time(base_gate);
    ulist.Set([&] {
      std::vector<int> local = list;
      for (auto& e : local) e = (e & ~0xffffff) | ((e & 0xffffff) % local_blocks);
      return local;
    }());
    const double lb0 = Time(base_gate), lv0 = Time(gate_launch), lv1 = Time(gate_launch), lb1 = Time(base_gate);
    ulist.Set(list);
    std::printf("%-8s pos0=%d base_us=%.1f var_us=%.1f ratio=%.3f local%d base_us=%.1f var_us=%.1f ratio=%.3f "
        "ctx_bitwise=%d xh_bitwise=%d\n",
        name, pos0, (b0 + b1) / 2, (v0 + v1) / 2, (v0 + v1) / (b0 + b1), local_blocks, (lb0 + lb1) / 2,
        (lv0 + lv1) / 2, (lv0 + lv1) / (lb0 + lb1), (int)cs, (int)gs);
    std::fflush(stdout);
  };
#define VARIANT_G(NAME, GRID, ...) \
  check(NAME, [&] { QsaUnionPipeKernel<false, __VA_ARGS__><<<GRID, 256>>>(f, u); }, \
        [&] { QsaUnionPipeKernel<true, __VA_ARGS__><<<GRID, 256>>>(f, u); })
#define VARIANT(NAME, ...) VARIANT_G(NAME, grid, __VA_ARGS__)
#define VARIANT_CU(NAME, GRID, ...) \
  check(NAME, [&] { QsaUnionPipeCuKernel<false, __VA_ARGS__><<<GRID, 256>>>(f, u); }, \
        [&] { QsaUnionPipeCuKernel<true, __VA_ARGS__><<<GRID, 256>>>(f, u); })
  const dim3 gridx(2, tiles);
  std::printf("source=%s\n", capture.c_str());
  VARIANT("copy", 0);
  VARIANT_CU("cu_gx_il1_d1", gridx, kPipeGX | kPipeIL1, 1);
  VARIANT_CU("cu_gx_il1_p2", gridx, kPipeGX | kPipeIL1 | kPipeP2, 1);
  VARIANT_CU("cu_gx_p2", gridx, kPipeGX | kPipeP2, 1);
#undef VARIANT
#undef VARIANT_G
#undef VARIANT_CU
  if (!all_same) std::exit(9);
}
}  // namespace

int main(int argc, char** argv) {
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0) return 2;
  if (argc < 3) {
    std::fprintf(stderr, "usage: qsa_flash_pipe LOCAL_BLOCKS CAPTURE_PREFIX|synthetic:POS0 ...\n");
    return 2;
  }
  const int local = std::stoi(argv[1]);
  if (local < 512 || local > 65536) return 2;
  for (int i = 2; i < argc; ++i) Run(argv[i], local);
}
