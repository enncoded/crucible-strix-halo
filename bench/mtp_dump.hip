// Teacher-forced trunk stacks for the MTP head fine-tune.
//
//   mtp_dump CKPT LIST OUTDIR [--ctx N] [--check 1] [--chain D]
//
// LIST lines: NAME<space>comma-separated token IDs. For each sequence the
// engine runs its production prompt path (bulk prefill when the checkpoint
// allows it) and writes the final trunk stacks (before the final HC mixer):
//   OUTDIR/NAME.stk   [n][10240] fp16 (raw IEEE half bits)
//   OUTDIR/NAME.ids   the token IDs, one line, comma-separated
// A stack value outside the fp16 range stops the run (exit 3).
// --check 1 also runs the engine MTP over every pair (H_p, ids[p+1]) at
// predictor position p, in 8-row catch-up chunks with the full head, and
// after each chunk a chain of D-1 more drafts from the last row (as decode
// does). It writes OUTDIR/NAME.f32 (fp32 stacks) and OUTDIR/NAME.mtp:
//   "p P TOP1 L1 L2"            catch-up row: argmax, top-1 and top-2 logits
//   "c P D1 D2 ..."             chain from pair P: depth-1..D draft IDs
#define ENGINE_LIBRARY
#include "../engine/engine.hip"

#include <cmath>
#include <cstdio>
#include <fstream>

namespace {

void WriteAll(const std::string& path, const void* data, size_t bytes) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f || std::fwrite(data, 1, bytes, f) != bytes || std::fclose(f) != 0) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(1);
  }
}

struct Top2 {
  int id = 0;
  float l1 = -INFINITY, l2 = -INFINITY;
};

Top2 RowTop2(const float* x, int n) {
  Top2 t;
  for (int v = 0; v < n; ++v) {
    if (x[v] > t.l1) { t.l2 = t.l1; t.l1 = x[v]; t.id = v; }
    else if (x[v] > t.l2) t.l2 = x[v];
  }
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || (argc - 4) % 2) {
    std::fprintf(stderr, "usage: mtp_dump CKPT LIST OUTDIR [--ctx N] [--check 1] [--chain D]\n");
    return 2;
  }
  int ctx = 8192, check = 0, chain = 4;
  for (int i = 4; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const int v = std::stoi(argv[i + 1]);
    if (k == "--ctx") ctx = v;
    else if (k == "--check") check = v;
    else if (k == "--chain") chain = v;
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }
  if (chain < 1 || chain > kMaxRows - 1) { std::fprintf(stderr, "--chain must be 1..7\n"); return 2; }
  if (!SetRuntimeMemoryDefaults()) return 1;
  std::vector<std::pair<std::string, std::vector<int>>> seqs;
  {
    std::ifstream f(argv[2]);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    for (std::string line; std::getline(f, line);) {
      const size_t sp = line.find(' ');
      if (line.empty() || sp == std::string::npos) continue;
      std::vector<int> ids = ParseIds(line.substr(sp + 1));
      if (ids.size() < 2 || (int)ids.size() > ctx) {
        std::fprintf(stderr, "sequence %s: need 2..%d IDs\n", line.substr(0, sp).c_str(), ctx);
        return 2;
      }
      seqs.emplace_back(line.substr(0, sp), std::move(ids));
    }
  }
  const std::string out = argv[3];
  Engine eng(argv[1], ctx, kPrefillAuto);
  if (check) eng.DraftHead({}, false);  // full head, every catch-up row scored
  size_t maxn = 0;
  for (auto& s : seqs) maxn = std::max(maxn, s.second.size());
  float* dstk = nullptr;
  float* cstk = nullptr;
  HIP_OK(hipMalloc(&dstk, maxn * kWide * sizeof(float)));
  HIP_OK(hipMalloc(&cstk, kWide * sizeof(float)));
  std::vector<float> host(maxn * kWide);
  std::vector<__half> half(maxn * kWide);
  std::vector<float> logits((size_t)kMaxRows * kVocab);
  double total_tokens = 0, prefill_s = 0;
  for (auto& [name, ids] : seqs) {
    const int n = (int)ids.size();
    eng.Reset();
    const double t0 = Now();
    const int step = eng.PrefillRows() > 0 ? eng.PrefillRows() : kMaxRows;
    for (int i = 0; i < n; i += step) {
      const int R = std::min(step, n - i);
      if (eng.PrefillRows() > 0) eng.Prefill(ids.data() + i, R, "", false);
      else eng.Forward(ids.data() + i, R, 0, R);
      HIP_OK(hipMemcpy(dstk + (size_t)i * kWide, eng.H(), (size_t)R * kWide * sizeof(float),
                       hipMemcpyDeviceToDevice));
    }
    HIP_OK(hipDeviceSynchronize());
    prefill_s += Now() - t0;
    total_tokens += n;
    HIP_OK(hipMemcpy(host.data(), dstk, (size_t)n * kWide * sizeof(float), hipMemcpyDeviceToHost));
    float amax = 0;
    for (size_t j = 0; j < (size_t)n * kWide; ++j) {
      const float a = std::fabs(host[j]);
      if (!(a <= 65504.f)) {
        std::fprintf(stderr, "sequence %s: stack value %g at %zu is outside fp16\n", name.c_str(), host[j], j);
        return 3;
      }
      amax = std::max(amax, a);
      half[j] = __float2half(host[j]);
    }
    WriteAll(out + "/" + name + ".stk", half.data(), (size_t)n * kWide * sizeof(__half));
    {
      std::ofstream f(out + "/" + name + ".ids");
      for (int j = 0; j < n; ++j) f << (j ? "," : "") << ids[j];
      f << "\n";
    }
    std::fprintf(stderr, "%s: %d tokens, max |stack| %.1f\n", name.c_str(), n, amax);
    if (!check) continue;
    WriteAll(out + "/" + name + ".f32", host.data(), (size_t)n * kWide * sizeof(float));
    std::ofstream mf(out + "/" + name + ".mtp");
    const int pairs = n - 1;
    for (int p = 0; p < pairs; p += kMaxRows) {
      const int R = std::min(kMaxRows, pairs - p);
      const std::vector<int> top = eng.Mtp(dstk + (size_t)p * kWide, ids.data() + p + 1, R, p);
      HIP_OK(hipMemcpy(logits.data(), eng.logits(), (size_t)R * kVocab * sizeof(float), hipMemcpyDeviceToHost));
      for (int r = 0; r < R; ++r) {
        const Top2 t = RowTop2(logits.data() + (size_t)r * kVocab, kVocab);
        if (t.id != top[r]) std::fprintf(stderr, "warning: host argmax %d != engine %d\n", t.id, top[r]);
        mf << "p " << p + r << " " << top[r] << " " << t.l1 << " " << t.l2 << "\n";
      }
      std::vector<int> drafts{top[R - 1]};
      for (int k = 1; k < chain; ++k) {
        HIP_OK(hipMemcpy(cstk, eng.H() + (size_t)(k == 1 ? R - 1 : 0) * kWide, kWide * sizeof(float),
                         hipMemcpyDeviceToDevice));
        drafts.push_back(eng.Mtp(cstk, &drafts.back(), 1, p + R + k - 1)[0]);
      }
      mf << "c " << p + R - 1;
      for (int d : drafts) mf << " " << d;
      mf << "\n";
    }
  }
  std::fprintf(stderr, "dumped %.0f tokens, prefill %.1f s (%.0f tok/s)\n", total_tokens, prefill_s,
               total_tokens / std::max(prefill_s, 1e-9));
  return 0;
}
