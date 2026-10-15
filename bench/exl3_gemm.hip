// Standalone prefill benchmark. Run every GPU command under timeout.
// exl3_gemm check CKPT
// exl3_gemm sweep CKPT PREFIX M
// exl3_gemm dense|grouped CKPT M [RM=auto] [LDS=1] [S=1] [PF=1]
#include "exl3_gemm_common.h"
#include "kernels/exl3_gemv.h"
using namespace gm;

static std::vector<float> Gemv(const Matrix& m, const __half* x) {
  float* out;
  Exl3Job* dj; Exl3Item* di;
  HIP_CHECK(hipMalloc(&out, (size_t)8 * m.O * 4));
  Exl3Job j{m.dev, x, out, m.I / 16, m.O / 16, 1};
  std::vector<Exl3Item> items;
  for (int b = 0; b < m.O / 16; b += 8)
    items.push_back({0, 0, b, std::min(8, m.O / 16 - b)});
  HIP_CHECK(hipMalloc(&dj, sizeof j));
  HIP_CHECK(hipMalloc(&di, items.size() * sizeof(Exl3Item)));
  HIP_CHECK(hipMemcpy(dj, &j, sizeof j, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(di, items.data(), items.size() * sizeof(Exl3Item), hipMemcpyHostToDevice));
  using Fn = void (*)(const Exl3Job*, const Exl3Item*, int);
  Fn fn = m.K == 3 ? Exl3GemvKernel<3, 8, 1, 8> :
          m.K == 4 ? Exl3GemvKernel<4, 8, 1, 8> : Exl3GemvKernel<5, 8, 1, 8>;
  hipLaunchKernelGGL(fn, dim3(items.size()), dim3(256), 8 * m.I * 2, nullptr,
                     dj, di, (int)items.size());
  HIP_CHECK(hipGetLastError());
  std::vector<float> y((size_t)8 * m.O);
  HIP_CHECK(hipMemcpy(y.data(), out, y.size() * 4, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(out)); HIP_CHECK(hipFree(dj)); HIP_CHECK(hipFree(di));
  return y;
}

static std::pair<double, double> CheckResult(const Matrix& m, const std::vector<__half>& xh,
    const std::vector<float>& y, const std::vector<float>& gv, const CpuOracle& oracle, int M) {
  for (float v : y) Require(std::isfinite(v), "nonfinite GEMM output");
  double worst = 0, vg = 0;
  for (int r = 0; r < 17; ++r)
    for (int o = 0; o < m.O; ++o) {
      const size_t idx = (size_t)r * m.O + o;
      worst = std::max(worst, std::fabs(y[idx] - oracle.y[idx]) / (oracle.mag[idx] + 1e-30));
      if (r < 8) vg = std::max(vg, std::fabs(y[idx] - gv[idx]) / (oracle.mag[idx] + 1e-30));
    }
  // All states and outputs are covered above. Extra rows sample columns to
  // exercise larger row blocks and gathers without a large CPU GEMM.
  for (int o = 0; o < m.O; o += std::max(1, m.O / 64)) {
    std::vector<double> w(m.I);
    for (int i = 0; i < m.I; ++i) w[i] = Weight(m, i, o);
    for (int r = 17; r < M; ++r) {
      double ref = 0, mag = 0;
      const int source = (r * 37) % 160;
      for (int i = 0; i < m.I; ++i) {
        const double v = __half2float(xh[(size_t)source * m.I + i]) * w[i];
        ref += v; mag += std::fabs(v);
      }
      worst = std::max(worst, std::fabs(y[(size_t)r * m.O + o] - ref) / (mag + 1e-30));
    }
  }
  return {worst, vg};
}

static int CheckMatrix(Matrix& m, bool tiled, bool full, bool pipe, bool packed, bool compact) {
  int fails = 0;
  Upload(m);
  auto xh = RandomX((size_t)160 * m.I);
  const CpuOracle oracle = FullOracle(m, xh);
  __half* x;
  HIP_CHECK(hipMalloc(&x, xh.size() * 2));
  HIP_CHECK(hipMemcpy(x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  auto gv = Gemv(m, x);
  auto xp = MakePackedInputs({m}, x, 160, packed && !compact);
  CheckPackedInputs(xp, xh, 160);
  const int rms[] = {1, 2, 4, 8, 16}, pfs[] = {1, 4, 8}, chunks[] = {4, 8, 16};
  for (int cfg = 0; cfg < (compact ? 54 : pipe ? 12 : tiled ? 30 : 120); ++cfg) {
        const int rm = rms[cfg % (compact || pipe ? 3 : 5)];
        const int lds = compact ? 14 + (cfg / 3) % 3 : packed ? 10 + (cfg / 5) % 4 :
                        pipe ? (full ? 7 : 6) : full ? 5 : tiled ? 4 : (cfg / 5) % 4;
        const int S = (cfg / (compact ? 9 : pipe ? 3 : tiled ? 5 : 20)) % 2 ? 3 : 1;
        const int pf = compact ? pfs[cfg / 18] : pipe ? chunks[cfg / 6] : tiled ? chunks[cfg / 10] : pfs[cfg / 40];
        const int M = rm * (tiled ? 64 : 16) + (full ? 0 : 1);
        std::vector<std::vector<int>> ids(1);
        for (int r = 0; r < M; ++r) ids[0].push_back(r < 8 ? r : (r * 37) % 160);
        Plan p = MakePlan({m}, M, rm, S, x, ids, 160, lds, xp.by_I);
        Launch(p, rm, lds, pf);
        uint64_t hash;
        const auto [worst, vg] = CheckResult(m, xh, Reduce(p, 0, &hash), gv, oracle, M);
        const bool ok = worst < 1e-5 && vg < 1e-5;
        fails += !ok;
        std::printf("%s K=%d M=%d RM=%d LDS=%d S=%d PF=%d fp64=%.3e gemv=%.3e %s hash=%016llx\n",
                    m.name.c_str(), m.K, M, rm, lds, S, pf, worst, vg, ok ? "PASS" : "FAIL",
                    (unsigned long long)hash);
        p.Free();
      }
  xp.Free(); HIP_CHECK(hipFree(x)); HIP_CHECK(hipFree(m.dev));
  return fails;
}

static int CheckGrouped(std::vector<Matrix> gs, bool tiled, bool pipe, bool packed, bool compact) {
  // Unequal gathered counts and an empty expert check compact partial offsets.
  for (auto& m : gs) Upload(m);
  auto hx = RandomX((size_t)32 * 2560);
  __half* dx;
  HIP_CHECK(hipMalloc(&dx, hx.size() * 2));
  HIP_CHECK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
  std::vector<std::vector<int>> ids(3);
  for (int r = 0; r < 17; ++r) ids[0].push_back((r * 7) % 32);
  for (int r = 0; r < 5; ++r) ids[1].push_back(31 - r);
  auto xp = MakePackedInputs(gs, dx, 32, packed && !compact);
  CheckPackedInputs(xp, hx, 32);
  const int layout = compact ? 14 : packed ? 10 : pipe ? 6 : tiled ? 4 : 1;
  Plan gp = MakePlan(gs, 32, 2, 3, dx, ids, 0, layout, xp.by_I);
  Launch(gp, 2, layout, tiled ? 8 : 1);
  double grouped = 0;
  uint64_t total_hash = 14695981039346656037ull;
  for (int j = 0; j < 2; ++j) {
    uint64_t hash;
    auto y = Reduce(gp, j, &hash);
    total_hash = (total_hash ^ hash) * 1099511628211ull;
    const auto& m = gs[j];
    for (int o = 0; o < m.O; o += m.O / 64) {
      std::vector<double> w(m.I);
      for (int i = 0; i < m.I; ++i) w[i] = Weight(m, i, o);
      for (size_t r = 0; r < ids[j].size(); ++r) {
        double ref = 0, mag = 0;
        for (int i = 0; i < m.I; ++i) {
          const double v = __half2float(hx[(size_t)ids[j][r] * m.I + i]) * w[i];
          ref += v; mag += std::fabs(v);
        }
        Require(std::isfinite(y[r * m.O + o]), "nonfinite grouped output");
        grouped = std::max(grouped, std::fabs(y[r * m.O + o] - ref) / (mag + 1e-30));
      }
    }
  }
  std::printf("mixed grouped jobs=3 counts=17,5,0 fp64=%.3e %s hash=%016llx\n", grouped,
              grouped < 1e-5 ? "PASS" : "FAIL", (unsigned long long)total_hash);
  gp.Free(); xp.Free(); HIP_CHECK(hipFree(dx));
  for (auto& m : gs) HIP_CHECK(hipFree(m.dev));
  return grouped >= 1e-5;
}

static int Check(st::Checkpoint& ck, bool tiled = false, bool full = false,
                 bool pipe = false, bool packed = false, bool compact = false) {
  const std::string L = "model.language_model.layers.";
  std::vector<Matrix> ms;
  for (const auto& n : {L + "0.linear_attn.in_proj_qkv", L + "3.self_attn.q_proj",
                       L + "0.mlp.experts.0.gate_proj", L + "0.mlp.experts.0.down_proj"})
    ms.push_back(Load(ck, n));
  Matrix k4{"synthetic-k4", 256, 128, 4, std::vector<uint32_t>(256 * 128 / 8)};
  std::mt19937 rng(19);
  for (auto& v : k4.host) v = rng();
  ms.push_back(std::move(k4));
  int fails = 0;
  for (auto& m : ms) fails += CheckMatrix(m, tiled, full, pipe, packed, compact);
  fails += CheckGrouped({ms[2], ms[3], ms[2]}, tiled, pipe, packed, compact);
  return fails ? 1 : 0;
}

static std::vector<Matrix> Dense(st::Checkpoint& ck) {
  std::vector<Matrix> ms;
  for (int l = 0; l < 48; ++l) {
    const std::string L = "model.language_model.layers." + std::to_string(l);
    if ((l + 1) % 4)
      for (const auto* n : {"in_proj_qkv", "in_proj_z", "out_proj"})
        ms.push_back(Load(ck, L + ".linear_attn." + n));
    else
      for (const auto* n : {"q_proj", "k_proj", "v_proj", "o_proj"})
        ms.push_back(Load(ck, L + ".self_attn." + n));
    for (const auto* n : {"gate_proj", "up_proj", "down_proj"})
      ms.push_back(Load(ck, L + ".mlp.shared_expert." + n));
  }
  ms.push_back(Load(ck, "lm_head"));
  return ms;
}

static int Families(st::Checkpoint& ck, int M, int lds, int pf) {
  Require(M >= 16 && M <= 4096, "M must be in [16,4096]");
  auto ms = Dense(ck);
  int Imax = 0;
  for (auto& m : ms) { Upload(m); Imax = std::max(Imax, m.I); }
  auto hx = RandomX((size_t)M * Imax);
  __half* x;
  HIP_CHECK(hipMalloc(&x, hx.size() * 2));
  HIP_CHECK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
  auto xp = MakePackedInputs(ms, x, M, SharedPackedLayout(lds));
  auto cold = MakeColdCache();
  std::puts("family,M,RM,LDS,PF,matrices,weight_MB,ms,TFLOPS");
  for (const std::string suffix : {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
       "linear_attn.out_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
       "self_attn.o_proj", "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
       "mlp.shared_expert.down_proj", "lm_head"}) {
    std::vector<Matrix> group;
    size_t bytes = 0;
    for (const auto& m : ms) {
      if (!m.name.ends_with(suffix)) continue;
      group.push_back({m.name, m.I, m.O, m.K, {}, m.dev});
      bytes += m.host.size() * 4;
    }
    for (int rm : {1, 2, 4, 8, 16}) {
      if ((lds == 6 || lds == 7 || CompactLayout(lds)) && rm > 4) continue;
      if ((lds == 5 || lds == 7) && M % (64 * rm)) continue;
      Plan p = MakePlan(group, M, rm, 1, x, {}, 0, lds, xp.by_I);
      const double t = Time(p, rm, lds, pf, &cold);
      if (lds >= 6) ValidateFinite(p);
      std::printf("%s,%d,%d,%d,%d,%zu,%.1f,%.4f,%.3f\n", suffix.c_str(), M,
                  rm, lds, pf, group.size(), bytes / 1e6, t, p.flops / t / 1e9);
      p.Free();
    }
  }
  cold.Free(); xp.Free(); HIP_CHECK(hipFree(x));
  for (const auto& m : ms) HIP_CHECK(hipFree(m.dev));
  return 0;
}

static std::vector<std::vector<int>> Routing(int M) {
  std::vector<std::vector<int>> ids(512 * 3);
  std::mt19937 rng(29);
  for (int r = 0; r < M; ++r) {
    std::vector<int> seen;
    while (seen.size() < 10) {
      const int e = rng() % 512;
      if (std::find(seen.begin(), seen.end(), e) != seen.end()) continue;
      seen.push_back(e);
      for (int p = 0; p < 3; ++p) ids[e * 3 + p].push_back(r);
    }
  }
  return ids;
}

static int Run(st::Checkpoint& ck, const std::string& mode, int M, int rm, int lds, int S,
               const std::string& prefix = "", int pf = 1, bool flush = false) {
  Require(M >= 16 && M <= 4096, "M must be in [16,4096]");
  std::vector<Matrix> ms;
  std::vector<std::vector<int>> ids;
  if (mode == "dense") ms = Dense(ck);
  else if (mode == "grouped") {
    for (int e = 0; e < 512; ++e)
      for (const auto* n : {"gate_proj", "up_proj", "down_proj"})
        ms.push_back(Load(ck, "model.language_model.layers.0.mlp.experts." + std::to_string(e) + "." + n));
    ids = Routing(M);
  } else ms.push_back(Load(ck, prefix));
  int Imax = 0;
  size_t bytes = 0;
  for (auto& m : ms) { Upload(m); Imax = std::max(Imax, m.I); bytes += m.host.size() * 4; }
  auto xh = RandomX((size_t)M * Imax);
  __half* x;
  HIP_CHECK(hipMalloc(&x, xh.size() * 2));
  HIP_CHECK(hipMemcpy(x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  auto xp = MakePackedInputs(ms, x, M, SharedPackedLayout(lds));
  if (PackedLayout(lds)) std::printf("# pack_ms=%.4f shared_widths=%zu setup_only=1\n", xp.ms, xp.by_I.size());
  std::printf("# set=%s matrices=%zu weights_MB=%.1f PF=%d; median of 3 launches\n", mode.c_str(), ms.size(), bytes / 1e6, pf);
  if (!ids.empty()) {
    int lo = M, hi = 0, active = 0;
    for (int e = 0; e < 512; ++e) {
      int n = ids[e * 3].size(); lo = std::min(lo, n); hi = std::max(hi, n); active += n > 0;
    }
    std::printf("# routing experts=%d min=%d max=%d mean=%.2f unique_top10\n", active, lo, hi, M * 10.0 / 512);
  }
  ColdCache cold{};
  if (flush) cold = MakeColdCache();
  std::printf("# strict_cold=%d\n", flush);
  std::printf("set,M,RM,LDS,S,PF,ms,TFLOPS,tok_s,output_MB\n");
  auto point = [&](int r, int d, int s) {
    Plan p = MakePlan(ms, M, r, s, x, ids, 0, d, xp.by_I);
    const double t = Time(p, r, d, pf, flush ? &cold : nullptr);
    if (d >= 6) ValidateFinite(p);
    if (CompactLayout(d)) std::printf("# compact_pack_ms=%.4f setup_only=1\n", p.input_pack_ms);
    std::printf("%s,%d,%d,%d,%d,%d,%.4f,%.3f,%.1f,%.1f\n", mode.c_str(), M, r, d, s, pf,
                t, p.flops / t / 1e9, M * 1000.0 / t, p.output * 4 / 1e6);
    p.Free();
  };
  if (mode == "sweep")
    for (int r : {1, 2, 4, 8, 16})
      for (int d : {0, 1, 2, 3})
        for (int s : {1, 4}) point(r, d, s);
  else point(rm, lds, S);
  if (flush) cold.Free();
  xp.Free(); HIP_CHECK(hipFree(x));
  for (auto& m : ms) HIP_CHECK(hipFree(m.dev));
  return 0;
}

int main(int argc, char** argv) {
  Require(argc >= 3, "usage: exl3_gemm check|dense|grouped|sweep CKPT [M] [RM LDS S]; sweep CKPT PREFIX M");
  st::Checkpoint ck(argv[2]);
  const std::string mode = argv[1];
  if (mode == "check") return Check(ck);
  if (mode == "check-tiled") return Check(ck, true);
  if (mode == "check-tiled-full") return Check(ck, true, true);
  if (mode == "check-pipe") return Check(ck, true, false, true);
  if (mode == "check-pipe-full") return Check(ck, true, true, true);
  if (mode == "check-packed") return Check(ck, false, false, false, true);
  if (mode == "check-compact") return Check(ck, false, false, false, true, true);
  if (mode == "sweep") {
    Require(argc == 5, "sweep needs PREFIX M");
    return Run(ck, mode, std::atoi(argv[4]), 4, true, 1, argv[3]);
  }
  Require(mode == "dense" || mode == "grouped" || mode == "families", "unknown mode");
  Require(argc >= 4, "need M");
  if (mode == "families")
    return Families(ck, std::atoi(argv[3]), argc > 4 ? std::atoi(argv[4]) : 1,
                    argc > 5 ? std::atoi(argv[5]) : 1);
  const int M = std::atoi(argv[3]);
  const int default_rm = mode == "dense" ? 2 : M <= 512 ? 1 : M <= 1024 ? 2 : 4;
  return Run(ck, mode, M, argc > 4 ? std::atoi(argv[4]) : default_rm,
             argc > 5 ? std::atoi(argv[5]) : 1, argc > 6 ? std::atoi(argv[6]) : 1,
             "", argc > 7 ? std::atoi(argv[7]) : 1, argc > 8 && std::atoi(argv[8]) != 0);
}
