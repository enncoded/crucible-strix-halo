#include "prefill_common.h"
#include "../engine/layers/kernels/prefill_qsa_select.h"
#include "../engine/layers/kernels/prefill_qsa_score_sgpr.h"
#include <bit>
#include <fstream>
#include <string>

using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;

namespace {
// Bounded selector chunks (idea from Gufo, gufo-org/gufo, MIT): past kQsaCandidates blocks, score and select
// `group` rows at a time, so a group's fp32 scores (group x pos/4 floats) and candidates stay in the 32 MB MALL
// between the score and the top-k kernels. Every row keeps the two-pass path and its own block count, so the
// selection is the same. Shorter parts keep one launch. Measured slower than one launch,
// so only this bench uses it.
template <class Select>
void QsaSelectGroups(QsaArgs a, int group, Select select) {
  if (group <= 0 || a.pos0/4 <= kQsaCandidates || a.R <= group) { select(a); return; }
  for (int g0 = 0; g0 < a.R; g0 += group) {
    QsaArgs b = a;
    b.pos0 = a.pos0+g0; b.R = std::min(group, a.R-g0);
    b.idx = a.idx+(size_t)g0*640; b.sel = a.sel+(size_t)g0*kSelBlocks; b.nsel = a.nsel+g0;
    select(b);
  }
}

template<class T> std::vector<T> Read(const std::string& path, size_t count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in || in.tellg() != (std::streamoff)(count*sizeof(T))) {
    std::fprintf(stderr, "wrong capture size: %s\n", path.c_str()); std::exit(2);
  }
  std::vector<T> x(count); in.seekg(0);
  if (!in.read(reinterpret_cast<char*>(x.data()), count*sizeof(T))) std::exit(2);
  return x;
}

size_t Differences(const std::vector<float>& x, const std::vector<float>& y) {
  size_t n = 0;
  for (size_t i = 0; i < x.size(); ++i) n += std::bit_cast<unsigned>(x[i]) != std::bit_cast<unsigned>(y[i]);
  return n;
}

template<int MR, int NK, int PF, bool BufferLoads = false, bool RowsFast = false> void Test(QsaArgs a, Buffer<float>& out,
    const std::vector<float>& reference, const std::vector<int>& ids,
    const std::vector<int>& counts, Buffer<float>& scratch, int repeats, const char* source) {
  a.scores = out.p; out.Zero();
  auto launch = [&] { LaunchQsaPrefillScores<MR,NK,PF,BufferLoads,RowsFast>(a); };
  launch(); CheckLaunch();
  const size_t different = Differences(out.Get(), reference);
  LaunchQsaTop(a, scratch.p); CheckLaunch();
  std::vector<int> actual(ids.size()), nc(counts.size());
  HIP_OK(hipMemcpy(actual.data(), a.sel, actual.size()*sizeof(int), hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(nc.data(), a.nsel, nc.size()*sizeof(int), hipMemcpyDeviceToHost));
  const bool selected = actual == ids && nc == counts;
  const double us = Time(launch, repeats);
  double fmas = 0;
  for (int r = 0; r < a.R; ++r) if ((a.pos0+r+1)/4 > 512) fmas += (a.pos0+r+1)/4*512.;
  std::printf("blocked source=%s ctx=%d rows=%d mr=%d nk=%d pf=%d buffer=%d rows_fast=%d us=%.3f roof_pct=%.2f score_diff=%zu selection_exact=%d\n",
      source, a.pos0+a.R, a.R, MR, NK, PF, BufferLoads, RowsFast, us, fmas/6.6e12/us*1e8, different, selected);
  std::fflush(stdout);
  if (different || !selected) std::exit(1);
}

template<int MR, int NK> void SgprTest(QsaArgs a, Buffer<float>& out,
    const std::vector<float>& reference, const std::vector<int>& ids,
    const std::vector<int>& counts, Buffer<float>& scratch, int repeats, const char* source) {
  a.scores = out.p; out.Zero();
  Buffer<float> qt((size_t)(a.R+MR-1)/MR*512*MR);
  auto launch = [&] { LaunchQsaPrefillSgprScores<MR,NK>(a,qt.p); };
  launch(); CheckLaunch();
  const size_t different = Differences(out.Get(), reference);
  LaunchQsaTop(a, scratch.p); CheckLaunch();
  std::vector<int> actual(ids.size()), nc(counts.size());
  HIP_OK(hipMemcpy(actual.data(), a.sel, actual.size()*sizeof(int), hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(nc.data(), a.nsel, nc.size()*sizeof(int), hipMemcpyDeviceToHost));
  const bool selected = actual == ids && nc == counts;
  const double us = Time(launch, repeats);
  double fmas = 0;
  for (int r = 0; r < a.R; ++r) if ((a.pos0+r+1)/4 > 512) fmas += (a.pos0+r+1)/4*512.;
  std::printf("sgpr source=%s ctx=%d rows=%d mr=%d nk=%d us=%.3f roof_pct=%.2f score_diff=%zu selection_exact=%d\n",
      source, a.pos0+a.R, a.R, MR, NK, us, fmas/6.6e12/us*1e8, different, selected);
  std::fflush(stdout);
  if (different || !selected) std::exit(1);
}

// The engine's select entry point with query tiles, as LaunchQsaPrefillUnion calls it.
void EngineTest(QsaArgs a, Buffer<float>& out, const std::vector<float>& reference,
    const std::vector<int>& ids, const std::vector<int>& counts, Buffer<float>& scratch,
    int repeats, const char* source) {
  a.scores = out.p; out.Zero();
  Buffer<float> qt(QsaQueryTileFloats(a.R));
  auto launch = [&] { LaunchQsaPrefillSelect(a,scratch.p,qt.p); };
  launch(); CheckLaunch();
  const size_t different = Differences(out.Get(), reference);
  std::vector<int> actual(ids.size()), nc(counts.size());
  HIP_OK(hipMemcpy(actual.data(), a.sel, actual.size()*sizeof(int), hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(nc.data(), a.nsel, nc.size()*sizeof(int), hipMemcpyDeviceToHost));
  const bool selected = actual == ids && nc == counts;
  const double us = Time(launch, repeats);
  std::printf("engine source=%s ctx=%d rows=%d select_us=%.3f score_diff=%zu selection_exact=%d\n",
      source, a.pos0+a.R, a.R, us, different, selected);
  std::fflush(stdout);
  if (different || !selected) std::exit(1);
}

// Bounded selector chunks: the engine select entry point on `group` rows at a time, with a score buffer of
// group rows (the MALL-sized working set), against the one-launch selection.
void GroupedTest(QsaArgs a, const std::vector<int>& ids, const std::vector<int>& counts, int repeats,
    const char* source) {
  Buffer<float> full_scores((size_t)a.R*a.sstride), qt(QsaQueryTileFloats(a.R));
  Buffer<float> scratch(QsaSelectScratchFloats(a.R));
  QsaArgs f = a; f.scores = full_scores.p;
  auto one = [&] { LaunchQsaPrefillSelect(f,scratch.p,qt.p); };
  one(); CheckLaunch();
  const double one_us = Time(one, repeats);
  for (int group : {64, 128, 256, 512}) {
    Buffer<float> scores((size_t)group*a.sstride), gscratch(QsaSelectScratchFloats(group));
    Buffer<int> sel(ids.size()), nsel(counts.size());
    QsaArgs g = a; g.scores = scores.p; g.sel = sel.p; g.nsel = nsel.p;
    auto grouped = [&] { QsaSelectGroups(g,group,[&](QsaArgs b) { LaunchQsaPrefillSelect(b,gscratch.p,qt.p); }); };
    grouped(); CheckLaunch();
    const bool selected = sel.Get() == ids && nsel.Get() == counts;
    const double us = Time(grouped, repeats);
    std::printf("grouped source=%s ctx=%d rows=%d group=%d one_us=%.3f grouped_us=%.3f ratio=%.4f selection_exact=%d\n",
        source, a.pos0+a.R, a.R, group, one_us, us, us/one_us, selected);
    std::fflush(stdout);
    if (!selected) std::exit(1);
  }
}

template<int MR, int PF, bool BufferLoads = false> void FusedTest(QsaArgs a, Buffer<float>& out,
    const std::vector<float>& reference, const std::vector<int>& ids,
    const std::vector<int>& counts, Buffer<float>& scratch, int repeats, const char* source) {
  a.scores = out.p; out.Zero();
  LaunchQsaPrefillScoreTop<MR,PF,true,BufferLoads>(a,scratch.p); CheckLaunch();
  const size_t different = Differences(out.Get(),reference);
  std::vector<int> actual(ids.size()), nc(counts.size());
  HIP_OK(hipMemcpy(actual.data(),a.sel,actual.size()*sizeof(int),hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(nc.data(),a.nsel,nc.size()*sizeof(int),hipMemcpyDeviceToHost));
  const bool captured = actual == ids && nc == counts;
  const double us = Time([&] { LaunchQsaPrefillScoreTop<MR,PF,false,BufferLoads>(a,scratch.p); },repeats);
  HIP_OK(hipMemcpy(actual.data(),a.sel,actual.size()*sizeof(int),hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(nc.data(),a.nsel,nc.size()*sizeof(int),hipMemcpyDeviceToHost));
  const bool selected = captured && actual == ids && nc == counts;
  double fmas = 0;
  for (int r = 0; r < a.R; ++r) if ((a.pos0+r+1)/4 > 512) fmas += (a.pos0+r+1)/4*512.;
  std::printf("fused source=%s ctx=%d rows=%d mr=%d pf=%d buffer=%d us=%.3f roof_pct=%.2f score_diff=%zu selection_exact=%d\n",
      source,a.pos0+a.R,a.R,MR,PF,BufferLoads,us,fmas/6.6e12/us*1e8,different,selected);
  std::fflush(stdout);
  if (different || !selected) std::exit(1);
}

// Variant sets: 0 all prototypes, 1 the engine shape, 2 MR4 and MR8, 3 row-fast grids,
// 4 SGPR query probes, 5 the engine select entry point.
void Run(int ctx, int rows, int mode, const std::string& capture, int repeats, int chosen = 0) {
  int pos0 = ctx-rows;
  if (!capture.empty()) {
    std::ifstream meta(capture+".meta");
    if (!(meta >> pos0 >> rows) || pos0 < 0 || rows < 1 || rows > 2048) std::exit(2);
    ctx = pos0+rows;
  }
  const int stride = (ctx+3)/4;
  auto iq = capture.empty() ? Random((size_t)rows*640, 1.f, 17) : Read<float>(capture+".idx.f32", (size_t)rows*640);
  auto pool = capture.empty() ? Random(QsaPoolFloats(ctx), 1.f, 19) : Read<float>(capture+".pool.f32", QsaPoolFloats(ctx));
  if (mode == 1) std::fill(pool.begin(), pool.end(), 0.f);
  if (mode == 2) {
    for (int b = 0; b < stride; ++b)
      for (int d = 0; d < 128; ++d) pool[QsaPoolOffset(b,d)] = (float)((b%7)-3)*.25f;
  }
  Buffer<float> query(iq), keys(pool), scores((size_t)rows*stride), out((size_t)rows*stride);
  Buffer<float> scratch(QsaSelectScratchFloats(rows));
  Buffer<int> sel((size_t)rows*512), nsel(rows);
  QsaArgs a{}; a.idx=query.p; a.c.ik_pool=keys.p; a.scores=scores.p; a.sstride=stride;
  a.sel=sel.p; a.nsel=nsel.p; a.pos0=pos0; a.R=rows;
  auto base = [&] { QsaScoreKernel<<<dim3((stride+255)/256,rows),256>>>(a); };
  base(); CheckLaunch();
  const auto reference = scores.Get();
  LaunchQsaTop(a, scratch.p); CheckLaunch();
  const auto ids = sel.Get(), counts = nsel.Get();
  if (!capture.empty()) {
    const auto expected = Read<int>(capture+".sel.i32",(size_t)rows*512);
    const auto nc = Read<int>(capture+".nsel.i32",rows);
    if (counts != nc) { std::fprintf(stderr,"capture count differs\n"); std::exit(1); }
    for (int r = 0; r < rows; ++r) if (counts[r] == 512)
      for (int i = 0; i < 512; ++i) if (ids[r*512+i] != expected[r*512+i]) {
        std::fprintf(stderr,"capture selection differs row=%d slot=%d\n",r,i); std::exit(1);
      }
  }
  double fmas = 0;
  for (int r = 0; r < rows; ++r) if ((pos0+r+1)/4 > 512) fmas += (pos0+r+1)/4*512.;
  const double base_us = Time(base,repeats);
  const double top_us = Time([&] { LaunchQsaTop(a,scratch.p); },repeats);
  const double legacy_top_us = Time([&] { QsaTopKernel<<<rows,1024>>>(a); },repeats);
  const char* source = capture.empty() ? (mode == 0 ? "random" : mode == 1 ? "zero" : "tied") : capture.c_str();
  std::printf("baseline source=%s ctx=%d pos0=%d rows=%d us=%.3f roof_pct=%.2f top_us=%.3f legacy_top_us=%.3f\n",
      source,ctx,pos0,rows,base_us,fmas/6.6e12/base_us*1e8,top_us,legacy_top_us);
  if (chosen == 1) {
    Test<4,8,2,true,true>(a,out,reference,ids,counts,scratch,repeats,source);
    return;
  }
  if (chosen == 5) {
    EngineTest(a,out,reference,ids,counts,scratch,repeats,source);
    return;
  }
  if (chosen == 6) {
    GroupedTest(a,ids,counts,repeats,source);
    return;
  }
  if (chosen) {
    Test<4,4,4,true>(a,out,reference,ids,counts,scratch,repeats,source);
    if (chosen == 2) {
      Test<8,4,2,true>(a,out,reference,ids,counts,scratch,repeats,source);
      Test<8,4,4,true>(a,out,reference,ids,counts,scratch,repeats,source);
      Test<8,2,4,true>(a,out,reference,ids,counts,scratch,repeats,source);
    }
    if (chosen == 4) {
      Test<4,8,2,true,true>(a,out,reference,ids,counts,scratch,repeats,source);
      SgprTest<8,4>(a,out,reference,ids,counts,scratch,repeats,source);
      SgprTest<8,2>(a,out,reference,ids,counts,scratch,repeats,source);
      SgprTest<16,2>(a,out,reference,ids,counts,scratch,repeats,source);
      SgprTest<4,8>(a,out,reference,ids,counts,scratch,repeats,source);
    }
    if (chosen == 3) {
      Test<4,4,4,true,true>(a,out,reference,ids,counts,scratch,repeats,source);
      Test<8,4,2,true>(a,out,reference,ids,counts,scratch,repeats,source);
      Test<8,4,2,true,true>(a,out,reference,ids,counts,scratch,repeats,source);
      Test<8,4,4,true,true>(a,out,reference,ids,counts,scratch,repeats,source);
      Test<4,8,2,true,true>(a,out,reference,ids,counts,scratch,repeats,source);
    }
    return;
  }
  Test<1,4,8>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<2,2,8>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<2,4,8>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<4,2,8>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<4,4,8>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<4,4,4>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<4,4,16>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<8,2,8>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<1,8,8,true>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<2,4,8,true>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<2,8,4,true>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<4,2,8,true>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<4,4,4,true>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<8,2,8,true>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<8,2,4,true>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<8,2,2,true>(a,out,reference,ids,counts,scratch,repeats,source);
  Test<8,4,2,true>(a,out,reference,ids,counts,scratch,repeats,source);
  FusedTest<1,8>(a,out,reference,ids,counts,scratch,repeats,source);
  FusedTest<2,4>(a,out,reference,ids,counts,scratch,repeats,source);
  FusedTest<2,8>(a,out,reference,ids,counts,scratch,repeats,source);
  FusedTest<1,8,true>(a,out,reference,ids,counts,scratch,repeats,source);
  FusedTest<2,4,true>(a,out,reference,ids,counts,scratch,repeats,source);
  FusedTest<2,8,true>(a,out,reference,ids,counts,scratch,repeats,source);
}
}

int VariantSet(const std::string& name) { return name == "chosen" ? 1 : name == "mr8" ? 2 : name == "rowsfast" ? 3 : name == "sgpr" ? 4 : name == "engine" ? 5 : name == "grouped" ? 6 : 0; }

int main(int argc, char** argv) {
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0) return 2;
  if (argc > 1 && std::string(argv[1]) == "capture") {
    const int set = argc == 4 ? VariantSet(argv[3]) : 0;
    if (argc != 3 && (argc != 4 || set == 0)) return 2;
    Run(0,0,0,argv[2],3,set); return 0;
  }
  const int ctx = argc > 1 ? std::stoi(argv[1]) : 16384;
  const int rows = argc > 2 ? std::stoi(argv[2]) : 2048;
  const int mode = argc > 3 ? std::stoi(argv[3]) : 0;
  const int repeats = argc > 4 ? std::stoi(argv[4]) : 5;
  const int chosen = argc == 6 ? VariantSet(argv[5]) : 0;
  if (ctx < rows || ctx > 262144 || rows < 1 || rows > 2048 || mode < 0 || mode > 2 || repeats < 1
      || argc > 6 || (argc == 6 && !chosen)) return 2;
  Run(ctx,rows,mode,"",repeats,chosen);
}
