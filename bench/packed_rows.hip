// Packed verify passes of up to kPackedRows rows against each span alone.
//
//   packed_rows CHECKPOINT IDS_FILE [reps] [ctx_tokens]
//
// Four streams get the prompt plus 0, 8, 16 and 24 greedy tokens of
// different length, then the next 8 greedy IDs of each stream are its draft
// rows (all right). For each row split (VERIFY_SPLITS, default
// "2,2,2,2;4,4;4,4,4,4;8,8;4,4,4,2;6,5,3"), every span is first verified
// alone (one ForwardVerifySpans call of at most 8 rows, the decode path), then
// all spans in one packed pass. The packed logits and residual stacks of each
// span must equal its lone pass bit for bit. Each sample restores every
// stream's state first. Prints the median wall ms of the packed pass and of
// the lone passes summed, and the distinct experts per layer of the pass.
// It is also the device regression test of packed passes past row 8: the
// QSA layers of a pass with more than one span run PackedQsaCore, whose
// split scratch holds kPackedRows rows; 8,8 and 6,5,3 end spans past row 8.
// With CRUCIBLE_VERIFY_QSA_CHECK=1 every packed QSA layer is also compared
// with the per-span chains; any mismatch fails the run. Exit 1 on a failure.
// Before each timed packed pass it launches TraceMarkKernel with grid
// 64 * split + rows (32 threads) for kernel traces (trace_verify.py
// with TAG_BASE=64). lookahead_mb: the median bytes that the lookahead fetched
// in one timed packed pass.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <fstream>

#include "common.h"

__global__ void TraceMarkKernel(int tag) {
  if (tag < 0) __builtin_trap();
}

// Bytes that the lookahead fetched in the last pass (0 when it does not run).
// The lookahead zeroes the count at Start; a pass without lookahead keeps the
// zero written here before it.
crucible::LookaheadState* LookaheadStatePtr() {
  crucible::LookaheadState* p = nullptr;
  HIP_OK(hipMemcpyFromSymbol(&p, HIP_SYMBOL(crucible::g_lookahead), sizeof(p)));
  return p;
}
void ZeroLookaheadFetched(crucible::LookaheadState* p) {
  if (p) HIP_OK(hipMemset(&p->fetched, 0, sizeof(p->fetched)));
}
double LookaheadFetchedMB(crucible::LookaheadState* p) {
  unsigned long long fetched = 0;
  if (p) HIP_OK(hipMemcpy(&fetched, &p->fetched, sizeof(fetched), hipMemcpyDeviceToHost));
  return fetched / 1e6;
}

namespace {
struct Saved { void* target; uint8_t* copy; size_t bytes; };
struct StreamState { std::vector<Saved> buffers; std::vector<int> hist; };

// Engine names this friend; both sit in the same unnamed namespace.
struct EngineStateIO {
  static StreamState Save(Engine& e, int stream) {
    StreamState s;
    const SequenceState& q = *e.sequences_[stream];
    for (const DeviceSpan& span : q.Spans()) {
      uint8_t* copy;
      HIP_OK(hipMalloc(&copy, span.bytes));
      HIP_OK(hipMemcpy(copy, span.pointer, span.bytes, hipMemcpyDeviceToDevice));
      s.buffers.push_back({const_cast<void*>(span.pointer), copy, span.bytes});
    }
    s.hist = q.hist;
    return s;
  }
  static void Restore(Engine& e, int stream, const StreamState& s) {
    for (const Saved& b : s.buffers) HIP_OK(hipMemcpyAsync(b.target, b.copy, b.bytes, hipMemcpyDeviceToDevice, 0));
    SequenceState& q = *e.sequences_[stream];
    q.hist = s.hist;
    q.retained.Reset();
    if (!q.retained.Forward((int)s.hist.size(), 0, (int)s.hist.size()).empty()) std::exit(1);
  }
  // Logits and residual stacks of rows [row, row + n) of the last pass.
  static std::pair<long, long> QsaChecks(const Engine& e) { return {e.qsa_checks_, e.qsa_mismatches_}; }
  static std::vector<uint8_t> Rows(Engine& e, int row, int n) {
    const size_t hb = size_t(n) * kWide * sizeof(float), lb = size_t(n) * kVocab * sizeof(float);
    std::vector<uint8_t> out(hb + lb);
    HIP_OK(hipMemcpy(out.data(), e.H_ + size_t(row) * kWide, hb, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(out.data() + hb, e.logits_ + size_t(row) * kVocab, lb, hipMemcpyDeviceToHost));
    return out;
  }
};

std::vector<std::vector<int>> Splits() {
  const char* env = std::getenv("VERIFY_SPLITS");
  std::stringstream list(env ? env : "2,2,2,2;4,4;4,4,4,4;8,8;4,4,4,2;6,5,3");
  std::vector<std::vector<int>> out;
  for (std::string item; std::getline(list, item, ';');) {
    std::vector<int> split;
    std::stringstream parts(item);
    for (std::string n; std::getline(parts, n, ',');) split.push_back(std::atoi(n.c_str()));
    out.push_back(split);
  }
  return out;
}
double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::fprintf(stderr, "usage: packed_rows CHECKPOINT IDS_FILE [reps] [ctx_tokens]\n");
    return 2;
  }
  std::ifstream file(argv[2]);
  std::stringstream text;
  text << file.rdbuf();
  const std::vector<int> prompt = ParseIds(text.str());
  const int reps = argc > 3 ? std::atoi(argv[3]) : 9;
  const int ctx_tokens = argc > 4 ? std::atoi(argv[4]) : 256;
  constexpr int kStreamsUsed = 4;
  if (!file || prompt.empty() || !ValidTokenIds(prompt, kVocab) || reps < 1 ||
      ctx_tokens < (int)prompt.size() + 8 * kStreamsUsed)
    return 2;
  Engine engine(argv[1], ctx_tokens + 64);
  engine.ConfigureStreams(kStreamsUsed);
  std::vector<std::vector<int>> drafts(kStreamsUsed);
  std::vector<StreamState> saved;
  for (int s = 0; s < kStreamsUsed; ++s) {
    engine.SelectStream(s);
    engine.Reset();
    std::vector<int> out;
    for (size_t i = 0; i < prompt.size(); i += 8) {
      const int n = (int)std::min<size_t>(8, prompt.size() - i);
      out = engine.Forward(prompt.data() + i, n, 0, n);
    }
    // Stream s: 8 s more greedy tokens, then its 8 draft rows.
    std::vector<int> gen{out.back()};
    while ((int)gen.size() < 8 * s + 9) gen.push_back(engine.Forward(&gen.back(), 1, 0, 1)[0]);
    drafts[s].assign(gen.end() - 9, gen.end() - 1);
    // Roll back the 8 draft rows: restore needs the state before them.
    engine.Reset();
    std::vector<int> prefix = prompt;
    prefix.insert(prefix.end(), gen.begin(), gen.end() - 9);
    for (size_t i = 0; i < prefix.size(); i += 8) {
      const int n = (int)std::min<size_t>(8, prefix.size() - i);
      engine.Forward(prefix.data() + i, n, 0, n);
    }
    saved.push_back(EngineStateIO::Save(engine, s));
  }
  engine.WarmPlans();
  const int max_rows = engine.MaxPassRows();
  std::printf("platform_profile=%s streams=%d max_pass_rows=%d\n", [] {
    std::ifstream f("/sys/firmware/acpi/platform_profile");
    std::string p;
    std::getline(f, p);
    return p;
  }().c_str(), kStreamsUsed, max_rows);
  auto restore_all = [&] {
    for (int s = 0; s < kStreamsUsed; ++s) EngineStateIO::Restore(engine, s, saved[s]);
    HIP_OK(hipDeviceSynchronize());
  };
  int fails = 0;
  bench::Sampler sampler;
  sampler.Start();
  auto* lookahead = LookaheadStatePtr();
  std::printf("split,rows,packed_ms,lone_sum_ms,experts,exact,lookahead_mb\n");
  const auto splits = Splits();
  for (size_t si = 0; si < splits.size(); ++si) {
    const auto& split = splits[si];
    std::vector<VerifySpan> spans;
    int rows = 0;
    std::string name;
    for (size_t s = 0; s < split.size() && s < (size_t)kStreamsUsed; ++s) {
      spans.push_back({(int)s, std::vector<int>(drafts[s].begin(), drafts[s].begin() + split[s])});
      rows += split[s];
      name += (s ? "+" : "") + std::to_string(split[s]);
    }
    if (rows > max_rows) {
      std::printf("%s,%d,,,,SKIP\n", name.c_str(), rows);
      continue;
    }
    // Lone references.
    std::vector<std::vector<uint8_t>> ref;
    for (const auto& span : spans) {
      restore_all();
      engine.ForwardVerifySpans({span});
      ref.push_back(EngineStateIO::Rows(engine, 0, (int)span.tokens.size()));
    }
    restore_all();
    engine.CountExperts(true);
    engine.ClearExpertCounts();
    engine.ForwardVerifySpans(spans);
    const auto counts = engine.ExpertCounts();
    engine.CountExperts(false);
    const double experts = (double)counts[2 * rows] / std::max<unsigned long long>(1, counts[2 * rows + 1]);
    bool exact = true;
    for (size_t s = 0, row = 0; s < spans.size(); row += spans[s].tokens.size(), ++s)
      exact = exact && EngineStateIO::Rows(engine, (int)row, (int)spans[s].tokens.size()) == ref[s];
    fails += !exact;
    std::vector<double> packed, lone, fetched;
    for (int rep = 0; rep <= reps; ++rep) {
      restore_all();
      ZeroLookaheadFetched(lookahead);
      TraceMarkKernel<<<64 * (int)si + rows, 32>>>(rows);
      HIP_OK(hipDeviceSynchronize());
      double t0 = Now();
      engine.ForwardVerifySpans(spans);
      if (rep) packed.push_back((Now() - t0) * 1e3);
      HIP_OK(hipDeviceSynchronize());
      if (rep) fetched.push_back(LookaheadFetchedMB(lookahead));
      double sum = 0;
      for (const auto& span : spans) {
        restore_all();
        t0 = Now();
        engine.ForwardVerifySpans({span});
        sum += (Now() - t0) * 1e3;
      }
      if (rep) lone.push_back(sum);
    }
    std::printf("%s,%d,%.3f,%.3f,%.2f,%s,%.1f\n", name.c_str(), rows, Median(packed), Median(lone), experts,
                exact ? "PASS" : "FAIL", Median(fetched));
    std::fflush(stdout);
  }
  sampler.Stop();
  std::printf("run %s\n", sampler.Summary().c_str());
  const auto [checks, mismatches] = EngineStateIO::QsaChecks(engine);
  std::printf("qsa_checks=%ld qsa_mismatches=%ld\n", checks, mismatches);
  return fails || mismatches ? 1 : 0;
}
