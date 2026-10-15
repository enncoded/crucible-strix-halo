// Greedy policy-switch probe (G6b, and packed verification, G6d). For each prompt, compares the consumed target
// logits and IDs of three batch paths: solo speculative steps (spec=7), solo spec=0 steps, and spec=0 rows in
// a packed pass with two peers. Equal bits at every position allow the scheduler to change a greedy
// request's depth or speculation by load without changing its output.
// Usage: batch_switch_probe CKPT TOKENS PROMPT_FILE...   (whitespace-separated token IDs per file)
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <cstring>
#include <fstream>

namespace {
struct Trace {
  std::vector<int> ids;
  std::vector<std::vector<float>> logits;
};

std::vector<int> ReadIds(const char* path) {
  std::ifstream file(path);
  std::vector<int> ids;
  for (int id; file >> id;) ids.push_back(id);
  if (ids.empty()) throw std::runtime_error(std::string("empty prompt file: ") + path);
  return ids;
}

crucible::ServeRequest Request(const std::vector<int>& ids, int tokens, int spec) {
  crucible::ServeRequest r;
  r.ids = ids; r.max_tokens = tokens; r.spec = spec; r.draft_adapt = 0; r.stop_ids.clear();
  return r;
}

// Appends the logits row of each published token after a command.
void Record(Engine& engine, Trace& trace, const std::vector<crucible::BatchEvent>& events, uint64_t id, int row) {
  int count = 0;
  for (const auto& event : events)
    if (event.request_id == id && event.token) { trace.ids.push_back(*event.token); ++count; }
  if (!count) return;
  const auto rows = engine.CopyLogits(count, row);
  for (int i = 0; i < count; ++i)
    trace.logits.emplace_back(rows.begin() + size_t(i) * kVocab, rows.begin() + size_t(i + 1) * kVocab);
}

bool Done(const std::vector<crucible::BatchEvent>& events, uint64_t id) {
  for (const auto& event : events) if (event.request_id == id && event.done) return true;
  return false;
}

std::vector<crucible::BatchEvent> PrefillAll(crucible::BatchSession<Engine>& session, uint64_t id) {
  for (;;) {
    auto events = session.Prefill(id);
    if (!events.empty()) return events;
  }
}

Trace Solo(Engine& engine, crucible::BatchSession<Engine>& session, uint64_t id, const crucible::ServeRequest& r) {
  Trace trace;
  session.Open(id, r, true);
  auto events = PrefillAll(session, id);
  Record(engine, trace, events, id, 0);
  while (!Done(events, id)) {
    events = r.spec ? session.SpecStep(id) : session.Step({id});
    Record(engine, trace, events, id, 0);
  }
  return trace;
}

Trace Packed(Engine& engine, crucible::BatchSession<Engine>& session, uint64_t id, const crucible::ServeRequest& r,
             const std::vector<crucible::ServeRequest>& peers, uint64_t first_peer) {
  Trace trace;
  std::vector<uint64_t> live;
  for (size_t index = 0; index < peers.size(); ++index) {
    session.Open(first_peer + index, peers[index], true);
    PrefillAll(session, first_peer + index);
    live.push_back(first_peer + index);
  }
  session.Open(id, r, true);
  auto events = PrefillAll(session, id);
  Record(engine, trace, events, id, 0);
  // The probed request takes the middle row, so its row has peers on both sides.
  live.insert(live.begin() + 1, id);
  while (!Done(events, id)) {
    const auto row = int(std::find(live.begin(), live.end(), id) - live.begin());
    events = session.Step(live);
    Record(engine, trace, events, id, row);
    for (const auto& event : events)
      if (event.done) live.erase(std::find(live.begin(), live.end(), event.request_id));
  }
  for (const uint64_t peer : live) session.Close(peer);
  return trace;
}

// The request in a packed verification group (G6d) with two speculative peers and one plain peer. Its rows sit
// at its span offset in the shared pass, so the accept hook captures them before any later draft work.
Trace Verify(Engine& engine, crucible::BatchSession<Engine>& session, crucible::BatchSpecHooks hooks, uint64_t id,
             const crucible::ServeRequest& r, const std::vector<crucible::ServeRequest>& peers, uint64_t first_peer) {
  std::vector<float> rows;
  const auto accept = hooks.accept;
  hooks.accept = [&](uint64_t request, int first, std::span<const int> greedy) {
    if (request == id) rows = engine.CopyLogits(int(greedy.size()), first);
    return accept(request, first, greedy);
  };
  session.SetSpecHooks(hooks);
  Trace trace;
  std::vector<uint64_t> live;
  for (size_t index = 0; index < peers.size(); ++index) {
    session.Open(first_peer + index, peers[index], true);
    PrefillAll(session, first_peer + index);
    live.push_back(first_peer + index);
  }
  session.Open(id, r, true);
  auto events = PrefillAll(session, id);
  Record(engine, trace, events, id, 0);
  live.insert(live.begin() + 1, id);
  while (!Done(events, id)) {
    events = session.VerifyStep(live);
    size_t count = 0;
    for (const auto& event : events)
      if (event.request_id == id && event.token) { trace.ids.push_back(*event.token); ++count; }
    for (size_t i = 0; i < count; ++i)
      trace.logits.emplace_back(rows.begin() + i * kVocab, rows.begin() + (i + 1) * kVocab);
    for (const auto& event : events)
      if (event.done) live.erase(std::find(live.begin(), live.end(), event.request_id));
  }
  for (const uint64_t peer : live) session.Close(peer);
  return trace;
}

struct Diff { bool ids = true; size_t first_id = 0, logit_rows = 0, first_logit = 0; double max_abs = 0; };

// Row 0 is the prompt prediction. After a speculative prefill, MTP pairing has reused the logits buffer,
// so the probe cannot read it back; both paths take that token from the same Prefill call.
Diff Compare(const Trace& a, const Trace& b, size_t first_row = 0) {
  Diff diff;
  diff.first_id = diff.first_logit = a.ids.size();
  if (a.ids.size() != b.ids.size()) { diff.ids = false; diff.first_id = 0; }
  for (size_t i = 0; i < std::min(a.ids.size(), b.ids.size()); ++i) {
    if (a.ids[i] != b.ids[i] && diff.ids) { diff.ids = false; diff.first_id = i; }
    if (i >= first_row && std::memcmp(a.logits[i].data(), b.logits[i].data(), sizeof(float) * kVocab)) {
      if (!diff.logit_rows) diff.first_logit = i;
      ++diff.logit_rows;
      for (int v = 0; v < kVocab; ++v) diff.max_abs = std::max(diff.max_abs, double(std::fabs(a.logits[i][v] - b.logits[i][v])));
    }
  }
  return diff;
}

void PrintDiff(const char* name, const Diff& d, bool last) {
  std::printf("\"%s\":{\"ids_equal\":%s,\"first_id_diff\":%zu,\"logit_rows_differ\":%zu,\"first_logit_diff\":%zu,"
              "\"max_abs\":%.9g}%s", name, d.ids ? "true" : "false", d.first_id, d.logit_rows, d.first_logit,
              d.max_abs, last ? "" : ",");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: batch_switch_probe CKPT TOKENS PROMPT_FILE...\n"); return 2; }
  const int tokens = std::stoi(argv[2]);
  std::vector<std::vector<int>> prompts;
  for (int index = 3; index < argc; ++index) prompts.push_back(ReadIds(argv[index]));
  if (prompts.size() < 3) { std::fprintf(stderr, "need at least three prompts (two serve as peers)\n"); return 2; }
  Engine engine(argv[1], 16384, kPrefillAuto);
  crucible::BatchSession<Engine> session(engine, 4);
  static_assert(kMaxRows == 8);
  BatchSpecDecoder spec(engine);
  const auto hooks = spec.Hooks();
  session.SetSpecHooks(hooks);
  engine.WarmPlans();
  HIP_OK(hipDeviceSynchronize());
  bool pass = true;
  uint64_t id = 1;
  std::printf("{\"tokens\":%d,\"prompts\":[", tokens);
  for (size_t p = 0; p < prompts.size(); ++p) {
    const auto& prompt = prompts[p];
    const Trace speculative = Solo(engine, session, id++, Request(prompt, tokens, 7));
    const Trace plain = Solo(engine, session, id++, Request(prompt, tokens, 0));
    std::vector<crucible::ServeRequest> peers{Request(prompts[(p + 1) % prompts.size()], tokens + 16, 0),
                                              Request(prompts[(p + 2) % prompts.size()], tokens + 16, 0)};
    const Trace packed = Packed(engine, session, id, Request(prompt, tokens, 0), peers, id + 1);
    id += 3;
    std::vector<crucible::ServeRequest> verify_peers{Request(prompts[(p + 1) % prompts.size()], tokens + 16, 7),
                                                     Request(prompts[(p + 2) % prompts.size()], tokens + 16, 0),
                                                     Request(prompts[(p + 2) % prompts.size()], tokens + 24, 7)};
    const Trace verified = Verify(engine, session, hooks, id, Request(prompt, tokens, 7), verify_peers, id + 1);
    session.SetSpecHooks(hooks);
    id += 4;
    const Diff a = Compare(speculative, plain, 1), b = Compare(plain, packed), c = Compare(speculative, verified, 1);
    pass = pass && a.ids && b.ids && c.ids && !a.logit_rows && !b.logit_rows && !c.logit_rows &&
           int(speculative.ids.size()) == tokens;
    std::printf("%s{\"length\":%zu,", p ? "," : "", prompt.size());
    PrintDiff("spec7_vs_spec0", a, false);
    PrintDiff("spec0_vs_packed", b, false);
    PrintDiff("spec7_vs_verify", c, true);
    std::printf("}");
  }
  std::printf("],\"status\":\"%s\"}\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
