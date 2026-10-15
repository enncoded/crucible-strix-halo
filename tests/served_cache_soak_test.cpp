// The Oct 5 served-cache incident on the CPU: the served layout's batch cache, measured snapshot sizes, and the
// host memory those snapshots pin, driven by the incident's prompt sequence (numbers only, no content).
// In the served build: once the 10 GiB cache was full, the 8 GiB host check ran before eviction, so every
// frontier above about 62K tokens was refused and one session re-prefilled up to 166K tokens per turn.
#include "../engine/serve/batch_conversation_cache.h"
#include <cassert>
#include <cstdio>
#include <map>
using namespace crucible;
struct Snapshot { std::vector<int> hist; ConversationResume resume; };

namespace {
constexpr int kRows = 4096;                                  // --prefill 4096
constexpr size_t kBudget = size_t(10240) << 20;              // --conversation-cache-mib 10240
// MemAvailable at 08:16, after load and before the first request (20.06 GiB). Snapshots are
// pinned host memory, so MemAvailable falls by the cache's bytes (20.06 to 9.6 GiB by 08:23 in the incident).
constexpr uint64_t kIdleAvailable = uint64_t(20.06 * 1073741824.0);
// Least-squares fit of 14 measured snapshot sizes (8K to 109K tokens, development measurements): 89 KB max error.
size_t SnapshotBytes(size_t tokens) { return size_t(152292328 + 32264 * tokens); }
constexpr size_t kTemplateTail = 4;  // "<|im_start|>assistant\n<think>\n" after cache_prefix_tokens
constexpr size_t kSharedPrefix = 11000;  // system and tools block, below the first user turn

struct Turn { int session; size_t prompt, completion; };
struct Result { size_t cached, prefilled, appended; };

class ServedCache {
 public:
  explicit ServedCache(bool replay) : replay_(replay) { policy_.prefill_rows = kRows; }
  // One request: restore, prefill in 4096-row chunks with frontier captures, then (replay) the generated tail.
  Result Run(const Turn& turn) {
    const auto ids = Stream(turn.session, turn.prompt);
    size_t cached = 0;
    bool from_tail = false;
    if (replay_)
      if (const auto tail = cache_.FindTail(ids, policy_)) cached = tail->snapshot->hist.size(), from_tail = true;
    if (!cached)
      if (const auto entry = cache_.Find(ids, policy_, ids.size() - kTemplateTail, ids.size() - 1, kRows))
        cached = entry->snapshot->hist.size();
    // As BatchSession: a frontier restore keeps the 4096-row grid, a replayed turn captures no prompt frontiers.
    const auto frontiers = BatchCacheFrontiers(ids.size(), ids.size() - kTemplateTail, kRows, kSharedPrefix);
    for (size_t cursor = cached + kRows; !from_tail && cursor < ids.size(); cursor += kRows)
      for (const auto& frontier : frontiers)
        if (frontier.position == cursor) Capture(ids, frontier);
    if (replay_) Tail(turn.session, turn.prompt + turn.completion - 1, from_tail);
    const size_t seen = seen_[turn.session];
    seen_[turn.session] = replay_ ? turn.prompt + turn.completion - 1 : turn.prompt;
    return {cached, turn.prompt - cached, turn.prompt - std::min(seen, turn.prompt)};
  }
  size_t refused = 0;

 private:
  static std::vector<int> Stream(int session, size_t length) {
    std::vector<int> ids(length);
    for (size_t i = 0; i < length; ++i) ids[i] = session * 1000000 + int(i);
    return ids;
  }
  uint64_t Available() const { return kIdleAvailable - cache_.Bytes(); }
  void Capture(const std::vector<int>& ids, const CacheFrontier& frontier) {
    const size_t position = frontier.position;
    refused += !cache_.Capture(ids, policy_, frontier, SnapshotBytes(position), Available(), [&] {
      auto s = std::make_shared<Snapshot>();
      s->hist.assign(ids.begin(), ids.begin() + position);
      s->resume.length = position; s->resume.prompt_only = true; s->resume.prompt_boundary = ids.size() - kTemplateTail;
      return std::shared_ptr<const Snapshot>(s);
    });
  }
  // As BatchPrefixCache: a tail of a replayed turn without reasoning is trusted.
  void Tail(int session, size_t length, bool trusted) {
    const auto hist = Stream(session, length);
    refused += !cache_.CaptureTail(hist, policy_, SnapshotBytes(length), Available(), [&] {
      auto s = std::make_shared<Snapshot>();
      s->hist = hist; s->resume.length = length;
      return std::shared_ptr<const Snapshot>(s);
    }, trusted);
  }
  const bool replay_;
  ConversationPolicy policy_;
  BatchConversationCache<Snapshot> cache_{kBudget};
  std::map<int, size_t> seen_;
};

// Requests from 08:16 to 08:39 in arrival order (prompt tokens only). Session 2 is the long agent session; 1 and 3 are
// shorter sessions that filled the cache first. Completions are not recorded there; 400 tokens stand in for them.
std::vector<Turn> IncidentTurns() {
  const std::vector<std::pair<int, size_t>> prompts{
      {1, 12341}, {3, 12492}, {1, 13031}, {1, 13466}, {1, 14213}, {1, 19765}, {1, 20565}, {1, 22258},
      {2, 12834}, {2, 17510}, {2, 22749}, {2, 23468}, {2, 24994}, {2, 25558}, {2, 27451}, {2, 29404},
      {2, 31016}, {2, 31607}, {2, 32332}, {2, 33560}, {2, 34299}, {2, 54752}, {2, 74721}, {2, 94222},
      {2, 113374}, {2, 134864}, {2, 155379}, {2, 179809}, {2, 183834}, {2, 187552}, {2, 206318}, {2, 209228},
      {2, 219071}};
  std::vector<Turn> turns;
  for (const auto& [session, prompt] : prompts) turns.push_back({session, prompt, 400});
  return turns;
}

// tools/agent_soak.py multi: the long session to 220K beside two short sessions to 40K, round robin. The system
// and tools block (about 12.2K tokens) is shared, as for Pi agents with one setup.
std::vector<Turn> MultiTurns() {
  const std::vector<size_t> steps{690, 435, 750, 5550, 800, 1700, 4700, 5250, 720, 1500, 560, 1900, 1950, 1600, 600,
                                  700, 1250, 740, 20450, 19950, 19500, 19150, 21500, 20500, 24400, 4000, 3700, 18750,
                                  2900, 9850};
  std::map<int, size_t> prompt{{1, 12200}, {2, 12200}, {3, 12200}};
  std::map<int, size_t> step{{1, 0}, {2, 0}, {3, 0}};
  std::vector<Turn> turns;
  for (bool any = true; any;) {
    any = false;
    for (int session : {1, 2, 3}) {
      const size_t limit = session == 1 ? 220000 : 40000;
      if (prompt[session] > limit) continue;
      turns.push_back({session, prompt[session], 120});
      any = true;
      size_t next = steps[step[session]++ % steps.size()];
      if (session != 1) while (next >= 10000) next = steps[step[session]++ % steps.size()];
      prompt[session] += 120 + next;
    }
  }
  return turns;
}

// The soak rule (tools/agent_soak.py): a turn re-prefills at most its appended tokens plus one chunk, or plus the
// 8-row decode-path head chunk with session replay.
bool Soak(bool replay, bool print, const std::vector<Turn>& turns = IncidentTurns()) {
  ServedCache cache(replay);
  bool pass = true;
  for (const auto& turn : turns) {
    const auto r = cache.Run(turn);
    const size_t margin = replay ? kMaxRows : kRows;
    const bool ok = !r.appended || r.prefilled <= r.appended + margin;
    pass = pass && ok;
    if (print || !ok)
      std::printf("replay=%d session=%d prompt=%zu cached=%zu prefilled=%zu appended=%zu %s\n", int(replay),
                  turn.session, turn.prompt, r.cached, r.prefilled, r.appended, ok ? "ok" : "FAIL");
  }
  std::printf("replay=%d refused=%zu %s\n", int(replay), cache.refused, pass ? "PASS" : "FAIL");
  return pass;
}
// The Oct 6 multi-session failure (decode worker, crucible-t12 h2h, results/h2h requests 59-70): four sessions share a
// 31K history, then run code, tool, code_sampled, think and prose turns at the same time. The think turn reaches
// the answer room after one token: the server cancels the first job and runs a continuation job, which saves a tail
// that holds the reasoning. The client keeps only the answer, so the next prompt repeats the prompt and not that
// tail. On crucible-t12 the continuation tails superseded each session's resume point, the next turn of all four
// sessions missed, and each re-prefilled 33K tokens.
class AgentCache {
 public:
  struct Job { std::vector<int> ids; size_t boundary = 0, cached = 0; bool from_tail = false; };
  explicit AgentCache(bool replay) : replay_(replay) { policy_.prefill_rows = kRows; }
  // Restore, then prefill in 4096-row chunks with the frontier captures of BatchSession.
  Job Open(std::vector<int> ids, size_t boundary) {
    Job job{std::move(ids), boundary};
    const auto& r = job.ids;
    if (replay_)
      if (const auto tail = cache_.FindTail(r, policy_)) job.cached = tail->snapshot->hist.size(), job.from_tail = true;
    if (!job.cached)
      if (const auto entry = cache_.Find(r, policy_, boundary, r.size() - 1, kRows)) job.cached = entry->snapshot->hist.size();
    const auto frontiers = BatchCacheFrontiers(r.size(), boundary, kRows, 0);
    for (size_t cursor = job.cached + kRows; !job.from_tail && cursor < r.size(); cursor += kRows)
      for (const auto& frontier : frontiers)
        if (frontier.position == cursor) Capture(r, boundary, frontier);
    return job;
  }
  // A finished job keeps its prompt and every output token but the last (ConversationTailState). The server guards
  // a continuation job and a turn with reasoning: their tails are never trusted.
  void Tail(const Job& job, const std::vector<int>& generated, bool guard) {
    if (!replay_ || generated.size() < 2) return;
    std::vector<int> hist(job.ids);
    hist.insert(hist.end(), generated.begin(), generated.end() - 1);
    refused += !cache_.CaptureTail(hist, policy_, SnapshotBytes(hist.size()), Available(), [&] {
      auto s = std::make_shared<Snapshot>();
      s->hist = hist; s->resume.length = hist.size();
      return std::shared_ptr<const Snapshot>(s);
    }, job.from_tail && !guard);
  }
  size_t refused = 0;

 private:
  uint64_t Available() const { return kIdleAvailable - cache_.Bytes(); }
  void Capture(const std::vector<int>& ids, size_t boundary, const CacheFrontier& frontier) {
    const size_t position = frontier.position;
    refused += !cache_.Capture(ids, policy_, frontier, SnapshotBytes(position), Available(), [&] {
      auto s = std::make_shared<Snapshot>();
      s->hist.assign(ids.begin(), ids.begin() + position);
      s->resume.length = position; s->resume.prompt_only = true; s->resume.prompt_boundary = boundary;
      return std::shared_ptr<const Snapshot>(s);
    });
  }
  const bool replay_;
  ConversationPolicy policy_;
  BatchConversationCache<Snapshot> cache_{kBudget};
};

struct Workload { const char* name; size_t completion; bool think; size_t tool_result; };

// tools/h2h_decode.py: each turn appends a user request; a tool turn also appends a short tool result. The think turn
// sends max_tokens 768, so the answer room (1024) closes the think block after one token.
bool AgentSoak(int sessions, size_t base_tokens, int rounds, bool print) {
  constexpr int kGenPrompt[] = {7, 8, 9, 10};  // "<|im_start|>assistant\n<think>\n"
  constexpr int kImEnd = 11;
  constexpr size_t kUser = 60, kClose = 25, kCap = 1;
  const Workload workloads[] = {{"code", 512, false, 0}, {"tool", 400, false, 120}, {"code_sampled", 512, false, 0},
                                {"think", 768, true, 0}, {"prose", 512, false, 0}};
  int fresh = 1 << 24;
  const auto Fresh = [&](size_t count) { std::vector<int> ids(count); for (auto& id : ids) id = fresh++; return ids; };
  const auto Join = [](std::vector<int> a, const std::vector<int>& b) { a.insert(a.end(), b.begin(), b.end()); return a; };
  AgentCache cache(true);
  std::vector<int> base(base_tokens);
  for (size_t i = 0; i < base_tokens; ++i) base[i] = int(i) + 100;
  std::vector<std::vector<int>> history(sessions);
  std::vector<size_t> previous(sessions, 0);
  bool pass = true;
  const auto Prompt = [&](int s) {
    auto ids = Join(history[s], Fresh(kUser));
    ids.insert(ids.end(), std::begin(kGenPrompt), std::end(kGenPrompt));
    return ids;
  };
  // Warm-up: each session sends its first prompt alone with max_tokens 1, so the shared frontiers exist.
  for (int s = 0; s < sessions; ++s) {
    history[s] = Join(base, Fresh(kUser));
    auto ids = Join(history[s], {7, 8, 9, 10});
    cache.Open(ids, ids.size() - 4);
  }
  for (int round = 0; round < rounds; ++round)
    for (const auto& work : workloads) {
      // All sessions open, prefill and decode at the same time; tails are saved as each job finishes.
      std::vector<AgentCache::Job> first(sessions), cont(sessions);
      std::vector<std::vector<int>> output(sessions);
      for (int s = 0; s < sessions; ++s) {
        auto ids = Prompt(s);
        const size_t boundary = ids.size() - 4;
        first[s] = cache.Open(std::move(ids), boundary);
      }
      for (int s = 0; s < sessions; ++s) {
        if (!work.think) { output[s] = Fresh(work.completion); continue; }
        // http.py close_thinking: cancel the first job (no tail), then continue on prompt + output + close.
        output[s] = Join(Fresh(kCap), Fresh(kClose));
        cont[s] = cache.Open(Join(first[s].ids, output[s]), first[s].boundary);
      }
      for (int s = 0; s < sessions; ++s) {
        if (!work.think) { cache.Tail(first[s], output[s], false); continue; }
        const auto answer = Fresh(work.completion - kCap - kClose);
        cache.Tail(cont[s], answer, true);
        output[s] = Join(output[s], answer);
      }
      for (int s = 0; s < sessions; ++s) {
        const auto& job = first[s];
        const size_t prefilled = job.ids.size() - job.cached, appended = job.ids.size() - previous[s];
        const bool ok = !previous[s] || prefilled <= appended + kRows;
        pass = pass && ok;
        if (print || !ok)
          std::printf("agents=%d base=%zu %s session=%d prompt=%zu cached=%zu prefilled=%zu appended=%zu %s\n", sessions,
                      base_tokens, work.name, s, job.ids.size(), job.cached, prefilled, appended, ok ? "ok" : "FAIL");
        previous[s] = job.ids.size();
        // The client repeats the answer. Without thinking it repeats every output token (replay hits); after a
        // think turn it renders only the answer, so the prompt differs from the output after the header.
        history[s] = work.think ? Join(Join(std::vector<int>(job.ids.begin(), job.ids.begin() + job.boundary), {7, 8}),
                                       Fresh(work.completion / 2))
                                : Join(job.ids, output[s]);
        history[s].push_back(kImEnd);
        if (work.tool_result) history[s] = Join(history[s], Fresh(work.tool_result));
      }
    }
  std::printf("agents=%d base=%zu refused=%zu %s\n", sessions, base_tokens, cache.refused, pass ? "PASS" : "FAIL");
  return pass;
}
}  // namespace

int main(int argc, char**) {
  const bool print = argc > 1;
  const bool frontiers = Soak(false, print), replay = Soak(true, print);
  const bool multi = Soak(false, print, MultiTurns()), multi_replay = Soak(true, print, MultiTurns());
  // The h2h layout of Oct 6: --streams 4 --ctx 131072 --conversation-cache-mib 10240, a 31K shared history. At 60K
  // the speculative tails no longer fit beside the four resume points, so they must not push those out. At 75K the
  // four resume points alone overflow 10 GiB: that needs the slot-resident resume (BatchSession), not this cache.
  const bool agents = AgentSoak(4, 31100, 2, print), served = AgentSoak(3, 31100, 2, print);
  const bool crowded = AgentSoak(4, 60000, 2, print);
  std::fflush(stdout);
  const bool pass = frontiers && replay && multi && multi_replay && agents && served && crowded;
  assert(pass);
  return pass ? 0 : 1;
}
