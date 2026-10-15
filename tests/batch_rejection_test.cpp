// CPU checks for the exact boundary between a rejected Open and uncertain device state.
#include "../engine/serve/batch_session.h"
#include <cassert>
#include <map>

struct RejectionEngine {
  int selected = 0, writes = 0;
  bool fail_reset = false;
  std::vector<int> capacities, last;
  std::map<int, std::vector<int>> histories;
  int Context() const { return 64; }
  int PrefillRows() const { return 8; }
  void ConfigureServeStreams(int count, uint64_t, int short_count, int short_ctx) {
    capacities.assign(size_t(count), Context());
    for (int slot = count - short_count; slot < count; ++slot) capacities[size_t(slot)] = short_ctx;
  }
  int StreamCapacity(int slot) const { return capacities.at(size_t(slot)); }
  void SelectStream(int slot) { ++writes; selected = slot; }
  void Reset() {
    ++writes;
    if (fail_reset) throw std::runtime_error("injected device failure");
    histories[selected].clear();
  }
  std::vector<int> Forward(const int* ids, int rows, int, int) {
    ++writes; last.clear();
    for (int row = 0; row < rows; ++row) {
      histories[selected].push_back(ids[row]);
      last.push_back(ids[row] + 1);
    }
    return last;
  }
  std::vector<int> Prefill(const int* ids, int rows, const std::string&, bool predict,
                           const crucible::PrefillBetween* = nullptr, int = 0) {
    auto output = Forward(ids, rows, 0, rows);
    return predict ? std::vector<int>{output.back()} : std::vector<int>{};
  }
  // Interleaved prefill instantiates the packed verify path; these tests never reach it.
  std::vector<int> ForwardVerifySpans(const std::vector<crucible::VerifySpan>&) {
    throw std::logic_error("rejection tests run no packed verification");
  }
  std::vector<int> ForwardBatch(const std::vector<int>& ids, const std::vector<int>& slots) {
    ++writes; last.clear();
    for (size_t row = 0; row < ids.size(); ++row) {
      histories[slots[row]].push_back(ids[row]);
      last.push_back(ids[row] + 1);
    }
    return last;
  }
  std::vector<float> CopyLogits(int, int = 0) const { return std::vector<float>(16, 0); }
};
using Session = crucible::BatchSession<RejectionEngine>;

crucible::ServeRequest Request(int count = 8) {
  crucible::ServeRequest request;
  request.ids = {1}; request.max_tokens = count; request.spec = 0;
  return request;
}

void Rejected(Session& session, RejectionEngine& engine, const crucible::ServeRequest& request) {
  session.SettlePrompt();  // as the serve loop does after each command: the peer's prompt end is done
  const auto before = engine.histories;
  const int writes = engine.writes;
  const size_t active = session.Active();
  try { session.Open(99, request, true); assert(false); }
  catch (const crucible::RecoverableOpenError& error) {
    assert(crucible::IsRecoverableBatchError(error, "open", session.Poisoned()));
    assert(!crucible::IsRecoverableBatchError(error, "step", false));
    assert(!crucible::IsRecoverableBatchError(error, "open", true));
  }
  assert(!session.Poisoned() && session.Active() == active);
  assert(engine.writes == writes && engine.histories == before);
  assert(session.Step({1})[0].token == 3);
  const auto lease = session.Open(100, Request(2), true);
  assert(lease[0].slot_generation == 1);
  session.Close(100);
}

void PolicyRejections() {
  // Site 1: the fixed speculative policy rejects before SelectStream or any lease mutation.
  {
    RejectionEngine engine; Session session(engine, 2);
    session.Open(1, Request());
    auto bad = Request(); bad.spec = 7;
    Rejected(session, engine, bad);  // no speculative hooks
  }
  {
    RejectionEngine engine; Session session(engine, 2);
    session.Open(1, Request());
    crucible::BatchSpecHooks hooks;
    hooks.check = [](const crucible::ServeRequest&) { return std::string("invalid fixed policy"); };
    session.SetSpecHooks(hooks);
    auto bad = Request(); bad.spec = 7;
    Rejected(session, engine, bad);
  }
}

void RequestRejections() {
  // Site 2: token, stop, sampling, and context validation only reads the request.
  for (int kind = 0; kind < 4; ++kind) {
    RejectionEngine engine; Session session(engine, 2);
    session.Open(1, Request());
    auto bad = Request();
    if (kind == 0) bad.ids = {-1};
    if (kind == 1) bad.stop_ids = {crucible::kVocab};
    if (kind == 2) bad.max_tokens = 65;
    if (kind == 3) bad.sampling.temperature = -1;
    Rejected(session, engine, bad);
  }
}

void SlotFitRejection() {
  // Site 3: a free short slot cannot hold a long request; the existing full-slot peer keeps its state.
  {
    RejectionEngine engine; Session session(engine, 2, 0, 1, 16);
    session.Open(1, Request(32));
    Rejected(session, engine, Request(32));
    const auto next = session.Open(99, Request(2), true);
    assert(next[0].slot_context == 16 && next[0].slot_generation == 2);
  }
}

void UnsafeErrors() {
  // ID/slot synchronization failures stay unsafe, even though they precede device work.
  {
    RejectionEngine engine; Session session(engine, 1);
    session.Open(1, Request());
    for (uint64_t id : {uint64_t(0), uint64_t(1), uint64_t(2)}) {
      try { session.Open(id, Request(), true); assert(false); }
      catch (const std::exception& error) {
        assert(!crucible::IsRecoverableBatchError(error, "open", session.Poisoned()));
      }
    }
  }
  // A reset failure can modify device state without publishing opened, so its boundary is not sufficient.
  {
    RejectionEngine engine; Session session(engine, 2);
    session.Open(1, Request());
    engine.fail_reset = true;
    try { session.Open(2, Request(), true); assert(false); }
    catch (const std::exception& error) {
      assert(session.Poisoned());
      assert(!crucible::IsRecoverableBatchError(error, "open", session.Poisoned()));
    }
  }
  const std::runtime_error unknown("unknown failure");
  assert(!crucible::IsRecoverableBatchError(unknown, "open", false));
}

int main() {
  PolicyRejections();
  RequestRejections();
  SlotFitRejection();
  UnsafeErrors();
}
