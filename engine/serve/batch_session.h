// One caller owns the engine. The frontend selects ready requests without blocking peers.
#pragma once
#include "../runtime/slot_checkpoints.h"
#include "serve_protocol.h"
#include "../runtime/batch_spec_contract.h"
#include "cache_status.h"
#include "../runtime/prefill_interleave.h"
#include "../model/flashnext/prefill_launch.h"
#include "../runtime/draft_warmup.h"
#include "../runtime/packed_rows_option.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <limits>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace crucible {
// Only pre-device Open validation uses this type. All other failures keep fail-closed semantics.
class RecoverableOpenError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};
inline bool IsRecoverableBatchError(const std::exception& error, std::string_view op, bool poisoned) {
  return op == "open" && !poisoned && dynamic_cast<const RecoverableOpenError*>(&error) != nullptr;
}

// A speculative request's draft counters over its whole life (BatchSpecDecoder): drafts proposed and accepted, and
// draft-and-verify steps. The done event carries them so the server can log them per turn.
struct BatchDraftStats {
  long proposed = 0, accepted = 0, steps = 0;
};

struct BatchEvent {
  uint64_t request_id = 0, slot_generation = 0;
  std::optional<int> token;
  bool done = false, stopped = false;
  size_t prompt_tokens = 0, completion_tokens = 0;
  double prefill_seconds = 0, decode_seconds = 0;
  size_t cached_tokens = 0, cache_bytes = 0;
  double restore_seconds = 0, capture_seconds = 0;
  bool cache_shared = false;
  int slot_context = 0;  // set on the opened event
  bool cache_tail = false;  // the restore was a previous turn's generated tail (session replay, G15)
  bool slot_resume = false; // that tail was still on the request's slot: no host restore
  bool checkpoint_resume = false;  // the restore was an in-slot checkpoint (slot_checkpoints.h)
  int tail_replay = 0;      // that tail's pending GDN rows (evidence for the replay gate)
  // Opened events of image requests: the prefill commands that run encoder slices before the first prompt chunk.
  std::optional<int> vision_slices;
  // A request that ended without a token because an input check failed after its open (an image file that changed).
  // The text is a fixed engine message; the request is closed and its slot is free.
  std::optional<std::string> error;
  // Done events only, when the status hook is set: this request's largest saved prompt frontier, its saved tail,
  // its last refused save, and the cache after the tail save (crucible_turn).
  size_t cache_saved_tokens = 0, cache_tail_tokens = 0, cache_entries = 0, cache_evictions = 0;
  std::string cache_save_refused;
  std::optional<uint64_t> host_available;
  bool cache_reported = false;  // the fields above are set
  std::optional<BatchDraftStats> draft;  // done events of speculative requests, when the spec hooks report them
};

// The done event's draft counters (JSON members with a leading comma); empty for a request without speculation, so
// its done line stays byte-identical.
inline std::string BatchDraftFields(const BatchEvent& event) {
  if (!event.done || !event.draft) return {};
  return ",\"draft_proposed\":" + std::to_string(event.draft->proposed) +
         ",\"draft_accepted\":" + std::to_string(event.draft->accepted) +
         ",\"spec_steps\":" + std::to_string(event.draft->steps);
}

// The done event's cache fields (JSON members with a leading comma); empty when the cache did not report.
inline std::string BatchCacheFields(const BatchEvent& event) {
  if (!event.done || !event.cache_reported) return {};
  std::string out = ",\"cache_saved_tokens\":" + std::to_string(event.cache_saved_tokens) +
                    ",\"cache_tail_tokens\":" + std::to_string(event.cache_tail_tokens) +
                    ",\"cache_entries\":" + std::to_string(event.cache_entries) +
                    ",\"cache_evictions\":" + std::to_string(event.cache_evictions);
  // Reasons are fixed identifiers (cache_status.h); anything else is not printed into the JSON line.
  if (!event.cache_save_refused.empty() &&
      std::all_of(event.cache_save_refused.begin(), event.cache_save_refused.end(),
                  [](char c) { return c >= 'a' && c <= 'z'; }))
    out += ",\"cache_save_refused\":\"" + event.cache_save_refused + "\"";
  if (event.host_available) out += ",\"host_free_mib\":" + std::to_string(*event.host_available >> 20);
  if (event.slot_resume) out += ",\"slot_resume\":true";
  if (event.checkpoint_resume) out += ",\"checkpoint_resume\":true";
  return out;
}

// A restore result. `shared` marks a hit on a system-and-tools frontier (G7).
struct BatchRestore {
  size_t tokens = 0;
  bool shared = false;
  // Session replay (G15): a generated tail can carry verified rows that still wait for GDN replay.
  int replay = 0;
  bool tail = false;
  bool slot = false;  // the slot already holds this state; nothing was copied
  bool guarded = false;  // the tail came from a guarded request (its output held reasoning)
  BatchRestore(size_t tokens = 0, bool shared = false, int replay = 0, bool tail = false, bool slot = false,
               bool guarded = false)
      : tokens(tokens), shared(shared), replay(replay), tail(tail), slot(slot), guarded(guarded) {}
};
// The tail that a finished request left on its slot (session replay): its length, pending GDN rows, and whether
// its request was guarded.
struct BatchResident { size_t length = 0; int replay = 0; bool guarded = false; };
// A slot resume restores a host tail instead only when that tail is longer, or has the same tokens with other
// pending rows; otherwise the slot state is the state the restore would write.
inline bool HostTailFirst(size_t length, int replay, const BatchResident& resident) {
  return length > resident.length || (length == resident.length && replay != resident.replay);
}
// The disk read for a request (state_disk.h): none (no longer state on disk, or no disk tier), reading, ready, or
// failed. `tokens`: the length of the disk state.
struct BatchStage { std::string state = "none"; size_t tokens = 0; };
// How a finished request started: from a previous tail (`replayed`), and whether that tail was guarded. `slot`: the
// slot that holds the finished state.
struct BatchTailSource { bool replayed = false, guarded = false; int slot = -1; };

// The output rows that an expected tail counts at most: the reserve stays near the usual tail, and a longer
// output pins its extra chunks at the save (about 300 rows per 8 MB chunk).
inline constexpr size_t kTailExpectRows = 4096;

// Worker-owned prefix cache callbacks (G2). Both run inside the device
// command of the request they serve, so a restore or capture never sits between a peer's launches.
struct BatchCacheHooks {
  // Restores the longest eligible prefix into the selected stream. Returns its length, 0 on a miss.
  std::function<BatchRestore(const ServeRequest&)> restore;
  // Runs after a non-final prompt chunk that ends at `cursor`. It may save that frontier.
  std::function<void(const ServeRequest&, size_t cursor)> capture;
  std::function<size_t()> bytes;
  // Session replay (G15, opt-in): runs on the finished request's selected stream at the end of the command that
  // published its last token, before any later command can reuse the slot. `source`: how the request started.
  // Returns the tail state that the slot now holds, if any.
  std::function<std::optional<BatchResident>(const ServeRequest&, std::span<const int> generated, int replay,
                                             BatchTailSource source)> tail;
  // Optional, with `tail`: the request on `slot` may save a tail of about `rows` rows when it ends (its open);
  // 0: no tail is due (after its tail save, or a close). No device work: the host pins the memory of that save while
  // the batch decodes, so the save on the device worker does not.
  std::function<void(const ServeRequest&, int slot, size_t rows)> expect;
  // Optional, with `tail`: the request repeats the tail that its slot still holds. Returns that slot state
  // (`slot` set, nothing copied) or a longer host state that it restored; both give the same output as a restore.
  std::function<BatchRestore(const ServeRequest&, BatchResident)> resume;
  // Optional, all three (slot_checkpoints.h): `checkpoint_save` copies the selected stream's fixed-size state into
  // buffer (slot, index); `checkpoint_load` writes it back over the slot, which still holds every row below it;
  // `peek` is the length that `restore` would give, without a change to the cache.
  std::function<void(int slot, int index)> checkpoint_save;
  std::function<void(int slot, int index, const SlotCheckpoint&)> checkpoint_load;
  std::function<size_t(const ServeRequest&)> peek;
  // Optional: a checkpoint resume left the slot's previous tail at `common` (the replay divergence line, counts only).
  std::function<void(size_t prompt, size_t restored, size_t tail, size_t common)> diverged;
  // Optional, with `resume`: the next request on `slot` overwrites the tail that the slot holds (`resumed`: it
  // continues from that tail). Runs on the selected slot before any write, so the host can save the state that
  // leaves the device ("One copy").
  std::function<void(int slot, bool resumed)> release;
  // Optional: the cache key of `tokens` (the request's prompt, or a prefix or extension of it). Slot resume and slot
  // checkpoints compare keys, not token IDs. Without the hook the key is RequestCacheKey: two images have the same
  // pad token IDs, so image tokens key by image (text keys equal the tokens).
  std::function<std::vector<int>(const ServeRequest&, std::span<const int> tokens)> key;
  // Optional, the disk tier (state_disk.h): start or report the read of a disk state longer than `floor` (what a
  // slot already holds for the prompt) and longer than the host cache's state. No device work.
  std::function<BatchStage(const ServeRequest&, size_t floor)> stage;
  // Optional: totals and the latest save try. A capture or tail hook that tried a save raises `attempts`.
  std::function<BatchCacheStatus()> status;
  // Optional, with `capture`: the first position after `cursor` that capture may save, or SIZE_MAX. Bulk launches that merge
  // grid chunks end there (PrefillLaunchRows). Without it, a worker with `capture` never merges chunks.
  std::function<size_t(const ServeRequest&, size_t cursor)> next_capture;
};

// Speculative decode callbacks (G6a). Each runs inside the command of the
// request it serves, after SelectStream, so per-request draft state never meets a peer's launches.
struct BatchSpecHooks {
  // Returns an empty string when this engine can decode the request's resolved policy exactly.
  std::function<std::string(const ServeRequest&)> check;
  std::function<void(uint64_t id, int slot, const ServeRequest&, size_t cached)> open;
  // Runs after each prompt chunk [start, start+rows); `bulk` marks the GEMM prefill path.
  std::function<void(uint64_t id, size_t start, int rows, bool bulk)> chunk;
  // Optional: the first row of chunk [start, start+rows) whose final trunk stacks `chunk` reads (rows if none).
  // The engine may skip the last layer's output for rows below it. Without it, every row is kept.
  std::function<int(uint64_t id, size_t start, int rows)> stacks;
  std::function<void(uint64_t id, int first)> first;
  // One solo draft-and-verify step. Returns the accepted tokens plus the correction (1 to 8 IDs).
  // `remaining` counts tokens still allowed; `position` is the absolute position of the next token (keyed law).
  std::function<std::vector<int>(uint64_t id, int last, size_t remaining, uint64_t position)> step;
  std::function<void(uint64_t id)> close;
  // Optional: the request's draft counters, read at its done event before `close` erases them.
  std::function<BatchDraftStats(uint64_t id)> stats;
  // Target rows that still wait for GDN replay (session replay saves them with the tail).
  std::function<int(uint64_t id)> replay;
  // Load-adaptive decode (G6c): a speculative request may join a packed step. `commit` runs first on the
  // request's stream. When verified rows still wait for GDN replay, it runs one solo plain row and returns its
  // greedy ID, because a packed pass needs committed state. `packed` then sees each such request's row of the
  // pass with the token it emitted, so the draft model can catch up before the next solo step.
  std::function<std::optional<int>(uint64_t id, int last)> commit;
  std::function<void(const std::vector<uint64_t>& ids, const std::vector<int>& rows, const std::vector<int>& tokens)> packed;
  // Packed verification (G6d). `propose` drafts up to `cap` tokens on the request's stream and returns its verify
  // rows (the last token first) with its pending replay count. After one shared pass, `accept` verifies the
  // request's rows [first, first + rows) and returns the emitted tokens; `settle` then runs deferred draft work.
  std::function<VerifySpan(uint64_t id, int last, size_t remaining, uint64_t position, int cap)> propose;
  std::function<std::vector<int>(uint64_t id, int first, std::span<const int> greedy)> accept;
  std::function<void()> settle;
  // Optional: drafts a request could copy from its own history now (prompt lookup), 0 if none. VerifyStep gives
  // spare rows to such requests first (DraftCapsFor). Like a cap, it changes speed only.
  std::function<int(uint64_t id, int last)> want;
  // Optional: propose for several requests at once (grouped MTP passes). Same contract as `propose` per request;
  // `last`, `remaining`, `position` and `cap` are per request, in the order of `ids`.
  std::function<std::vector<VerifySpan>(const std::vector<uint64_t>& ids, const std::vector<int>& last,
                                        const std::vector<size_t>& remaining, const std::vector<uint64_t>& position,
                                        const std::vector<int>& cap)> propose_all;
};

// Spare rows of a packed verify pass, after one row per request, go to speculative requests in equal shares;
// `turn` rotates the remainder. A cap changes only how many drafts a request proposes, never its output. One span
// holds at most kMaxRows rows (SpecSpansError), so no cap exceeds kMaxRows - 1; that binds only above 8 rows.
inline std::vector<int> DraftCaps(const std::vector<bool>& speculative, int rows, uint64_t turn) {
  std::vector<int> caps(speculative.size(), 0);
  std::vector<size_t> order;
  for (size_t i = 0; i < speculative.size(); ++i) if (speculative[i]) order.push_back(i);
  const int spare = rows - int(speculative.size());
  if (order.empty() || spare <= 0) return caps;
  const int share = spare / int(order.size()), extra = spare % int(order.size());
  for (size_t k = 0; k < order.size(); ++k) {
    const size_t index = order[(k + turn) % order.size()];
    caps[index] = std::min(kMaxRows - 1, share + (int(k) < extra ? 1 : 0));
  }
  return caps;
}

// DraftCaps with priority: requests that want copy rows (want > 0) take spare rows first, longest want first
// (index order on ties), each up to its want. The rows left go to the other speculative requests as DraftCaps does.
inline std::vector<int> DraftCapsFor(const std::vector<bool>& speculative, const std::vector<int>& want, int rows,
                                     uint64_t turn) {
  std::vector<int> caps(speculative.size(), 0);
  int spare = rows - int(speculative.size());
  std::vector<size_t> order;
  for (size_t i = 0; i < speculative.size(); ++i) if (speculative[i] && want[i] > 0) order.push_back(i);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return want[a] > want[b]; });
  for (const size_t i : order) {
    caps[i] = std::max(0, std::min({want[i], spare, kMaxRows - 1}));
    spare -= caps[i];
  }
  std::vector<bool> rest(speculative.size(), false);
  for (size_t i = 0; i < speculative.size(); ++i) rest[i] = speculative[i] && want[i] <= 0;
  // DraftCaps reserves one row per listed request; every request here already holds its row.
  const auto shared = DraftCaps(rest, std::max(0, spare) + int(rest.size()), turn);
  for (size_t i = 0; i < caps.size(); ++i) if (rest[i]) caps[i] = shared[i];
  return caps;
}

template <class E>
class BatchSession {
 public:
  BatchSession(E& engine, int slots, uint64_t cache_budget = 0, int short_slots = 0, int short_ctx = 0)
      : engine_(engine), slots_(slots) {
    if (slots < 1 || slots > kMaxRows) throw std::invalid_argument("batch slots must be in [1,8]");
    if (short_slots < 0 || short_slots >= slots || (short_slots && (short_ctx < 1 || short_ctx >= engine.Context())))
      throw std::invalid_argument("short slots need 1..slots-1 slots and a context below the full context");
    // Live states and the whole cache budget share one admission check.
    engine_.ConfigureServeStreams(slots, cache_budget, short_slots, short_ctx);
    resident_.resize(size_t(slots));
    checkpoints_ = SlotCheckpoints(slots);
    for (int slot = 0; slot < slots; ++slot) {
      contexts_.push_back(slot < slots - short_slots ? engine.Context() : short_ctx);
      // Short slots may carry verification headroom beyond their admitted context.
      if (engine_.StreamCapacity(slot) < contexts_.back()) throw std::logic_error("slot capacity below its class");
    }
  }
  void SetSpecHooks(BatchSpecHooks hooks) { spec_ = std::move(hooks); }
  const std::vector<int>& SlotContexts() const { return contexts_; }
  void SetCacheHooks(BatchCacheHooks hooks) { cache_ = std::move(hooks); }
  size_t Active() const { return active_.size(); }
  // The prompt cursor after the last Prefill. The frontend ranks waiting turns by the rows that remain (SRPT).
  // An encoder slice moves no prompt row, so it leaves no cursor.
  std::optional<size_t> PrefillCursor() const { return prefill_cursor_; }
  bool Poisoned() const { return poisoned_; }

  std::vector<BatchEvent> Open(uint64_t id, const ServeRequest& request, bool deferred = false) {
    SettlePrompt();
    ValidateOpen(id, request);
    const int slot = FreeSlot(request);
    if (generations_[slot] == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("sequence slot generation limit reached; restart the worker");
    auto state = std::make_unique<Request>(id, slot, ++generations_[slot], request);
    std::optional<BatchResident> resident;
    const bool held = resident_[size_t(slot)].has_value();
    const std::vector<int> key = Key(request, request.ids);
    if (cache_.resume && Resumable(slot, request, key))
      resident = BatchResident{resident_[size_t(slot)]->key.size(), resident_[size_t(slot)]->replay,
                               resident_[size_t(slot)]->guarded};
    const auto point = resident ? std::nullopt : Checkpoint(slot, request, key);
    // Where the prompt leaves the slot's previous tail, for the divergence line of a checkpoint resume.
    std::optional<std::pair<size_t, size_t>> left;
    if (point && resident_[size_t(slot)]) {
      const auto& tail = resident_[size_t(slot)]->key;
      const size_t limit = std::min(tail.size(), key.size());
      const size_t common = size_t(std::mismatch(tail.begin(), tail.begin() + std::ptrdiff_t(limit), key.begin()).first -
                                   tail.begin());
      if (common < tail.size()) left = {tail.size(), common};
    }
    resident_[size_t(slot)].reset();  // this request writes the slot
    DeviceUpdate([&] {
      engine_.SelectStream(slot);
      if (held && cache_.release) cache_.release(slot, resident.has_value());
      if (!resident && !point) engine_.Reset();
      Request& owned = *state;
      active_.emplace(id, std::move(state));
      if (cache_.restore) {
        const double start = Clock();
        BatchRestore hit;
        if (point) {
          const SlotCheckpoint& saved = checkpoints_.At(slot, *point);
          cache_.checkpoint_load(slot, *point, saved);
          owned.grid_point = saved.kind == CheckpointKind::Grid;
          hit = BatchRestore{saved.hist.size(), false, saved.replay, true, false, saved.guarded};
        } else {
          hit = resident ? cache_.resume(owned.options, *resident) : cache_.restore(owned.options);
        }
        // A host restore resets the stream itself; a miss after all must not keep the old slot state.
        if (resident && !hit.slot && !hit.tokens) engine_.Reset();
        if (hit.slot && (!resident || hit.tokens != resident->length || !hit.tail))
          throw std::logic_error("slot resume must keep the resident tail");
        // Only a slot or checkpoint resume keeps the slot's rows (below the restored length).
        checkpoints_.Keep(slot, hit.slot || point ? hit.tokens : 0);
        owned.from_checkpoint = point.has_value();
        if (left && left->second > hit.tokens && cache_.diverged)
          cache_.diverged(owned.options.ids.size(), hit.tokens, left->first, left->second);
        owned.cached = owned.cursor = hit.tokens;
        owned.cache_shared = hit.shared;
        owned.replay = hit.replay;
        owned.from_tail = hit.tail;
        owned.slot_resume = hit.slot;
        owned.tail_guarded = hit.tail && hit.guarded;
        owned.tail_replay = hit.tail ? hit.replay : 0;
        owned.restore = Clock() - start;
        // The last prompt row must run here: it produces the first token's logits.
        if (owned.cached >= owned.options.ids.size())
          throw std::logic_error("prefix cache restored the whole prompt");
        // A turn that starts on a tail can be retried or edited from here.
        if (hit.tail && hit.tokens) SaveCheckpoint(owned, CheckpointKind::Start, hit.replay, owned.tail_guarded);
      }
      // On the selected slot, before any hook can select another stream.
      if (cache_.tail && cache_.expect && owned.options.cache_control != "bypass")
        cache_.expect(owned.options, slot, owned.options.ids.size() +
                                           std::min(size_t(std::max(owned.options.max_tokens, 0)), kTailExpectRows));
      // Every open rebinds, so a text request never sees a previous request's images. The encoder work waits
      // for this request's prefill commands, so peers decode between its slices.
      if constexpr (requires { engine_.BindImages(engine_.ActiveSequence(), owned.options.images, size_t{0}, true); }) {
        owned.vision = engine_.BindImages(engine_.ActiveSequence(), owned.options.images, owned.cached, true);
        owned.vision_slices = engine_.VisionSlicesLeft(engine_.ActiveSequence());
      }
      if (owned.spec) spec_.open(id, slot, owned.options, owned.cached);
    });
    if (deferred) {
      BatchEvent opened{id, generations_[slot], {}, false, false, request.ids.size()};
      opened.slot_context = contexts_[slot];
      opened.cached_tokens = active_.at(id)->cached;
      if (!request.images.empty()) opened.vision_slices = active_.at(id)->vision_slices;
      return {opened};
    }
    for (;;) {
      auto events = Prefill(id);
      if (!events.empty()) return events;
    }
  }

  // The disk tier: start the read of a disk state for `request` before its open, so the open restores from memory
  // and the device worker never waits for the disk. Reads no device state and changes no slot.
  BatchStage Stage(const ServeRequest& request) const {
    RequireHealthy();
    if (!cache_.stage || request.cache_control != "auto" || request.ids.size() < 2) return {};
    const std::vector<int> key = Key(request, request.ids);
    size_t floor = 0;
    for (int slot = 0; slot < slots_; ++slot) {
      if (Resumable(slot, request, key)) floor = std::max(floor, resident_[size_t(slot)]->key.size());
      if (cache_.checkpoint_load)
        if (const auto best = checkpoints_.Best(slot, key)) floor = std::max(floor, checkpoints_.At(slot, *best).hist.size());
    }
    return cache_.stage(request, floor);
  }

  // Interleaved prefill: verify passes for `decoders` run between the layers of a bulk
  // chunk, every `every` layers; `part` receives each pass's events, tail saves done, before the chunk continues.
  // Decoders that finish drop out of later passes. A small-row chunk runs no pass.
  using PartSink = std::function<void(std::vector<BatchEvent>)>;
  std::vector<BatchEvent> Prefill(uint64_t id, const std::vector<uint64_t>& decoders = {}, int every = 0,
                                  const PartSink& part = {}) {
    SettlePrompt();
    RequireHealthy();
    if (!active_.contains(id) || active_.at(id)->last >= 0)
      throw std::invalid_argument("request is not waiting for prefill");
    if (!decoders.empty()) {
      if (every <= 0 || !part) throw std::invalid_argument("interleaved prefill needs a layer interval and a sink");
      if (std::find(decoders.begin(), decoders.end(), id) != decoders.end())
        throw std::invalid_argument("interleaved prefill cannot decode its own request");
      CheckVerify(decoders);
    }
    // The next image's pixels are read and checked on the host before any device work, outside DeviceUpdate: a file
    // that changed after the open ends only this request, and the session stays healthy (review F1).
    prefill_cursor_.reset();  // a command that ends early (below) names no cursor
    if constexpr (requires { engine_.PrepareVisionPixels(engine_.ActiveSequence()); }) {
      const Request& waiting = *active_.at(id);
      engine_.SelectStream(waiting.slot);
      // The longest chunk this command can run (a merged bulk launch, below, can pass the grid); a head chunk is
      // shorter, so at most it reads one chunk early.
      const size_t step = size_t(engine_.PrefillRows() > 0 ? std::max(engine_.PrefillRows(), EngineLaunchRows(engine_))
                                                           : kMaxRows);
      const size_t end = waiting.cursor + std::min(step, waiting.options.ids.size() - waiting.cursor);
      try {
        if (engine_.VisionSlicesBefore(engine_.ActiveSequence(), end) > 0)
          engine_.PrepareVisionPixels(engine_.ActiveSequence());
      }
      catch (const std::runtime_error& error) { return FailRequest(id, error.what()); }
      // The host buffer of one image (up to 12 MB) is the only allocation here; no state changed.
      catch (const std::bad_alloc&) { return FailRequest(id, ImageBufferError()); }
    }
    PrefillBetween between{every, [&] {
      std::vector<uint64_t> live;
      for (const uint64_t decoder : decoders) if (active_.contains(decoder)) live.push_back(decoder);
      if (live.empty()) return;
      auto events = VerifyEvents(live, CheckVerify(live));
      SaveTails(&events);
      part(std::move(events));
    }};
    return DeviceUpdate([&]() -> std::vector<BatchEvent> {
    Request& state = *active_.at(id);
    engine_.SelectStream(state.slot);
    prefill_cursor_.reset();
    const int prefill = engine_.PrefillRows();
    const int step = prefill > 0 ? prefill : kMaxRows;
    const auto& ids = state.options.ids;
    // A restored tail first runs one small-row Forward (with its pending GDN rows, if any), as serial RunPrompt does.
    const int replay = state.replay;
    // A Grid checkpoint is a chunk end of a replayed turn, after its head: the resume continues that turn's chunk
    // schedule, so it gives the same rows. A second head would shift every later chunk end.
    const bool head = replay || (state.from_tail && state.cursor == state.cached && !state.grid_point);
    const bool bulk = prefill > 0 && !head;
    int rows = int(std::min<size_t>(head ? std::min(step, kMaxRows) : step, ids.size() - state.cursor));
    // Whole grid chunks merge into one launch when no capture point falls between them and the launch ends before
    // the MTP warm-up rows; a replayed turn keeps the grid (it saves a slot checkpoint at each chunk end). With
    // decoders waiting the grid stays: a merged layer is longer, so the decode gap grows (3 x 32K, interleave 10:
    // 0.75 s against 0.65 s) and the extra passes cancel the launch gain (TTFT 27.1 s against 27.0 s).
    if (bulk && !state.from_tail && decoders.empty() && EngineLaunchRows(engine_) > step &&
        (!cache_.capture || cache_.next_capture)) {
      size_t stop = cache_.capture ? cache_.next_capture(state.options, state.cursor) : SIZE_MAX;
      if (state.spec) stop = std::min(stop, DraftWarmupStart(ids.size(), state.options.mtp_prefix));
      rows = PrefillLaunchRows(state.cursor, ids.size(), step, EngineLaunchRows(engine_), stop);
    }
    const bool final = state.cursor + rows == ids.size();
    // The images that this chunk reads get their encoder slices first, one slice per command, so peers decode
    // between them; the embeddings of images behind the cursor are freed after the chunk.
    if constexpr (requires { engine_.VisionSlicesBefore(engine_.ActiveSequence(), size_t{0}); })
      if (engine_.VisionSlicesBefore(engine_.ActiveSequence(), state.cursor + size_t(rows)) > 0) {
        state.vision += engine_.EncodeVisionSlice(engine_.ActiveSequence());
        return {};
      }
    const double start = Clock();
    const int keep = !state.spec ? rows : spec_.stacks ? spec_.stacks(id, state.cursor, rows) : 0;
    // The PLE rows of the next two chunks start their disk reads now (CRUCIBLE_PLE_HOST).
    if constexpr (requires { engine_.HintPle(ids, 0, 0); })
      if (bulk) engine_.HintPle(ids, state.cursor + rows, state.cursor + rows + 2 * size_t(step));
    const auto output = bulk
        ? engine_.Prefill(ids.data() + state.cursor, rows, "", final, decoders.empty() ? nullptr : &between, keep)
        : engine_.Forward(ids.data() + state.cursor, rows, replay, rows);
    state.replay = 0;
    // A final chunk samples before the speculative hook: MTP pairing reuses the logits buffer, and serial
    // RunPrompt also reads the prompt logits before it pairs the MTP rows.
    if (final && output.empty()) throw std::runtime_error("engine returned no prompt prediction");
    const int token = final ? Sample(state, output.back(), bulk ? 0 : rows - 1) : -1;
    // A first token that does not end its request is published before the rest of the prompt end:
    // SettlePrompt runs it before any later device work, in the old order. Outputs do not depend on drafts.
    if (final && !Ends(state, token)) {
      prompt_end_ = PromptEnd{id, state.cursor, rows, bulk, token};
      state.cursor += rows;
      prefill_cursor_ = state.cursor;
      // The MTP rows take token embeddings only (no image overlay), so the image rows go now.
      if constexpr (requires { engine_.ReleaseImageRows(engine_.ActiveSequence(), size_t{0}); })
        engine_.ReleaseImageRows(engine_.ActiveSequence(), state.cursor);
      state.prefill += Clock() - start;
      return Publish(id, token);
    }
    if (state.spec) spec_.chunk(id, state.cursor, rows, bulk);
    state.cursor += rows;
    prefill_cursor_ = state.cursor;
    if constexpr (requires { engine_.ReleaseImageRows(engine_.ActiveSequence(), size_t{0}); })
      engine_.ReleaseImageRows(engine_.ActiveSequence(), state.cursor);
    if (!final) {
      // A replayed turn saves no prompt frontiers (below), so its slot keeps the latest chunk boundary instead.
      if (state.from_tail) SaveCheckpoint(state, CheckpointKind::Grid, 0, false);
      state.prefill += Clock() - start;
      // A replayed turn's rows follow a shifted chunk schedule, so its prompt frontiers are not default-path state.
      if (cache_.capture && !state.from_tail) {
        const double saved = Clock();
        const uint64_t tried = CacheAttempts();
        cache_.capture(state.options, state.cursor);
        NoteSave(tried, state.saved_tokens, state.refused);
        state.capture += Clock() - saved;
      }
      return {};
    }
    FinishPrompt(state, token);
    state.prefill += Clock() - start;
    return Publish(id, token);
    });
  }

  // The rest of a published prompt end: the final chunk's MTP warm-up, the draft start and the End
  // checkpoint. The serve loop calls it after each command's events; every command calls it first.
  void SettlePrompt() {
    if (!prompt_end_) return;
    const PromptEnd end = *std::exchange(prompt_end_, std::nullopt);
    if (!active_.contains(end.id)) return;
    DeviceUpdate([&] {
      Request& state = *active_.at(end.id);
      const double start = Clock();
      engine_.SelectStream(state.slot);
      if (state.spec) spec_.chunk(end.id, end.start, end.rows, end.bulk);
      FinishPrompt(state, end.token);
      state.prefill += Clock() - start;
    });
  }
  // The serve loop's call after a command's boundary: the text of a failed prompt end (the session is then poisoned),
  // which the loop sends as an error event.
  std::optional<std::string> TrySettlePrompt() {
    try { SettlePrompt(); }
    catch (const std::exception& error) { return std::string(error.what()); }
    return std::nullopt;
  }

  std::vector<BatchEvent> Step(const std::vector<uint64_t>& ids) {
    SettlePrompt();
    RequireHealthy();
    if (ids.empty() || ids.size() > size_t(slots_))
      throw std::invalid_argument("batch step must contain one to eight active requests");
    std::set<uint64_t> unique;
    for (uint64_t id : ids) {
      if (!unique.insert(id).second || !active_.contains(id) || active_.at(id)->last < 0)
        throw std::invalid_argument("batch step contains a duplicate or inactive request");
      if (active_.at(id)->spec && (!spec_.commit || !spec_.packed))
        throw std::invalid_argument("this worker decodes a speculative request only with spec_step");
    }
    return DeviceUpdate([&] {
    std::vector<BatchEvent> events;
    std::vector<uint64_t> batch;
    std::vector<int> tokens, slots;
    for (uint64_t id : ids) {
      Request& state = *active_.at(id);
      if (state.spec) {
        engine_.SelectStream(state.slot);
        const double start = Clock();
        if (const auto greedy = spec_.commit(id, state.last)) {
          const int token = Sample(state, *greedy, 0);
          spec_.packed({id}, {0}, {token});
          state.decode += Clock() - start;
          const auto one = Publish(id, token);
          events.insert(events.end(), one.begin(), one.end());
          continue;
        }
      }
      batch.push_back(id);
      tokens.push_back(state.last);
      slots.push_back(state.slot);
    }
    if (batch.empty()) return events;
    const double start = Clock();
    const auto greedy = engine_.ForwardBatch(tokens, slots);
    if (greedy.size() != batch.size()) throw std::runtime_error("engine returned the wrong batch width");
    std::vector<int> output;
    for (size_t row = 0; row < batch.size(); ++row)
      output.push_back(Sample(*active_.at(batch[row]), greedy[row], int(row)));
    // The draft catch-up reads the pass's rows, so it runs before any publication can close a request.
    std::vector<uint64_t> spec_ids;
    std::vector<int> spec_rows, spec_tokens;
    for (size_t row = 0; row < batch.size(); ++row)
      if (active_.at(batch[row])->spec) {
        spec_ids.push_back(batch[row]); spec_rows.push_back(int(row)); spec_tokens.push_back(output[row]);
      }
    if (!spec_ids.empty()) spec_.packed(spec_ids, spec_rows, spec_tokens);
    const double elapsed = Clock() - start;
    for (size_t row = 0; row < batch.size(); ++row) {
      active_.at(batch[row])->decode += elapsed;
      const auto one = Publish(batch[row], output[row]);
      events.insert(events.end(), one.begin(), one.end());
    }
    return events;
    });
  }

  // Rows of one packed verification pass (packed_rows_option.h): kMaxRows unless set. A fixed cap of 9 to 16 needs
  // an engine that runs such passes (Engine::MaxPassRows); else this refuses it. With wide_from > 0 (automatic),
  // a pass of fewer decoders uses kMaxRows, and the setting falls back to kMaxRows where a wide pass cannot run
  // (fewer slots than wide_from: then the engine is not asked, because its first MaxPassRows call builds the
  // 16-row plans). Speed only: the rows never change a request's output.
  void SetPassRows(int rows, int wide_from = 0) {
    static_assert(kMaxRows == 8 && kPackedRows == 16, "the messages name the limits");
    if (rows < kMaxRows || rows > kPackedRows || wide_from < 0) throw std::invalid_argument("packed verify rows must be 8 to 16");
    if (wide_from && rows > kMaxRows && (slots_ < wide_from || engine_.MaxPassRows() < rows)) rows = kMaxRows;
    else if (!wide_from && rows > kMaxRows && rows > engine_.MaxPassRows())
      throw std::invalid_argument("this engine runs packed verify passes of at most 8 rows");
    pass_rows_ = rows;
    wide_from_ = rows > kMaxRows ? wide_from : 0;
  }
  int PassRows() const { return pass_rows_; }
  int WideFrom() const { return wide_from_; }

  // One packed verification pass over every listed request (G6d). Speculative requests propose drafts within
  // their row caps; plain requests take one row. Each request's output equals its solo steps.
  std::vector<BatchEvent> VerifyStep(const std::vector<uint64_t>& ids) {
    SettlePrompt();
    const auto speculative = CheckVerify(ids);
    return DeviceUpdate([&] { return VerifyEvents(ids, speculative); });
  }

 private:
  std::vector<bool> CheckVerify(const std::vector<uint64_t>& ids) const {
    RequireHealthy();
    if (ids.empty() || ids.size() > size_t(slots_))
      throw std::invalid_argument("verify step must contain one to eight active requests");
    std::set<uint64_t> unique;
    std::vector<bool> speculative;
    for (uint64_t id : ids) {
      if (!unique.insert(id).second || !active_.contains(id) || active_.at(id)->last < 0)
        throw std::invalid_argument("verify step contains a duplicate or inactive request");
      speculative.push_back(active_.at(id)->spec);
      if (speculative.back() && (!spec_.propose || !spec_.accept || !spec_.settle))
        throw std::invalid_argument("this worker has no packed verification");
    }
    return speculative;
  }
  std::vector<BatchEvent> VerifyEvents(const std::vector<uint64_t>& ids, const std::vector<bool>& speculative) {
    const double start = Clock();
    std::vector<int> want(ids.size(), 0);
    if (spec_.want)
      for (size_t i = 0; i < ids.size(); ++i)
        if (speculative[i]) want[i] = spec_.want(ids[i], active_.at(ids[i])->last);
    const int cap_rows = wide_from_ && ids.size() < size_t(wide_from_) ? kMaxRows : pass_rows_;
    const auto caps = DraftCapsFor(speculative, want, cap_rows, verify_turn_++);
    std::vector<VerifySpan> spans;
    std::vector<int> firsts;
    int rows = 0;
    std::map<uint64_t, VerifySpan> grouped;
    if (spec_.propose_all) {
      std::vector<uint64_t> group;
      std::vector<int> lasts, group_caps;
      std::vector<size_t> remaining;
      std::vector<uint64_t> positions;
      for (size_t i = 0; i < ids.size(); ++i) {
        const Request& state = *active_.at(ids[i]);
        if (!state.spec) continue;
        group.push_back(ids[i]);
        lasts.push_back(state.last);
        remaining.push_back(size_t(state.options.max_tokens) - state.count);
        positions.push_back(state.options.ids.size() + state.count);
        group_caps.push_back(caps[i]);
      }
      const auto proposed = spec_.propose_all(group, lasts, remaining, positions, group_caps);
      if (proposed.size() != group.size()) throw std::logic_error("grouped proposal count does not match");
      for (size_t k = 0; k < group.size(); ++k) grouped.emplace(group[k], proposed[k]);
    }
    for (size_t i = 0; i < ids.size(); ++i) {
      Request& state = *active_.at(ids[i]);
      VerifySpan span{state.slot, {state.last}, 0, 1};
      if (state.spec && grouped.contains(ids[i])) {
        span = grouped.at(ids[i]);
        span.stream = state.slot;
        if (span.tokens.empty() || span.tokens.front() != state.last || span.tokens.size() > size_t(1 + caps[i]))
          throw std::logic_error("speculative proposal does not fit its row cap");
      } else if (state.spec) {
        engine_.SelectStream(state.slot);
        span = spec_.propose(ids[i], state.last, size_t(state.options.max_tokens) - state.count,
                             state.options.ids.size() + state.count, caps[i]);
        span.stream = state.slot;
        if (span.tokens.empty() || span.tokens.front() != state.last || span.tokens.size() > size_t(1 + caps[i]))
          throw std::logic_error("speculative proposal does not fit its row cap");
      }
      firsts.push_back(rows);
      rows += int(span.tokens.size());
      spans.push_back(std::move(span));
    }
    const auto greedy = engine_.ForwardVerifySpans(spans);
    if (greedy.size() != size_t(rows)) throw std::runtime_error("engine returned the wrong verify width");
    // Check mode: copy every row now; draft proposals for the next step reuse the logits buffer.
    std::vector<std::vector<float>> checked;
    if (check_) for (int row = 0; row < rows; ++row) checked.push_back(engine_.CopyLogits(1, row));
    std::vector<std::vector<int>> outputs;
    for (size_t i = 0; i < ids.size(); ++i) {
      Request& state = *active_.at(ids[i]);
      const auto span = std::span<const int>(greedy).subspan(size_t(firsts[i]), spans[i].tokens.size());
      if (!state.spec) { outputs.push_back({Sample(state, span[0], firsts[i])}); continue; }
      outputs.push_back(spec_.accept(ids[i], firsts[i], span));
      if (outputs.back().empty() || outputs.back().size() > spans[i].tokens.size())
        throw std::runtime_error("speculative acceptance returned no tokens");
    }
    if (std::find(speculative.begin(), speculative.end(), true) != speculative.end()) spec_.settle();
    if (check_) CheckRows(ids, spans, firsts, outputs, checked);
    const double elapsed = Clock() - start;
    std::vector<BatchEvent> events;
    for (size_t i = 0; i < ids.size(); ++i) {
      active_.at(ids[i])->decode += elapsed;
      for (const int token : outputs[i]) {
        if (token < 0 || token >= kVocab) throw std::runtime_error("engine returned an invalid token ID");
        const auto one = Publish(ids[i], token);
        events.insert(events.end(), one.begin(), one.end());
        if (!active_.contains(ids[i])) break;
      }
    }
    return events;
  }
  // Rows [0, n) of a span have fully accepted prefixes (row 0 reads the last token; row k the k-th accepted
  // draft), so their logits depend only on the request: equal keys across runs must hash equal.
  void CheckRows(const std::vector<uint64_t>& ids, const std::vector<VerifySpan>& spans, const std::vector<int>& firsts,
                 const std::vector<std::vector<int>>& outputs, const std::vector<std::vector<float>>& checked) {
    for (size_t i = 0; i < ids.size(); ++i) {
      const Request& state = *active_.at(ids[i]);
      const uint64_t prompt = Fnv1a(std::span<const int>(state.options.ids));
      const uint64_t position = state.options.ids.size() + state.count;
      for (size_t k = 0; k < outputs[i].size(); ++k) {
        const auto& row = checked.at(size_t(firsts[i]) + k);
        check_.Row(prompt, position + k, spans[i].tokens.at(k), Fnv1a(row.data(), row.size() * sizeof(float)));
      }
    }
  }

 public:
  std::vector<BatchEvent> SpecStep(uint64_t id) {
    SettlePrompt();
    RequireHealthy();
    if (!active_.contains(id) || active_.at(id)->last < 0 || !active_.at(id)->spec)
      throw std::invalid_argument("spec_step needs an active speculative request after prefill");
    return DeviceUpdate([&] {
      Request& state = *active_.at(id);
      engine_.SelectStream(state.slot);
      const double start = Clock();
      const auto tokens = spec_.step(id, state.last, size_t(state.options.max_tokens) - state.count,
                                     state.options.ids.size() + state.count);
      if (tokens.empty() || tokens.size() > size_t(kMaxRows)) throw std::runtime_error("spec step returned no tokens");
      state.decode += Clock() - start;
      std::vector<BatchEvent> events;
      // Publish stops at the first stop ID or the token limit, as serial generation trims its window.
      for (const int token : tokens) {
        if (token < 0 || token >= kVocab) throw std::runtime_error("engine returned an invalid token ID");
        const auto one = Publish(id, token);
        events.insert(events.end(), one.begin(), one.end());
        if (!active_.contains(id)) break;
      }
      return events;
    });
  }

  void Close(uint64_t id) {
    SettlePrompt();
    RequireHealthy();
    const auto found = active_.find(id);
    if (found == active_.end()) throw std::invalid_argument("request is not active");
    if (cache_.expect) cache_.expect(found->second->options, found->second->slot, 0);
    const bool spec = found->second->spec;
    active_.erase(found);
    if (spec) spec_.close(id);
  }

 private:
  // Ends a request before any device work of its command: one error event, then the slot is free.
  std::vector<BatchEvent> FailRequest(uint64_t id, const char* message) {
    BatchEvent failed{id, active_.at(id)->generation};
    failed.error = message;
    Close(id);
    return {failed};
  }
  struct Request {
    const uint64_t id;
    const int slot;
    const uint64_t generation;
    const ServeRequest options;
    int last = -1;
    size_t count = 0, cursor = 0, cached = 0;
    bool cache_shared = false, from_tail = false, slot_resume = false, tail_guarded = false, from_checkpoint = false;
    bool grid_point = false;  // resumed from a Grid checkpoint
    int tail_replay = 0;
    int replay = 0;
    size_t saved_tokens = 0;
    std::string refused;
    std::vector<int> generated;  // kept only when the tail hook is set
    // Serial generation runs one plain step when a single token is requested; so does batch serving.
    const bool spec;
    double prefill = 0, decode = 0, restore = 0, capture = 0, vision = 0;
    int vision_slices = 0;
    Request(uint64_t id, int slot, uint64_t generation, const ServeRequest& options)
        : id(id), slot(slot), generation(generation), options(options),
          spec(options.spec > 0 && options.max_tokens > 1) {}
  };
  // A published first token whose prompt end waits for SettlePrompt; `start`: the final chunk's first row.
  struct PromptEnd { uint64_t id; size_t start; int rows; bool bulk; int token; };
  void FinishPrompt(Request& state, int token) {
    if (state.spec) spec_.first(state.id, token);
    // The prompt end: a next prompt that leaves this request's output (a token split, dropped reasoning, a retry)
    // resumes here and prefills only the rest.
    SaveCheckpoint(state, CheckpointKind::End, 0, false);
  }
  bool Stops(const Request& state, int token) const {
    const auto& stops = state.options.stop_ids;
    return std::find(stops.begin(), stops.end(), token) != stops.end();
  }
  // Whether publishing `token` ends the request (Publish).
  bool Ends(const Request& state, int token) const {
    return Stops(state, token) || state.count + 1 >= size_t(state.options.max_tokens);
  }
  static double Clock() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void RequireHealthy() const {
    if (poisoned_) throw std::runtime_error("batch worker state is uncertain; restart the worker");
  }
  template <class F>
  auto DeviceUpdate(F&& operation) {
    RequireHealthy();
    try {
      if constexpr (std::is_void_v<decltype(operation())>) { operation(); SaveTails(nullptr); }
      else if constexpr (std::is_same_v<decltype(operation()), std::vector<BatchEvent>>) {
        auto result = operation(); SaveTails(&result); return result;
      }
      else { auto result = operation(); SaveTails(nullptr); return result; }
    }
    catch (...) { poisoned_ = true; throw; }
  }
  struct FinishedTail {
    uint64_t id; int slot; ServeRequest options; std::vector<int> generated; int replay; BatchTailSource source;
  };
  // A slot's state after its request finished: the saved tail's tokens, until the next request on the slot.
  // `key`: the cache key of the held tail's tokens.
  struct Resident { std::vector<int> key; int replay = 0; bool guarded = false; uint64_t used = 0; };
  // Runs after the command's last publication and draft settle, so every stream holds its final state.
  // The command's done events then report the tail saves and the cache they leave.
  void SaveTails(std::vector<BatchEvent>* events) {
    for (const auto& tail : tails_) {
      engine_.SelectStream(tail.slot);
      const uint64_t tried = CacheAttempts();
      // The tail is due now: the reserve helper must not pin a new share for this slot during its copy.
      if (cache_.expect) cache_.expect(tail.options, tail.slot, 0);
      const auto kept = cache_.tail(tail.options, tail.generated, tail.replay, tail.source);
      if (kept && kept->length >= tail.options.ids.size() &&
          kept->length - tail.options.ids.size() <= tail.generated.size()) {
        std::vector<int> tokens(tail.options.ids);
        tokens.insert(tokens.end(), tail.generated.begin(),
                      tail.generated.begin() + std::ptrdiff_t(kept->length - tail.options.ids.size()));
        resident_[size_t(tail.slot)] = Resident{Key(tail.options, tokens), kept->replay, kept->guarded, ++resident_tick_};
      }
      if (events)
        for (auto& event : *events)
          if (event.done && event.request_id == tail.id) NoteSave(tried, event.cache_tail_tokens, event.cache_save_refused);
    }
    tails_.clear();
    // Decode steps without a done event skip the status read (it reads /proc/meminfo).
    if (!events || !cache_.status ||
        std::none_of(events->begin(), events->end(), [](const BatchEvent& event) { return event.done; })) return;
    const auto status = cache_.status();
    for (auto& event : *events) {
      if (!event.done) continue;
      event.cache_entries = status.entries;
      event.cache_bytes = status.bytes;
      event.cache_evictions = status.evictions;
      event.host_available = status.host_available;
      event.cache_reported = true;
    }
  }
  uint64_t CacheAttempts() const { return cache_.status ? cache_.status().attempts : 0; }
  // Records the save that a hook just tried, if it tried one.
  void NoteSave(uint64_t tried, size_t& tokens, std::string& refused) const {
    if (!cache_.status) return;
    const auto status = cache_.status();
    if (status.attempts == tried) return;
    if (status.last.saved) tokens = std::max(tokens, status.last.tokens);
    else refused = status.last.refused;
  }
  void ValidateOpen(uint64_t id, const ServeRequest& request) const {
    RequireHealthy();
    if (id == 0 || active_.contains(id)) throw std::invalid_argument("request ID must be unique and positive");
    if (active_.size() >= size_t(slots_)) throw std::invalid_argument("all sequence slots are in use");
    if (request.spec > 0 && request.max_tokens > 1) {
      const std::string error = spec_.check ? spec_.check(request) : "this batch worker has no speculative decoder";
      if (!error.empty()) throw RecoverableOpenError(error);
    }
    if (!ValidTokenIds(request.ids, kVocab) || !ValidTokenIds(request.stop_ids, kVocab) ||
        !RequestFitsContext(request.ids.size(), request.max_tokens, engine_.Context()) ||
        !request.sampling.Error(kVocab).empty())
      throw RecoverableOpenError("invalid batch request sampling, token IDs, or context bounds");
    if (!request.images.empty()) {
      bool vision = false;
      if constexpr (requires { engine_.HasVision(); }) vision = engine_.HasVision();
      if (!vision) throw RecoverableOpenError("this engine runs without the vision tower; start serve with --vision 1");
      // The pixel files must exist before the open changes state; a slice reads one and checks its digest.
      for (const auto& image : request.images) {
        try { CheckImageFile(image); }
        catch (const std::runtime_error& error) { throw RecoverableOpenError(error.what()); }
      }
    }
  }
  // The smallest free slot class that holds the prompt and all requested tokens; the frontend mirrors this rule.
  // Within the class, a returning session takes the slot that still holds its tail, else its own older slot when the
  // prompt repeats almost all of that state; then an empty slot; then a slot that shares more than the system prompt
  // (its own older slot); else the slot left longest ago, so other sessions keep their states.
  int FreeSlot(const ServeRequest& request) const {
    const std::vector<int> key = Key(request, request.ids);
    int best = -1;
    std::tuple<int, size_t, uint64_t> best_key{};
    for (int slot = 0; slot < slots_; ++slot) {
      bool used = false;
      for (const auto& [id, state] : active_) used |= state->slot == slot;
      if (used || !RequestFitsContext(request.ids.size(), request.max_tokens, contexts_[slot])) continue;
      const auto rank = Affinity(slot, request, key);
      if (best < 0 || contexts_[slot] < contexts_[best] || (contexts_[slot] == contexts_[best] && rank > best_key))
        best = slot, best_key = rank;
    }
    if (best < 0) throw RecoverableOpenError("no free sequence slot holds this request; retry after a close");
    return best;
  }
  std::vector<int> Key(const ServeRequest& request, std::span<const int> tokens) const {
    return cache_.key ? cache_.key(request, tokens) : RequestCacheKey(request, tokens);
  }
  // `key`: the request's prompt key.
  bool Resumable(int slot, const ServeRequest& request, const std::vector<int>& key) const {
    const auto& held = resident_[size_t(slot)];
    return held && request.cache_control == "auto" && held->key.size() < key.size() &&
           std::equal(held->key.begin(), held->key.end(), key.begin());
  }
  std::tuple<int, size_t, uint64_t> Affinity(int slot, const ServeRequest& request, const std::vector<int>& key) const {
    const auto& held = resident_[size_t(slot)];
    if (held && Resumable(slot, request, key)) return {4, held->key.size(), 0};
    // A slot checkpoint that the prompt repeats is the session's own older state.
    const size_t point = request.cache_control == "auto" ? checkpoints_.Common(slot, key) : 0;
    const size_t common = held ? size_t(std::mismatch(held->key.begin(), held->key.end(), key.begin(), key.end()).first -
                                        held->key.begin()) : 0;
    if (point > size_t(request.shared_prefix_tokens)) return {3, std::max(point, common), 0};
    if (!held) return {2, 0, 0};
    if (common > size_t(request.shared_prefix_tokens) && common >= held->key.size() - held->key.size() / 16)
      return {3, common, 0};
    if (common > size_t(request.shared_prefix_tokens)) return {1, common, 0};
    return {0, 0, std::numeric_limits<uint64_t>::max() - held->used};
  }
  // The slot checkpoint to resume from, when it is longer than what the host cache would restore.
  // `key`: the request's prompt key.
  std::optional<int> Checkpoint(int slot, const ServeRequest& request, const std::vector<int>& key) const {
    if (!cache_.checkpoint_load || !cache_.restore || request.cache_control != "auto") return std::nullopt;
    const auto best = checkpoints_.Best(slot, key);
    if (!best || (cache_.peek && cache_.peek(request) >= checkpoints_.At(slot, *best).hist.size())) return std::nullopt;
    return best;
  }
  void SaveCheckpoint(Request& state, CheckpointKind kind, int replay, bool guarded) {
    if (!cache_.checkpoint_save || state.options.cache_control == "bypass") return;
    const std::span<const int> below(state.options.ids.data(), state.cursor);
    auto key = Key(state.options, below);
    const auto victim = checkpoints_.Victim(state.slot, key, kind, state.id);
    if (!victim) return;
    cache_.checkpoint_save(state.slot, *victim);
    checkpoints_.Store(state.slot, *victim, {std::vector<int>(below.begin(), below.end()), std::move(key),
                                             replay, guarded, kind, state.id});
  }
  int Sample(Request& state, int greedy, int row) {
    if (state.options.sampling.temperature == 0) {
      if (greedy < 0 || greedy >= kVocab) throw std::runtime_error("engine returned an invalid token ID");
      return greedy;
    }
    const auto logits = engine_.CopyLogits(1, row);
    // Keyed law: the draw depends only on (seed, absolute position, token), as in serial generation.
    return sampling::FromLogits(logits, state.options.sampling)
        .KeyedSample(state.options.sampling.seed, state.options.ids.size() + state.count);
  }
  std::vector<BatchEvent> Publish(uint64_t id, int token) {
    Request& state = *active_.at(id);
    state.last = token;
    ++state.count;
    const bool stopped = Stops(state, token);
    const BatchEvent event{id, state.generation, token, false, stopped, state.options.ids.size(), state.count,
                           state.prefill, state.decode, state.cached, cache_.bytes ? cache_.bytes() : 0,
                           state.restore, state.capture, state.cache_shared};
    std::vector<BatchEvent> events{event};
    events.back().cache_tail = state.from_tail;
    events.back().tail_replay = state.tail_replay;
    events.back().slot_resume = state.slot_resume;
    events.back().checkpoint_resume = state.from_checkpoint;
    if (cache_.tail) state.generated.push_back(token);
    if (stopped || state.count >= size_t(state.options.max_tokens)) {
      BatchEvent done = events.front();
      done.token.reset();
      done.done = true;
      done.cache_saved_tokens = state.saved_tokens;
      done.cache_save_refused = state.refused;
      done.slot_context = contexts_[size_t(state.slot)];
      const bool spec = state.spec;
      if (spec && spec_.stats) done.draft = spec_.stats(id);
      events.push_back(done);
      if (cache_.tail && state.options.cache_control != "bypass")
        tails_.push_back({id, state.slot, state.options, std::move(state.generated),
                          spec && spec_.replay ? spec_.replay(id) : 0,
                          {state.from_tail && !state.from_checkpoint, state.tail_guarded, state.slot}});
      active_.erase(id);
      if (spec) spec_.close(id);
    }
    return events;
  }
  E& engine_;
  const int slots_;
  std::vector<int> contexts_;
  bool poisoned_ = false;
  std::optional<size_t> prefill_cursor_;
  std::optional<PromptEnd> prompt_end_;
  InterleaveCheck check_;
  BatchCacheHooks cache_;
  BatchSpecHooks spec_;
  std::vector<FinishedTail> tails_;
  std::vector<std::optional<Resident>> resident_;
  SlotCheckpoints checkpoints_;
  uint64_t resident_tick_ = 0;
  std::array<uint64_t, kMaxRows> generations_{};
  uint64_t verify_turn_ = 0;
  int pass_rows_ = kMaxRows;  // SetPassRows
  int wide_from_ = 0;         // SetPassRows: decoders from which a pass uses pass_rows_ (0: always)
  std::map<uint64_t, std::unique_ptr<Request>> active_;
};
}  // namespace crucible
