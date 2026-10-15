// Keep cached and fresh prompt arithmetic on the same row schedule.
#pragma once
#include "conversation_resume.h"
#include "draft_warmup.h"
#include "../model/flashnext/prefill_launch.h"
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>

namespace crucible {
template <class E>
concept HasGemmPrefill = requires(E& e, const int* t, const std::string& d) {
  { e.PrefillRows() } -> std::convertible_to<int>;
  { e.PromptStacks() } -> std::convertible_to<float*>;
  e.Prefill(t, 1, d, true);
};

// A bounded window starts at relative position zero. Target state keeps its replay count.
template <class Engine, class Copy, class Mtp, class Final>
std::vector<int> RunPrompt(Engine& eng, std::span<const int> ids, int chunk, bool mtp_on,
    const std::string& dump, float* stk, float* carry, int& mpos,
    Copy copy, Mtp mtp, Final on_final, int mtp_prefix = 0,
    const ConversationResume& resume = {}, ConversationHooks* conversation = nullptr) {
  if (conversation) conversation->deferred_prompt.reset();
  std::vector<int> last{resume.next_id};
  bool have_carry = resume.have_carry;
  int replay = resume.replay;
  const size_t mtp_start = DraftWarmupStart(ids.size(), mtp_prefix);
  const size_t boundary = conversation ? conversation->boundary : 0;
  // One canonical schedule in every server mode: the zero-origin grid, unless the caller opts in to a boundary split.
  const size_t schedule_boundary = conversation && conversation->split_at_boundary ? boundary : 0;
  int prefill = 0;
  float* stacks = nullptr;
  if constexpr (HasGemmPrefill<Engine>) {
    prefill = eng.PrefillRows();
    stacks = eng.PromptStacks();
  }
  const int step = prefill > 0 ? prefill : chunk;
  if (mtp_on && resume.length < ids.size() && !resume.pending_tokens.empty()) {
    auto tokens = resume.pending_tokens;
    tokens.back() = ids[resume.length];
    mtp(stk, tokens.data(), int(tokens.size()), mpos);
    mpos += int(tokens.size());
    have_carry = false;
  }
  for (size_t i = resume.length; i < ids.size();) {
    // A tail restore (session replay) always starts with one small-row decode-path chunk: pending GDN rows depend
    // on the spec and packing schedule, not on the history, so they must not choose the path.
    const bool head = replay || (conversation && conversation->continue_tail && i == resume.length);
    int R = ConversationRows(i, ids.size(), head ? std::min(step, kMaxRows) : step, schedule_boundary);
    if constexpr (requires { eng.PrefillLaunchRows(); }) {
      // Whole grid chunks merge into one bulk launch when no snapshot can fall between them and the launch ends
      // before the MTP warm-up rows, so the row results and the warm-up pairs stay those of the grid schedule.
      const int launch = eng.PrefillLaunchRows();
      if (prefill > 0 && !head && launch > step && !(conversation && conversation->save)) {
        const size_t limit = schedule_boundary > i && schedule_boundary < ids.size() ? schedule_boundary : ids.size();
        R = PrefillLaunchRows(i, limit, step, launch, mtp_on ? mtp_start : SIZE_MAX);
      }
    }
    const bool final = i + R == ids.size();
    const bool frontier = mtp_on && mtp_prefix && i + R <= mtp_start && mtp_start - (i + R) < size_t(step);
    const size_t stable_end = boundary && boundary < ids.size() ? boundary : ids.size();
    const bool bulk_frontier = prefill > 0 && i + R < stable_end && stable_end - (i + R) < size_t(step);
    const bool save = conversation && conversation->save &&
                      (!conversation->admit_prompt || conversation->admit_prompt(i + R)) &&
                      (conversation->admit_prompt || final || (!conversation->skip_prompt_interior_save &&
                                 (i + R == boundary || frontier || bulk_frontier)));
    const int start_row = DraftWarmupRow(i, R, mtp_start);
    bool bulk = false;
    if constexpr (HasGemmPrefill<Engine>) {
      bulk = prefill > 0 && !head;
      // The PLE rows of the next two chunks start their disk reads now (CRUCIBLE_PLE_HOST).
      if constexpr (requires { eng.HintPle(ids, i, i); })
        if (bulk) eng.HintPle(ids, i + R, i + R + 2 * size_t(step));
      // Only the MTP warm-up reads this chunk's final trunk stacks (from start_row on); the engine may skip the
      // last layer's output rows below that.
      if (bulk) last = eng.Prefill(ids.data() + i, R, dump, final || save, nullptr, mtp_on ? start_row : R);
    }
    if (!bulk) last = eng.Forward(ids.data() + i, R, replay, R, dump);
    replay = 0;
    if (final) on_final(bulk ? 0 : R - 1);
    std::vector<float> logits;
    if constexpr (requires { eng.CopyLogits(1, 0); })
      if (save) logits = eng.CopyLogits(1, bulk ? 0 : R - 1);
    if (mtp_on && start_row < R) {
      const int warm_rows = R - start_row;
      float* source = eng.H() + size_t(start_row) * kWide;
      if (bulk) {
        if (warm_rows > prefill) std::abort();  // PromptStacks() holds one grid chunk
        copy(stacks, source, size_t(warm_rows) * kWide);
        source = stacks;
      }
      const int pairs = warm_rows - 1 + int(have_carry);
      // A long warm-up (a large or full MTP window) runs in one bulk pass (mtp_bulk.h); the 8-row loop is skipped.
      bool warmed = false;
      if constexpr (requires { eng.MtpWarm(stacks, stacks, ids.data(), 1, 0); })
        if (bulk && pairs > 0) {
          warmed = eng.MtpWarm(have_carry ? carry : nullptr, source, ids.data() + i + start_row + (have_carry ? 0 : 1),
                               pairs, mpos);
          if (warmed) mpos += pairs;
        }
      for (int p = 0; p < pairs && !warmed; p += kMaxRows) {
        const int count = std::min(kMaxRows, pairs - p);
        const int first = p == 0 && have_carry ? 1 : 0;
        if (first) copy(stk, carry, kWide);
        const int source_row = p - int(have_carry) + first;
        if (count > first)
          copy(stk + size_t(first) * kWide, source + size_t(source_row) * kWide, size_t(count - first) * kWide);
        std::vector<int> tokens;
        for (int j = 0; j < count; ++j) tokens.push_back(ids[i + start_row + p + j + (have_carry ? 0 : 1)]);
        // Consume the old carry before replacing it with the current trunk stack.
        if (!bulk) copy(carry, source + size_t(warm_rows - 1) * kWide, kWide);
        mtp(stk, tokens.data(), count, mpos);
        mpos += count;
      }
      if (bulk || pairs == 0) copy(carry, source + size_t(warm_rows - 1) * kWide, kWide);
      have_carry = true;
    }
    if (save) {
      ConversationResume state{i + R, mpos, 0, last.back(), have_carry, {}, logits};
      state.prompt_only = true;
      state.prompt_boundary = boundary;
      if (final && conversation->defer_prompt_end_save) conversation->deferred_prompt = std::move(state);
      else conversation->save(state, stk, carry);
    }
    i += R;
  }
  return last;
}
// Finish the synchronous snapshot between first-token publication and the first decode mutation.
inline void FlushDeferredPrompt(ConversationHooks* hooks, float* stk, float* carry) {
  if (!hooks || !hooks->deferred_prompt || !hooks->save) return;
  const auto state = std::move(*hooks->deferred_prompt);
  hooks->deferred_prompt.reset();
  hooks->save(state, stk, carry);
}
}  // namespace crucible
