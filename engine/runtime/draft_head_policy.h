// Choose the proposal head from verified tokens per millisecond, not language.
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace crucible {

class DraftHeadPolicy {
 public:
  bool UseFull() const { return full_; }
  bool IncumbentFull() const { return incumbent_; }
  long Steps(bool full) const { return scores_[full].steps; }
  double Rate(bool full) const {
    const Score& s = scores_[full];
    return s.ms > 0 ? s.tokens / s.ms : 0;
  }

  void Observe(int tokens, double milliseconds) {
    if (tokens < 1 || !(milliseconds > 0) || !std::isfinite(milliseconds))
      throw std::invalid_argument("head policy needs positive tokens and finite time");
    Score& score = scores_[full_];
    score.tokens = kDecay * score.tokens + tokens;
    score.ms = kDecay * score.ms + milliseconds;
    ++score.steps;
    if (++window_steps_ < window_limit_) return;
    window_steps_ = 0;
    if (!initialized_) {
      if (!full_) {
        full_ = true;
        return;
      }
      initialized_ = true;
      Select();
    } else if (probing_) {
      Select();
    } else {
      // Periodic probes detect a change in the continuation without making
      // every step pay for both heads.
      full_ = !incumbent_;
      probing_ = true;
      window_limit_ = kProbeSteps;
    }
  }

 private:
  void Select() {
    if (Rate(!incumbent_) > Rate(incumbent_) * (1 + kHysteresis))
      incumbent_ = !incumbent_;
    full_ = incumbent_;
    probing_ = false;
    window_limit_ = kExploitSteps;
  }

  static constexpr int kProbeSteps = 4, kExploitSteps = 24;
  static constexpr double kDecay = 0.9, kHysteresis = 0.03;
  struct Score { double tokens = 0, ms = 0; long steps = 0; };
  std::array<Score, 2> scores_{};
  bool full_ = false, incumbent_ = false, initialized_ = false, probing_ = false;
  int window_steps_ = 0, window_limit_ = kProbeSteps;
};

}  // namespace crucible
