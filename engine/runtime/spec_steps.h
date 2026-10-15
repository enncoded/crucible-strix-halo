// Per-call timing of the speculative decode loop (--steps 1).
//
// Each engine call ends in a blocking device-to-host copy, so a pair of
// events around the call gives its GPU span and the host clock gives its wall
// time. The step wall time minus the summed call spans is host work between
// calls plus sync and launch latency.
#pragma once

#include <array>
#include <cstdio>
#include <vector>

struct SpecSteps {
  enum Kind { kVerify, kMtpFirst, kMtpChain, kKinds };

  struct Cell {
    long calls = 0;
    double gpu_ms = 0, wall_ms = 0;
  };

  void Enable(bool on) {
    on_ = on;
    if (!on_ || !events_.empty()) return;
    events_.resize(2);
    for (auto& e : events_) HIP_OK(hipEventCreate(&e));
  }

  void Begin() {
    if (!on_) return;
    if (step_t0_ < 0) step_t0_ = Now();
    HIP_OK(hipEventRecord(events_[0], 0));
    call_t0_ = Now();
  }

  void End(Kind kind, int rows) {
    if (!on_) return;
    HIP_OK(hipEventRecord(events_[1], 0));
    HIP_OK(hipEventSynchronize(events_[1]));
    const double wall = (Now() - call_t0_) * 1e3;
    float gpu = 0;
    HIP_OK(hipEventElapsedTime(&gpu, events_[0], events_[1]));
    Cell& c = cells_[kind][rows];
    ++c.calls;
    c.gpu_ms += gpu;
    c.wall_ms += wall;
    step_gpu_ += gpu;
  }

  // MTP top-2 logit margin of draft k in the current step.
  void Margin(int k, float m) {
    if (!on_) return;
    if ((int)margins_.size() <= k) margins_.resize(k + 1);
    margins_[k] = m;
  }

  void Accepted(int depth, int a) {
    if (!on_) return;
    ++by_depth_[depth][a];
    // Draft k is informative only if drafts 0..k-1 were accepted. A negative
    // first margin marks a lookup step.
    const bool mtp = !margins_.empty() && margins_[0] >= 0.f;
    for (int k = 0; mtp && k < depth && k < (int)margins_.size() && k <= a; ++k) {
      const int b = MarginBucket(margins_[k]);
      ++margin_n_[std::min(k, kMaxRows - 1)][b];
      if (k < a) ++margin_ok_[std::min(k, kMaxRows - 1)][b];
    }
    margins_.clear();
  }

  static int MarginBucket(float m) {
    static constexpr float edges[] = {1, 2, 3, 4, 6, 8, 12};
    int b = 0;
    while (b < 7 && m >= edges[b]) ++b;
    return b;
  }

  void StepDone() {
    if (!on_) return;
    const double wall = (Now() - step_t0_) * 1e3;
    ++steps_;
    step_wall_ += wall;
    gap_ += wall - step_gpu_sum_delta();
    step_t0_ = -1;
  }

  void Print(FILE* f) const {
    if (!on_ || steps_ == 0) return;
    static const char* names[] = {"verify", "mtp_first", "mtp_chain"};
    std::fprintf(f, "steps %ld: wall %.3f ms/step, host gap %.3f ms/step\n", steps_,
                 step_wall_ / steps_, gap_ / steps_);
    for (int k = 0; k < kKinds; ++k)
      for (int r = 0; r <= kMaxRows; ++r) {
        const Cell& c = cells_[k][r];
        if (c.calls == 0) continue;
        std::fprintf(f, "  %-9s R=%d calls %5ld  gpu %8.3f ms/call  wall %8.3f ms/call  %.3f ms/step\n",
                     names[k], r, c.calls, c.gpu_ms / c.calls, c.wall_ms / c.calls, c.gpu_ms / steps_);
      }
    long any = 0;
    for (auto& row : margin_n_)
      for (long v : row) any += v;
    if (any) {
      std::fprintf(f, "  accept rate by margin bucket <1 <2 <3 <4 <6 <8 <12 >=12 (count)\n");
      for (int k = 0; k < kMaxRows; ++k) {
        long n = 0;
        for (long v : margin_n_[k]) n += v;
        if (!n) continue;
        std::fprintf(f, "  draft %d:", k + 1);
        for (int b = 0; b < 8; ++b)
          std::fprintf(f, " %.2f(%ld)", margin_n_[k][b] ? (double)margin_ok_[k][b] / margin_n_[k][b] : 0.,
                       margin_n_[k][b]);
        std::fprintf(f, "\n");
      }
    }
    for (int d = 1; d <= kMaxRows; ++d) {
      long tot = 0;
      for (long v : by_depth_[d]) tot += v;
      if (tot == 0) continue;
      std::fprintf(f, "  depth %d: %ld steps, accepted histogram", d, tot);
      for (int a = 0; a <= d; ++a) std::fprintf(f, " %ld", by_depth_[d][a]);
      std::fprintf(f, "\n");
    }
  }

  double WallPerStep() const { return steps_ ? step_wall_ / steps_ : 0; }
  double GapPerStep() const { return steps_ ? gap_ / steps_ : 0; }
  double GpuPerStep(Kind k) const {
    double s = 0;
    for (const Cell& c : cells_[k]) s += c.gpu_ms;
    return steps_ ? s / steps_ : 0;
  }
  long Steps() const { return steps_; }

 private:
  double step_gpu_sum_delta() {
    const double d = step_gpu_;
    step_gpu_ = 0;
    return d;
  }

  bool on_ = false;
  std::vector<hipEvent_t> events_;
  std::array<std::array<Cell, kMaxRows + 1>, kKinds> cells_{};
  std::array<std::array<long, kMaxRows + 1>, kMaxRows + 1> by_depth_{};
  std::array<std::array<long, 8>, kMaxRows> margin_n_{}, margin_ok_{};
  std::vector<float> margins_;
  double call_t0_ = 0, step_t0_ = -1, step_gpu_ = 0, step_wall_ = 0, gap_ = 0;
  long steps_ = 0;
};
