#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

using ProbeClock = std::chrono::steady_clock;
inline auto ProbeNow() { return ProbeClock::now(); }
inline double ProbeSeconds(ProbeClock::time_point start) {
  return std::chrono::duration<double>(ProbeNow() - start).count();
}
inline void ProbePhase(const char* name, ProbeClock::time_point start) {
  std::printf("{\"phase\":\"%s\",\"seconds\":%.9f}\n", name, ProbeSeconds(start));
  std::fflush(stdout);
}
inline bool ProbeEager() {
  const char* value = std::getenv("CRUCIBLE_PROBE_EAGER");
  return value && value[0] == '1';
}
enum class ProbeBucket { PinAlloc, PinFree, StreamCreate, Copy, Sync, Fill, Count };
constexpr size_t kProbeBuckets = static_cast<size_t>(ProbeBucket::Count);
inline std::array<std::atomic<int64_t>, kProbeBuckets> probe_us{}, probe_calls{}, probe_max_us{};
inline thread_local std::array<int64_t, kProbeBuckets> probe_thread_us{};
struct ProbeTimer {
  ProbeBucket bucket;
  ProbeClock::time_point start = ProbeNow();
  explicit ProbeTimer(ProbeBucket value) : bucket(value) {}
  ~ProbeTimer() {
    const size_t index = static_cast<size_t>(bucket);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(ProbeNow() - start).count();
    probe_us[index].fetch_add(us);
    probe_calls[index].fetch_add(1);
    probe_thread_us[index] += us;
  }
};
inline void ProbeWorkerEnd() {
  for (size_t i = 0; i < kProbeBuckets; ++i) {
    int64_t old = probe_max_us[i].load();
    while (old < probe_thread_us[i] && !probe_max_us[i].compare_exchange_weak(old, probe_thread_us[i])) {}
  }
}
inline void ProbeCounters() {
  constexpr std::array names = {"pin_alloc", "pin_free", "stream_create", "copy", "copy_sync", "fill"};
  for (size_t i = 0; i < kProbeBuckets; ++i)
    std::printf("{\"bucket\":\"%s\",\"calls\":%lld,\"thread_seconds_sum\":%.6f,\"max_worker_seconds\":%.6f}\n",
                names[i], static_cast<long long>(probe_calls[i].load()),
                probe_us[i].load() / 1e6, probe_max_us[i].load() / 1e6);
  std::fflush(stdout);
}
