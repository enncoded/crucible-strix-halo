// Controlled test: which host allocation kinds make KFD evict this process's
// GPU queues when the kernel invalidates or migrates their pages.
//
// usage: host_evict MODE TRIGGER SECONDS [MB]
//   MODE     pageable   own mmap (4 KB pages), copied to the device with 32 MB
//                       hipMemcpy calls, as EngineStateIO does with std::vector
//            hostmalloc hipHostMalloc (GTT or USERPTR, by HSA_USERPTR_FOR_PAGED_MEM)
//            none       device memory only
//   TRIGGER  none
//            collapse   MADV_COLLAPSE on the host range (the kernel copies its
//                       4 KB pages into huge pages, as khugepaged does)
//            protect    mprotect read-only and back (an invalidation, no copy)
//            compact    THP faults on a separate 6 GB MADV_HUGEPAGE region, so
//                       the allocator compacts the zone (migrates movable pages)
// The trigger runs 2 s after the timing loop starts. The loop launches a
// 256 MB device read and synchronizes; a stalled queue shows as a long step.
#include <hip/hip_runtime.h>
#include <glob.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

#define HIP_OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)

namespace {

__global__ void Read(const uint4* p, size_t n, unsigned* out) {
  unsigned acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) out[0] = acc;
}

double Now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

long long EvictedMs() {
  glob_t g;
  const std::string pattern = "/sys/class/kfd/kfd/proc/" + std::to_string(getpid()) + "/stats_*/evicted_ms";
  long long v = -1;
  if (glob(pattern.c_str(), 0, nullptr, &g) == 0 && g.gl_pathc > 0) std::ifstream(g.gl_pathv[0]) >> v;
  globfree(&g);
  return v;
}

std::map<std::string, long long> Vm() {
  std::map<std::string, long long> m;
  std::ifstream f("/proc/vmstat");
  std::string k;
  long long v;
  while (f >> k >> v) m[k] = v;
  return m;
}

long long MemAvailableKb() {
  std::ifstream f("/proc/meminfo");
  std::string k, unit;
  long long v;
  while (f >> k >> v >> unit) if (k == "MemAvailable:") return v;
  return -1;
}

// Returns a short description of what the trigger did.
std::string Trigger(const std::string& kind, char* host, size_t bytes) {
  char buf[160];
  const double t0 = Now();
  if (kind == "collapse") {
    if (madvise(host, bytes, MADV_HUGEPAGE) != 0) return std::string("MADV_HUGEPAGE failed: ") + strerror(errno);
    const int r = madvise(host, bytes, MADV_COLLAPSE);
    std::snprintf(buf, sizeof buf, "MADV_COLLAPSE ret=%d (%s) %.3f s", r, r ? strerror(errno) : "ok", Now() - t0);
  } else if (kind == "protect") {
    const int a = mprotect(host, bytes, PROT_READ), b = mprotect(host, bytes, PROT_READ | PROT_WRITE);
    std::snprintf(buf, sizeof buf, "mprotect ret=%d,%d %.3f s", a, b, Now() - t0);
  } else if (kind == "compact") {
    const size_t region = size_t(6) << 30;
    if (MemAvailableKb() < (long long)(16) << 20) return "skipped: MemAvailable below 16 GiB";
    char* p = static_cast<char*>(mmap(nullptr, region + (2 << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (p == MAP_FAILED) return "mmap failed";
    char* a = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + (2 << 20) - 1) & ~uintptr_t((2 << 20) - 1));
    madvise(a, region, MADV_HUGEPAGE);
    for (size_t i = 0; i < region; i += 2 << 20) a[i] = 1;
    munmap(p, region + (2 << 20));
    std::snprintf(buf, sizeof buf, "6 GiB MADV_HUGEPAGE faults %.3f s", Now() - t0);
  } else {
    return "none";
  }
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: host_evict pageable|hostmalloc|none none|collapse|protect|compact SECONDS [MB]\n");
    return 2;
  }
  const std::string mode = argv[1], trigger = argv[2];
  const double seconds = std::atof(argv[3]);
  const size_t bytes = size_t(argc > 4 ? std::atoi(argv[4]) : 1024) << 20;
  const char* policy = std::getenv("HSA_USERPTR_FOR_PAGED_MEM");

  const size_t dev_bytes = size_t(256) << 20;
  uint4* dev = nullptr;
  unsigned* sink = nullptr;
  char* stage = nullptr;
  HIP_OK(hipMalloc(&dev, std::max(dev_bytes, bytes)));
  HIP_OK(hipMalloc(&sink, 4));
  HIP_OK(hipMemset(dev, 1, std::max(dev_bytes, bytes)));

  char* host = nullptr;
  if (mode == "pageable") {
    host = static_cast<char*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (host == MAP_FAILED) { std::perror("mmap"); return 1; }
    madvise(host, bytes, MADV_NOHUGEPAGE);  // 4 KB pages, so a collapse must copy them
    std::memset(host, 3, bytes);
    const char* chunk_env = std::getenv("HOST_EVICT_CHUNK_MB");
    const size_t chunk = size_t(chunk_env ? std::atoi(chunk_env) : 32) << 20;
    for (size_t o = 0; o < bytes; o += chunk)
      HIP_OK(hipMemcpy(reinterpret_cast<char*>(dev) + o, host + o, std::min(chunk, bytes - o), hipMemcpyHostToDevice));
  } else if (mode == "hostmalloc") {
    HIP_OK(hipHostMalloc(reinterpret_cast<void**>(&stage), bytes));
    std::memset(stage, 3, bytes);
    HIP_OK(hipMemcpy(dev, stage, bytes, hipMemcpyHostToDevice));
    host = stage;
  }
  HIP_OK(hipDeviceSynchronize());

  std::vector<double> step_ms;
  std::atomic<bool> fire{false};
  std::string what = "none";
  double fired_at = 0, done_at = 0;
  std::thread worker([&] {
    while (!fire.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    fired_at = Now();
    what = host || trigger == "compact" ? Trigger(trigger, host, bytes) : "no host range";
    done_at = Now();
  });

  const long long ev0 = EvictedMs();
  const auto vm0 = Vm();
  const double start = Now();
  std::vector<double> stamp;
  while (Now() - start < seconds) {
    if (!fire.load() && Now() - start > 2.0) fire.store(true);
    const double t = Now();
    Read<<<640, 256>>>(dev, dev_bytes / sizeof(uint4), sink);
    HIP_OK(hipDeviceSynchronize());
    step_ms.push_back((Now() - t) * 1e3);
    stamp.push_back(t - start);
  }
  fire.store(true);
  worker.join();
  const long long ev1 = EvictedMs();
  auto vm1 = Vm();

  std::vector<double> sorted = step_ms;
  std::sort(sorted.begin(), sorted.end());
  const double med = sorted[sorted.size() / 2], p99 = sorted[sorted.size() * 99 / 100];
  int slow = 0;
  double slow_sum = 0;
  for (double v : step_ms) if (v > 2 * med) { ++slow; slow_sum += v - med; }
  auto d = [&](const char* k) { return vm1[k] - vm0.at(k); };
  std::printf("mode=%s policy=%s trigger=%s MB=%zu steps=%zu median=%.3f p99=%.3f max=%.3f slow=%d excess_ms=%.1f "
              "evicted_ms=%lld pgmigrate=%lld compact_stall=%lld thp_collapse=%lld trigger_s=%.3f [%s]\n",
              mode.c_str(), policy ? policy : "unset", trigger.c_str(), bytes >> 20, step_ms.size(), med, p99,
              sorted.back(), slow, slow_sum, ev1 - ev0, d("pgmigrate_success"), d("compact_stall"),
              d("thp_collapse_alloc"), done_at - fired_at, what.c_str());
  for (size_t i = 0, shown = 0; i < step_ms.size() && shown < 40; ++i)
    if (step_ms[i] > 2 * med && ++shown) std::printf("  slow step t=%.3f s %.2f ms\n", stamp[i], step_ms[i]);
  if (stage) HIP_OK(hipHostFree(stage));
  if (mode == "pageable") munmap(host, bytes);
  return 0;
}
