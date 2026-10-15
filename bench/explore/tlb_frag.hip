// Drift study: does the physical contiguity of GPU memory change read speed?
// GPU memory on this APU is GTT: TTM takes pages from the buddy allocator,
// highest order first, and the GPU page table can use larger translation
// fragments for contiguous runs. The order mix that one buffer receives is
// read from the change in /proc/buddyinfo while it is allocated (no root).
//
// usage: tlb_frag GB [PRECOMPACT_GB]
//   Allocates GB buffers of 1 GiB one by one, like the engine's arenas. For
//   each: free bytes lost per buddy order, then sequential and random-block
//   (16 KiB, 256 KiB) read bandwidth. PRECOMPACT_GB > 0 first faults that
//   many GiB of MADV_HUGEPAGE memory (direct compaction makes 2 MiB blocks)
//   and frees it, so the next buffers can take high-order blocks.
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define HIP_OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)

namespace {

constexpr size_t kBuf = size_t(1) << 30;
constexpr int kOrders = 11;

__global__ void Seq(const uint4* p, size_t n, unsigned* out) {
  unsigned acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) out[0] = acc;
}

// Each block reads `chunks` contiguous chunks of `chunk` bytes at hashed offsets.
__global__ void Random(const uint4* p, size_t chunk, size_t nchunks, int per_block, unsigned* out) {
  unsigned acc = 0;
  const size_t words = chunk / sizeof(uint4);
  for (int k = 0; k < per_block; ++k) {
    uint64_t h = (uint64_t)(blockIdx.x * per_block + k) * 0x9e3779b97f4a7c15ull;
    h ^= h >> 29;
    const size_t base = (h % nchunks) * words;
    for (size_t i = threadIdx.x; i < words; i += blockDim.x) {
      const uint4 v = p[base + i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x9e3779b9u) out[0] = acc;
}

std::vector<long long> Buddy() {
  std::ifstream f("/proc/buddyinfo");
  std::string line;
  std::vector<long long> c(kOrders, 0);
  while (std::getline(f, line)) {
    if (line.find("Normal") == std::string::npos) continue;
    std::istringstream s(line.substr(line.find("Normal") + 6));
    for (int o = 0; o < kOrders; ++o) s >> c[o];
  }
  return c;
}

float Time(hipEvent_t a, hipEvent_t b) {
  float ms;
  HIP_OK(hipEventElapsedTime(&ms, a, b));
  return ms;
}

void Precompact(size_t gib) {
  const size_t region = gib << 30, huge = size_t(2) << 20;
  char* p = static_cast<char*>(mmap(nullptr, region + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (p == MAP_FAILED) { std::perror("mmap"); return; }
  char* a = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + huge - 1) & ~uintptr_t(huge - 1));
  madvise(a, region, MADV_HUGEPAGE);
  const auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < region; i += huge) a[i] = 1;
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  munmap(p, region + huge);
  std::printf("precompact %zu GiB of THP faults in %.2f s\n", gib, s);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: tlb_frag GB [PRECOMPACT_GB]\n"); return 2; }
  const int gb = std::atoi(argv[1]);
  const size_t pre = argc > 2 ? std::atoi(argv[2]) : 0;
  unsigned* sink;
  HIP_OK(hipMalloc(&sink, 4));
  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  if (pre) Precompact(pre);
  std::printf("buf  lost_ge2M_MB lost_lt2M_MB  orders(net blocks lost 0..10)  seq_GBs  r16K_GBs  r256K_GBs\n");
  std::vector<void*> bufs;
  for (int i = 0; i < gb; ++i) {
    const auto b0 = Buddy();
    void* p;
    HIP_OK(hipMalloc(&p, kBuf));
    HIP_OK(hipMemset(p, 1, kBuf));
    HIP_OK(hipDeviceSynchronize());
    const auto b1 = Buddy();
    bufs.push_back(p);
    double hi = 0, lo = 0;
    std::string orders;
    for (int o = 0; o < kOrders; ++o) {
      const long long lost = b0[o] - b1[o];
      const double mb = lost * (4096.0 * (1 << o)) / (1 << 20);
      (o >= 9 ? hi : lo) += mb;
      orders += " " + std::to_string(lost);
    }
    const uint4* q = static_cast<const uint4*>(p);
    float seq = 1e9, r16 = 1e9, r256 = 1e9;
    for (int rep = 0; rep < 3; ++rep) {
      HIP_OK(hipEventRecord(e0));
      Seq<<<640, 256>>>(q, kBuf / sizeof(uint4), sink);
      HIP_OK(hipEventRecord(e1));
      HIP_OK(hipEventSynchronize(e1));
      seq = std::min(seq, Time(e0, e1));
      HIP_OK(hipEventRecord(e0));
      Random<<<4096, 256>>>(q, 16384, kBuf / 16384, 16, sink);  // 1 GiB in 16 KiB chunks
      HIP_OK(hipEventRecord(e1));
      HIP_OK(hipEventSynchronize(e1));
      r16 = std::min(r16, Time(e0, e1));
      HIP_OK(hipEventRecord(e0));
      Random<<<1024, 256>>>(q, 262144, kBuf / 262144, 4, sink);  // 1 GiB in 256 KiB chunks
      HIP_OK(hipEventRecord(e1));
      HIP_OK(hipEventSynchronize(e1));
      r256 = std::min(r256, Time(e0, e1));
    }
    const double g = kBuf / 1e9;
    std::printf("%3d %12.0f %12.0f %s  %7.1f %9.1f %10.1f\n", i, hi, lo, orders.c_str(), g / (seq * 1e-3),
                g / (r16 * 1e-3), g / (r256 * 1e-3));
    std::fflush(stdout);
  }
  for (void* p : bufs) HIP_OK(hipFree(p));
  return 0;
}
