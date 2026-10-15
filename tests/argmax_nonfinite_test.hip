// ArgmaxKernel (layers/ops.h) gives the first index of the largest finite logit for a finite row
// (-Inf is a masked logit; an all -Inf row gives 0), and kNonfiniteId for a row with a NaN or +Inf logit.
// Every row is checked against a CPU reference and against ArgmaxBefore, a copy of the kernel before an earlier change, so a
// finite row must give the same ID as before. The output is filled with a sentinel before each launch, so an
// unwritten row cannot pass, and a mutant (one changed ID) must be rejected.
// --bench N: N ABAB pairs of ArgmaxBefore and ArgmaxKernel on 16 finite rows of kVocab logits (hipEvent time).
#include "../engine/layers/ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace crucible;

namespace {

// ArgmaxKernel as it was on a0ab9417 (reference only; the engine never launches it).
__global__ void __launch_bounds__(1024) ArgmaxBefore(const float* __restrict__ logits, int V, int* __restrict__ out) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  const int r = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const float* lg = logits + (size_t)r * V;
  float b = -INFINITY;
  int i0 = 0;
  for (int i = threadIdx.x; i < V; i += 1024)
    if (lg[i] > b) {
      b = lg[i];
      i0 = i;
    }
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    const float ov = __shfl_xor(b, m);
    const int oi = __shfl_xor(i0, m);
    if (ov > b || (ov == b && oi < i0)) {
      b = ov;
      i0 = oi;
    }
  }
  if (lane == 0) {
    bv[w] = b;
    bi[w] = i0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    b = bv[0];
    i0 = bi[0];
    for (int i = 1; i < 32; ++i)
      if (bv[i] > b || (bv[i] == b && bi[i] < i0)) {
        b = bv[i];
        i0 = bi[i];
      }
    out[r] = i0;
  }
}

constexpr int kSentinel = 0x7fffffff;

struct Row {
  std::string name;
  std::vector<float> v;
};

int Reference(const std::vector<float>& v) {
  int best = 0;
  float b = -INFINITY;
  for (size_t i = 0; i < v.size(); ++i) {
    if (std::isnan(v[i]) || v[i] == INFINITY) return kNonfiniteId;
    if (v[i] > b) { b = v[i]; best = int(i); }
  }
  return best;
}

std::vector<float> Random(int V, uint32_t seed) {
  std::vector<float> v(V);
  uint32_t x = seed * 2654435761u + 1;
  for (float& f : v) {
    x = x * 1664525u + 1013904223u;
    f = 30.f * (float(x >> 8) / float(1u << 24) - 0.5f);
  }
  return v;
}

std::vector<Row> Rows(int V) {
  std::vector<Row> rows;
  const float nan = std::nanf("");
  auto add = [&](std::string name, std::vector<float> v) { rows.push_back({std::move(name), std::move(v)}); };
  for (uint32_t s = 1; s <= 6; ++s) add("random" + std::to_string(s), Random(V, s));
  auto ties = Random(V, 7);
  for (int i : {V - 1, 4097, 33, 1056})
    if (i < V) ties[i] = 100.f;  // equal maxima in other lanes and waves: the lowest wins
  add("ties", ties);
  auto masked = Random(V, 8);
  for (int i = 0; i < V; i += 3) masked[i] = -INFINITY;
  add("some -inf", masked);
  add("all -inf", std::vector<float>(V, -INFINITY));
  add("all equal", std::vector<float>(V, 0.25f));
  for (int at : {0, 1023, 1024, V / 2, V - 1}) {
    if (at >= V) continue;
    auto v = Random(V, 9 + at);
    v[at] = nan;
    add("nan@" + std::to_string(at), v);
    v[at] = INFINITY;
    add("+inf@" + std::to_string(at), v);
  }
  add("all nan", std::vector<float>(V, nan));
  auto nan_and_inf = std::vector<float>(V, -INFINITY);
  nan_and_inf[17] = nan;
  add("nan in -inf", nan_and_inf);
  return rows;
}

template <class T> T* Device(const std::vector<T>& h) {
  T* d;
  HIP_OK(hipMalloc(&d, h.size() * sizeof(T)));
  HIP_OK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

std::vector<int> Launch(bool before, const float* d_logits, int R, int V, int* d_out) {
  HIP_OK(hipMemcpy(d_out, std::vector<int>(R, kSentinel).data(), sizeof(int) * R, hipMemcpyHostToDevice));
  if (before) ArgmaxBefore<<<R, 1024>>>(d_logits, V, d_out);
  else ArgmaxKernel<<<R, 1024>>>(d_logits, V, d_out);
  HIP_OK(hipGetLastError());
  std::vector<int> out(R);
  HIP_OK(hipMemcpy(out.data(), d_out, sizeof(int) * R, hipMemcpyDeviceToHost));
  return out;
}

// Returns the number of failed rows.
int Check(const std::vector<Row>& rows, const std::vector<int>& got, const std::vector<int>& before, bool quiet) {
  int failed = 0;
  for (size_t r = 0; r < rows.size(); ++r) {
    const int want = Reference(rows[r].v);
    const bool finite = want != kNonfiniteId;
    if (got[r] != want || (finite && before[r] != got[r])) {
      ++failed;
      if (!quiet)
        std::fprintf(stderr, "FAIL row %zu (%s): got %d, want %d, before %d\n", r, rows[r].name.c_str(), got[r], want,
                     before[r]);
    }
  }
  return failed;
}

int Bench(int pairs) {
  const int R = 16, V = kVocab;
  std::vector<float> h;
  for (int r = 0; r < R; ++r) {
    const auto v = Random(V, 100 + r);
    h.insert(h.end(), v.begin(), v.end());
  }
  float* d = Device(h);
  int* out;
  HIP_OK(hipMalloc(&out, sizeof(int) * R));
  hipEvent_t a, b;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  constexpr int kLaunches = 200;
  auto time = [&](bool before, int rows) {
    HIP_OK(hipEventRecord(a));
    for (int i = 0; i < kLaunches; ++i) {
      if (before) ArgmaxBefore<<<rows, 1024>>>(d, V, out);
      else ArgmaxKernel<<<rows, 1024>>>(d, V, out);
    }
    HIP_OK(hipEventRecord(b));
    HIP_OK(hipEventSynchronize(b));
    float ms;
    HIP_OK(hipEventElapsedTime(&ms, a, b));
    return 1000.f * ms / kLaunches;
  };
  for (int rows : {1, 8, 16}) {
    time(true, rows), time(false, rows);  // warm-up
    double sum_before = 0, sum_after = 0;
    for (int p = 0; p < pairs; ++p) {
      const float tb = time(true, rows), ta = time(false, rows);
      sum_before += tb;
      sum_after += ta;
      std::printf("bench R=%d pair=%d before_us=%.3f after_us=%.3f\n", rows, p, tb, ta);
    }
    std::printf("bench R=%d mean before_us=%.3f after_us=%.3f ratio=%.4f\n", rows, sum_before / pairs,
                sum_after / pairs, sum_after / sum_before);
  }
  HIP_OK(hipFree(d));
  HIP_OK(hipFree(out));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--bench") return Bench(std::atoi(argv[2]));
  int failed = 0;
  for (int V : {kVocab, 4096, 1000}) {
    const auto rows = Rows(V);
    const int R = int(rows.size());
    std::vector<float> h;
    for (const auto& row : rows) h.insert(h.end(), row.v.begin(), row.v.end());
    float* d = Device(h);
    int* out;
    HIP_OK(hipMalloc(&out, sizeof(int) * R));
    const auto got = Launch(false, d, R, V, out);
    const auto before = Launch(true, d, R, V, out);
    failed += Check(rows, got, before, false);
    // A row with a NaN or +Inf gave a normal token before an earlier change: the old kernel must fail this check.
    if (Check(rows, before, before, true) == 0) {
      std::fprintf(stderr, "FAIL: V=%d the check accepts the kernel before an earlier change\n", V);
      ++failed;
    }
    // Mutant: one changed ID in a finite row must be rejected.
    auto mutant = got;
    mutant[0] ^= 1;
    if (Check(rows, mutant, before, true) == 0) {
      std::fprintf(stderr, "FAIL: V=%d the check accepts a mutant\n", V);
      ++failed;
    }
    // Each row alone (R = 1) gives the bits of the same row in the multi-row call.
    for (int r = 0; r < R; ++r) {
      const auto one = Launch(false, d + (size_t)r * V, 1, V, out);
      if (one[0] != got[r]) {
        std::fprintf(stderr, "FAIL: V=%d row %d alone %d, in the batch %d\n", V, r, one[0], got[r]);
        ++failed;
      }
    }
    HIP_OK(hipFree(d));
    HIP_OK(hipFree(out));
  }
  if (failed) return 1;
  std::puts("argmax_nonfinite_test: ok");
}
