// Vision encoder: one image -> bf16 embeddings [tokens][2560] on the device.
// The EXL3 linears reuse the prefill GEMM (PrefillLinear) in row chunks; plans are built once per
// chunk size with block 0 and retargeted to other blocks by swapping their job arrays.
#pragma once
#include <map>
#include "model.h"
#include "multimodal.h"
#include "../../exl3/prefill_linear.h"
#include "../../layers/kernels/vision.h"

namespace crucible {

class VisionEncoder {
 public:
  static constexpr int kChunk = 4096;
  // Each image size needs its own plans (about 200 small device arrays). A long-running server sees
  // many sizes, so the least recently used plans are freed; a rebuild costs a few milliseconds.
  static constexpr size_t kMaxPlans = 24;
  static_assert(kVisionMaxPixels / (kVisionUnit * kVisionUnit) <= kChunk, "CheckImages must enforce the merger limit");

  // partial and xh are the engine's GEMM workspaces. Encoding runs on the engine stream between
  // forwards, so it never overlaps another reader of them.
  VisionEncoder(const std::string& dir, float* partial, size_t partial_cap, __half* xh, size_t xh_cap)
      : partial_(partial), partial_cap_(partial_cap), xh_(xh), xh_cap_(xh_cap) {
    const auto t0 = std::chrono::steady_clock::now();
    Loader(dir, true).LoadVision(w_);
    std::fprintf(stderr, "vision tower %.2f GB loaded in %.1f s\n", w_.bytes / 1e9,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }

  size_t WeightBytes() const { return w_.bytes; }
  size_t ScratchBytes() const { return scratch_bytes_; }
  size_t PlanCount() const { return plans_.size(); }

  // Batch serve runs the encoder in slices between other sessions' commands.
  // Slice 0 embeds the patches, each middle slice runs BlocksPerSlice blocks, and the last slice is the merger.
  // Between slices only the residual stream x (owned by the caller) carries state, and every slice writes
  // the zero padding it reads, so slices of different images may interleave with each other and with
  // forwards. Each kernel sees the same inputs as in one pass, so the embeddings are bitwise equal.
  static constexpr int kSliceWork = 16384;  // patches times blocks: about 60-80 ms on gfx1151
  static int Patches(const ImageInput& image) { return image.GridH() * image.GridW(); }
  static int BlocksPerSlice(int n) { return std::clamp(kSliceWork / std::max(n, 1), 1, vision::kDepth); }
  static int Slices(const ImageInput& image) {
    const int per = BlocksPerSlice(Patches(image));
    return 2 + (vision::kDepth + per - 1) / per;
  }
  static size_t ResidualFloats(const ImageInput& image) { return size_t(Patches(image)) * vision::kHidden; }
  // The scratch that Reserve allocates for n patches and rgb_bytes of pixels (startup admission counts it).
  static size_t ScratchBytesFor(int n, size_t rgb_bytes) {
    const size_t N = size_t(n), C = vision::kHidden, chunk = std::min<size_t>(N, kChunk);
    const size_t npad = (N + 63) / 64 * 64, heads = size_t(vision::kHeads) * vision::kHeadPad * npad;
    return (2 * N * C + 2 * chunk * C + std::max(chunk * vision::kInter, N * C) +
            std::max(size_t(3) * kChunk * C, N / 4 * vision::kMergeIn)) * sizeof(float) +
           3 * heads * sizeof(__half) + rgb_bytes;
  }

  // rgb (image.Bytes() host bytes) is read by slice 0 only. Synchronous on the null stream's order.
  void EncodeSlice(const ImageInput& image, const uint8_t* rgb, float* x, int slice, uint16_t* out) {
    const int gh = image.GridH(), gw = image.GridW(), n = gh * gw;
    if (n / 4 > kChunk) throw std::runtime_error("an image above 4096 tokens is not supported");
    if (slice < 0 || slice >= Slices(image)) throw std::logic_error("vision slice out of range");
    Reserve(n, image.Bytes());
    const int npad = (n + 63) / 64 * 64;
    if (slice == 0) {
      // Build the plans before the launches: their uploads then do not wait for this image's kernels.
      ++epoch_;
      for (int r0 = 0; r0 < n; r0 += kChunk) PlansFor(std::min(kChunk, n - r0), 0);
      PlansFor(0, n / 4);
      if (!rgb) throw std::logic_error("vision slice 0 needs the pixels");
      HIP_OK(hipMemcpyAsync(rgb_, rgb, image.Bytes(), hipMemcpyHostToDevice, 0));
      __half* patches = reinterpret_cast<__half*>(attn_);
      const size_t elements = size_t(n) * vision::kPatchIn;
      vision::PatchifyKernel<<<(elements + 255) / 256, 256>>>(rgb_, image.width, gw, n, patches);
      vision::PosEmbedKernel<<<n, 128>>>(w_.pos, gh, gw, x);
      vision::PatchEmbedKernel<<<dim3((n + 15) / 16, vision::kHidden / 64), 128>>>(patches, n, w_.patch_w,
                                                                                   w_.patch_b, x);
      return;
    }
    const int per = BlocksPerSlice(n), first = (slice - 1) * per;
    if (first >= vision::kDepth) { Merger(x, n, out); return; }
    // Padded keys must be exact zeros: P = 0 times stale data must not become NaN. Another image may have
    // used these buffers since the last slice, so each slice clears them.
    HIP_OK(hipMemsetAsync(vt_, 0, size_t(vision::kHeads) * vision::kHeadPad * npad * sizeof(__half), 0));
    HIP_OK(hipMemsetAsync(qh_, 0, size_t(vision::kHeads) * npad * vision::kHeadPad * sizeof(__half), 0));
    HIP_OK(hipMemsetAsync(kh_, 0, size_t(vision::kHeads) * npad * vision::kHeadPad * sizeof(__half), 0));
    for (int b = first; b < std::min(first + per, vision::kDepth); ++b) Block(x, b, n, gw, npad);
  }

  // Writes image.Tokens() bf16 rows of 2560 to out: every slice in order, with the encoder's own residual.
  void Encode(const ImageInput& image, const uint8_t* rgb, uint16_t* out) {
    Reserve(Patches(image), image.Bytes());
    for (int slice = 0; slice < Slices(image); ++slice) EncodeSlice(image, rgb, x_, slice, out);
  }

 private:
  struct Group {
    PrefillLinear base;                       // built with block 0
    std::vector<std::pair<LinJob*, Exl3GemmJob*>> per;  // job arrays of every block
  };
  struct Plans {
    Group qkv, o, fc1, fc2, m1, m2;
    uint64_t used = 0;  // epoch of the last image that used these plans
  };

  static LinJob Job(const Lin& L, const float* src, int lds, float* dst, int ldd, int R) {
    LinJob j{};
    j.tiles = L.tiles; j.suh = L.suh; j.svh = L.svh;
    j.src = src; j.dst = dst;
    j.I = L.I; j.O = L.O; j.K = L.K; j.S = 1; j.R = R;
    j.lds = lds; j.ldd = ldd; j.epi = kEpiStore;
    return j;
  }

  // Same per-job device records as PrefillLinear::Build, for each block's weights.
  void BuildGroup(Group& g, const std::vector<std::vector<LinJob>>& blocks) {
    g.base.Build(blocks[0], PrefillGemmShape{8, true}, nullptr, partial_, partial_cap_, xh_, xh_cap_);
    for (const auto& jobs : blocks) {
      std::vector<LinJob> lin = g.base.jobs;
      std::vector<Exl3GemmJob> gemm;
      for (size_t i = 0; i < lin.size(); ++i) {
        lin[i].tiles = jobs[i].tiles; lin[i].suh = jobs[i].suh; lin[i].svh = jobs[i].svh;
        if (jobs[i].K != lin[i].K || jobs[i].I != lin[i].I || jobs[i].O != lin[i].O) std::abort();
        const LinJob& j = lin[i];
        gemm.push_back({j.tiles, j.xh, j.partial, nullptr, j.I, j.O, j.R, 1, j.K, j.R, 0});
      }
      g.per.push_back({PrefillUpload(lin), PrefillUpload(gemm)});
    }
  }

  void Run(const Group& g, int block) const {
    PrefillLinear p = g.base;
    p.linear = g.per[block].first;
    p.gemm = g.per[block].second;
    p.Run(true, true);
  }

  Plans& PlansFor(int R, int merged) {
    const auto key = std::make_pair(R, merged);
    if (auto it = plans_.find(key); it != plans_.end()) {
      it->second->used = epoch_;
      return *it->second;
    }
    if (plans_.size() >= kMaxPlans) EvictOldest();
    const auto t0 = std::chrono::steady_clock::now();
    auto p = std::make_unique<Plans>();
    if (R) {
      std::vector<std::vector<LinJob>> qkv, o, fc1, fc2;
      for (const auto& b : w_.blocks) {
        qkv.push_back({Job(b.q, ln_, vision::kHidden, t1_, vision::kHidden, R),
                       Job(b.k, ln_, vision::kHidden, t1_ + size_t(kChunk) * vision::kHidden, vision::kHidden, R),
                       Job(b.v, ln_, vision::kHidden, t1_ + size_t(2) * kChunk * vision::kHidden, vision::kHidden, R)});
        o.push_back({Job(b.o, ln_, vision::kHidden, y_, vision::kHidden, R)});
        fc1.push_back({Job(b.fc1, ln_, vision::kHidden, big_, vision::kInter, R)});
        fc2.push_back({Job(b.fc2, big_, vision::kInter, y_, vision::kHidden, R)});
      }
      BuildGroup(p->qkv, qkv); BuildGroup(p->o, o); BuildGroup(p->fc1, fc1); BuildGroup(p->fc2, fc2);
    }
    if (merged) {
      BuildGroup(p->m1, {{Job(w_.m1, big_, vision::kMergeIn, t1_, vision::kMergeIn, merged)}});
      BuildGroup(p->m2, {{Job(w_.m2, t1_, vision::kMergeIn, big_, vision::kOut, merged)}});
    }
    std::fprintf(stderr, "vision plan for %d rows (%d merged) built in %.1f ms\n", R, merged,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    p->used = epoch_;
    return *(plans_[key] = std::move(p));
  }

  // Frees the least recently used plans that the current image does not use.
  void EvictOldest() {
    auto oldest = plans_.end();
    for (auto it = plans_.begin(); it != plans_.end(); ++it)
      if (it->second->used < epoch_ && (oldest == plans_.end() || it->second->used < oldest->second->used))
        oldest = it;
    if (oldest == plans_.end()) return;
    HIP_OK(hipDeviceSynchronize());
    FreePlans(*oldest->second);
    plans_.erase(oldest);
  }

  static void FreePlans(Plans& p) {
    for (Group* g : {&p.qkv, &p.o, &p.fc1, &p.fc2, &p.m1, &p.m2}) {
      g->base.Free();
      for (auto& [lin, gemm] : g->per) { HIP_OK(hipFree(lin)); HIP_OK(hipFree(gemm)); }
    }
  }

  void Block(float* x, int b, int n, int gw, int npad) {
    const auto& w = w_.blocks[b];
    const size_t C = vision::kHidden;
    for (int r0 = 0; r0 < n; r0 += kChunk) {
      const int R = std::min(kChunk, n - r0);
      const Plans& p = PlansFor(R, 0);
      vision::LayerNormKernel<<<R, 128>>>(x + r0 * C, R, w.n1w, w.n1b, ln_);
      Run(p.qkv, b);
      vision::QkvPostKernel<<<dim3(R, vision::kHeads), 128>>>(t1_, t1_ + size_t(kChunk) * C,
          t1_ + size_t(2) * kChunk * C, w.bq, w.bk, w.bv, r0, R, gw, npad, qh_, kh_, vt_);
    }
    vision::AttentionKernel<<<dim3(npad / 64, vision::kHeads), 128>>>(qh_, kh_, vt_, n, npad, attn_);
    for (int r0 = 0; r0 < n; r0 += kChunk) {
      const int R = std::min(kChunk, n - r0);
      const Plans& p = PlansFor(R, 0);
      const size_t count = size_t(R) * C;
      HIP_OK(hipMemcpyAsync(ln_, attn_ + r0 * C, count * sizeof(float), hipMemcpyDeviceToDevice, 0));
      Run(p.o, b);
      vision::ResidualKernel<<<(count + 255) / 256, 256>>>(x + r0 * C, y_, w.bo, R);
      vision::LayerNormKernel<<<R, 128>>>(x + r0 * C, R, w.n2w, w.n2b, ln_);
      Run(p.fc1, b);
      const size_t inter = size_t(R) * vision::kInter;
      vision::GeluKernel<true><<<(inter + 255) / 256, 256>>>(big_, w.bfc1, R, vision::kInter);
      Run(p.fc2, b);
      vision::ResidualKernel<<<(count + 255) / 256, 256>>>(x + r0 * C, y_, w.bfc2, R);
    }
  }

  void Merger(const float* x, int n, uint16_t* out) {
    const int merged = n / 4;
    const Plans& p = PlansFor(0, merged);
    vision::LayerNormKernel<<<n, 128>>>(x, n, w_.mnw, w_.mnb, big_);
    Run(p.m1, 0);
    const size_t mid = size_t(merged) * vision::kMergeIn, outs = size_t(merged) * vision::kOut;
    vision::GeluKernel<false><<<(mid + 255) / 256, 256>>>(t1_, w_.bm1, merged, vision::kMergeIn);
    Run(p.m2, 0);
    vision::OutputKernel<<<(outs + 255) / 256, 256>>>(big_, w_.bm2, merged, out);
  }

  // Grow-only buffers. Growth drops the plans, which hold pointers into the old buffers.
  void Reserve(int n, size_t rgb_bytes) {
    if (n <= capacity_ && rgb_bytes <= rgb_capacity_) return;
    HIP_OK(hipDeviceSynchronize());
    for (auto& [key, p] : plans_) FreePlans(*p);
    plans_.clear();
    for (void* q : {(void*)x_, (void*)ln_, (void*)y_, (void*)big_, (void*)t1_, (void*)attn_, (void*)qh_, (void*)kh_,
                    (void*)vt_, (void*)rgb_})
      if (q) HIP_OK(hipFree(q));
    capacity_ = std::max(n, capacity_);
    rgb_capacity_ = std::max(rgb_bytes, rgb_capacity_);
    const size_t N = capacity_, C = vision::kHidden, chunk = std::min<size_t>(N, kChunk);
    const size_t npad = (N + 63) / 64 * 64, heads = size_t(vision::kHeads) * vision::kHeadPad * npad;
    scratch_bytes_ = 0;
    auto alloc = [&](auto*& p, size_t count) {
      using T = std::remove_reference_t<decltype(*p)>;
      HIP_OK(hipMalloc(&p, count * sizeof(T)));
      scratch_bytes_ += count * sizeof(T);
    };
    alloc(x_, N * C);
    alloc(attn_, N * C);
    alloc(ln_, chunk * C);
    alloc(y_, chunk * C);
    // big_: fc1 output of one chunk, the merger LayerNorm of every patch, or the merger output.
    alloc(big_, std::max(chunk * vision::kInter, N * C));
    // t1_: q, k, v of one chunk at kChunk strides, or the merger fc1 output.
    alloc(t1_, std::max(size_t(3) * kChunk * C, N / 4 * vision::kMergeIn));
    alloc(qh_, heads);
    alloc(kh_, heads);
    alloc(vt_, heads);
    alloc(rgb_, rgb_capacity_);
    if (scratch_bytes_ != ScratchBytesFor(capacity_, rgb_capacity_)) throw std::logic_error("vision scratch accounting");
    std::fprintf(stderr, "vision scratch for %d patches: %.1f MiB\n", capacity_, scratch_bytes_ / 1048576.0);
  }

  VisionW w_;
  float* partial_;
  size_t partial_cap_;
  __half* xh_;
  size_t xh_cap_;
  std::map<std::pair<int, int>, std::unique_ptr<Plans>> plans_;
  uint64_t epoch_ = 0;
  int capacity_ = 0;
  size_t rgb_capacity_ = 0, scratch_bytes_ = 0;
  float *x_ = nullptr, *attn_ = nullptr, *ln_ = nullptr, *y_ = nullptr, *big_ = nullptr, *t1_ = nullptr;
  __half *qh_ = nullptr, *kh_ = nullptr, *vt_ = nullptr;
  uint8_t* rgb_ = nullptr;
};
}  // namespace crucible
