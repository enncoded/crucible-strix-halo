// Included after Group and the allocation helpers in engine.hip.
// A separate head workspace lets known MTP catch-up rows run during verification.
#pragma once

class AsyncHead {
 public:
  AsyncHead() {
    x_ = Scratch<float>(kMaxRows * kHidden);
    stacks_ = Scratch<float>(kMaxRows * kWide);
    logits_ = Scratch<float>((size_t)kMaxRows * kVocab);
    partial_ = Scratch<float>((size_t)kMaxRows * kVocab);
    xh_ = Scratch<__half>(kMaxRows * kHidden);
    ids_ = Zeros<int>(kMaxRows);
    HIP_OK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
    HIP_OK(hipEventCreateWithFlags(&ready_, hipEventDisableTiming));
  }

  ~AsyncHead() {
    HIP_OK(hipStreamDestroy(stream_));
    HIP_OK(hipEventDestroy(ready_));
    for (const auto& [r, group] : groups_) HIP_OK(hipFree(group.d_jobs));
    for (auto* p : {x_, stacks_, logits_, partial_}) HIP_OK(hipFree(p));
    HIP_OK(hipFree(xh_));
    HIP_OK(hipFree(ids_));
  }

  void Begin(const Group& source, const float* x, const float* stacks, int R) {
    if (!groups_.count(R)) {
      Group g = source;
      if (g.jobs.size() != 1 || g.jobs[0].S != 1) std::abort();
      g.jobs[0].src = x_;
      g.jobs[0].xh = xh_;
      g.jobs[0].partial = partial_;
      g.jobs[0].dst = logits_;
      g.d_jobs = Upload(g.jobs);
      // A front-loaded head (d_front == d_jobs) has only front-order tiles.
      g.d_front = source.d_front == source.d_jobs ? g.d_jobs : nullptr;
      groups_[R] = g;
    }
    HIP_OK(hipMemcpyAsync(x_, x, sizeof(float) * R * kHidden, hipMemcpyDeviceToDevice, 0));
    HIP_OK(hipMemcpyAsync(stacks_, stacks, sizeof(float) * R * kWide, hipMemcpyDeviceToDevice, 0));
    HIP_OK(hipEventRecord(ready_, 0));
    HIP_OK(hipStreamWaitEvent(stream_, ready_, 0));
    const Group& g = groups_.at(R);
    LinInKernel<<<(g.nin + 7) / 8, 256, 0, stream_>>>(g.d_jobs, g.d_in, g.nin);
    switch (R) {
      case 1: Gemv<1>(g); break;
      case 2: Gemv<2>(g); break;
      case 3: Gemv<3>(g); break;
      case 4: Gemv<4>(g); break;
      case 5: Gemv<5>(g); break;
      case 6: Gemv<6>(g); break;
      case 7: Gemv<7>(g); break;
      default: Gemv<8>(g); break;
    }
    LinOutKernel<<<(g.nout + 7) / 8, 256, 0, stream_>>>(g.d_jobs, g.d_out, g.nout);
    ArgmaxKernel<<<R, 1024, 0, stream_>>>(logits_, kVocab, ids_);
  }

  const float* Stacks() const { return stacks_; }

  std::vector<int> Finish(float* stacks, int R) {
    HIP_OK(hipStreamSynchronize(stream_));
    HIP_OK(hipMemcpyAsync(stacks, stacks_, sizeof(float) * R * kWide, hipMemcpyDeviceToDevice, 0));
    std::vector<int> ids(R);
    HIP_OK(hipMemcpy(ids.data(), ids_, sizeof(int) * R, hipMemcpyDeviceToHost));
#if CRUCIBLE_DEBUG_NONFINITE & 1
    RequireNoNonfiniteSites();
#endif
    // These are the target rows of a verification pass, not drafts.
    sampling::RequireFiniteHeadIds(ids);
    return ids;
  }

 private:
  template <int R> void Gemv(const Group& g) {
    if (g.d_front) LinGemvKernel<R, 6, true><<<g.nitems, 256, g.lds, stream_>>>(g.d_jobs, g.d_items, g.nitems);
    else LinGemvKernel<R><<<g.nitems, 256, g.lds, stream_>>>(g.d_jobs, g.d_items, g.nitems);
  }

  std::map<int, Group> groups_;
  float *x_, *stacks_, *logits_, *partial_;
  __half* xh_;
  int* ids_;
  hipStream_t stream_;
  hipEvent_t ready_;
};
