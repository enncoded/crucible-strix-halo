// GDN lanes scan: production GdnScanLanesKernel<true> against the prefetching
// GdnScanLanesPfKernel on the same inputs. y and the final state must match
// bit for bit; then ABAB event timing per 2048-row part.
// Usage: gdn_scan_pf [rows ...] (default 1 17 64 2047 2048)
#include "prefill_common.h"
#include "../engine/layers/kernels/prefill_gdn.h"
#include <algorithm>
#include <cstring>

using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;

namespace {
// Negative probe (2026-10-05 05:25): kept here so the
// result can be reproduced; the engine keeps GdnScanLanesKernel<true>.
// GdnScanLanesKernel<true> with the next row's inputs loaded before the
// current row's update. The scan waits about one memory latency per row
// (0.9 us per row per part), because the loads of a row only start after the
// previous update. The loads do not depend on S, so they can start one row
// early. Q selects whether q is also loaded early (more VGPRs, fewer waves).
// Every value and every operation is the same as in GdnScanLanesKernel<true>.
template <bool Q>
__global__ void __launch_bounds__(128) GdnScanLanesPfKernel(GdnPrefill a) {
  const int h=blockIdx.x>>2, lane=threadIdx.x&31, sub=lane&3;
  const int j=(blockIdx.x&3)*32+(threadIdx.x>>5)*8+(lane>>2);
  const int R=a.input.R;
  float S[32];
  float* dst=a.input.state+(size_t)h*128*128+(size_t)sub*32*128+j;
#pragma unroll
  for (int i=0;i<32;++i) S[i]=dst[i*128];
  const size_t rows=(size_t)a.chunks*64, base=(size_t)h*rows, kbase=(size_t)(h/3)*rows;
  float scale=1.f;
  float4 nk[8], nq[8];
  float ng, nb, nv;
  // Rows past R load row R-1 again: they may lie past the buffers.
  const auto load=[&](int r) {
    r=min(r,R-1);
    const float4* kp=reinterpret_cast<const float4*>(a.k+(kbase+r)*128+sub*32);
#pragma unroll
    for (int i=0;i<8;++i) nk[i]=kp[i];
    if constexpr (Q) {
      const float4* qp=reinterpret_cast<const float4*>(a.q+(kbase+r)*128+sub*32);
#pragma unroll
      for (int i=0;i<8;++i) nq[i]=qp[i];
    }
    ng=a.g[base+r]; nb=a.beta[base+r]; nv=a.v[(base+r)*128+j];
  };
  load(0);
#pragma unroll 1
  for (int r=0;r<R;++r) {
    const size_t b=base+r;
    float k[32],q[32];
    if constexpr (!Q) {
      const float4* qp=reinterpret_cast<const float4*>(a.q+(kbase+r)*128+sub*32);
#pragma unroll
      for (int i=0;i<8;++i) nq[i]=qp[i];
    }
#pragma unroll
    for (int i=0;i<8;++i) {
      const float4 kv=nk[i],qv=nq[i];
      k[4*i]=kv.x;k[4*i+1]=kv.y;k[4*i+2]=kv.z;k[4*i+3]=kv.w;
      q[4*i]=qv.x;q[4*i+1]=qv.y;q[4*i+2]=qv.z;q[4*i+3]=qv.w;
    }
    const float decay=expf(ng),beta=nb,v=nv;
    load(r+1);
    scale*=decay;
    if (scale<0x1p-20f) {  // uniform: every lane of the block has head h
#pragma unroll
      for (int i=0;i<32;++i) S[i]*=scale;
      scale=1.f;
    }
    float m0=0,m1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) { m0+=S[i]*k[i];m1+=S[i+1]*k[i+1]; }
    float mem=m0+m1;
    mem+=__shfl_xor(mem,1);mem+=__shfl_xor(mem,2);
    const float delta=beta*(v-mem*scale), d=delta/scale;
    float y0=0,y1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) {
      S[i]+=k[i]*d;S[i+1]+=k[i+1]*d;
      y0+=S[i]*q[i];y1+=S[i+1]*q[i+1];
    }
    float y=y0+y1;
    y+=__shfl_xor(y,1);y+=__shfl_xor(y,2);
    if (sub==0) a.y[b*128+j]=y*scale;
  }
#pragma unroll
  for (int i=0;i<32;++i) S[i]*=scale;
#pragma unroll
  for (int i=0;i<32;++i) dst[i*128]=S[i];
}

std::vector<float> Uniform(size_t n, float lo, float hi, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(lo, hi);
  std::vector<float> x(n);
  for (auto& v : x) v = u(rng);
  return x;
}

struct Inputs {
  int R, chunks;
  size_t rows;
  Buffer<float> q, k, v, g, beta, state0, state, y;
  explicit Inputs(int r)
      : R(r), chunks((r + 63) / 64), rows((size_t)chunks * 64),
        q(Random(16 * rows * 128, 0.0884f, 11)), k(Random(16 * rows * 128, 0.0884f, 12)),
        v(Random(48 * rows * 128, 1.f, 13)), g(Uniform(48 * rows, -1.5f, -0.002f, 14)),
        beta(Uniform(48 * rows, 0.f, 1.f, 15)), state0(Random(48 * 128 * 128, .01f, 16)),
        state(48 * 128 * 128), y(48 * rows * 128) {}
  GdnPrefill Args() {
    GdnPrefill a{};
    a.input.state = state.p; a.input.R = R; a.input.commit = R;
    a.q = q.p; a.k = k.p; a.v = v.p; a.g = g.p; a.beta = beta.p; a.y = y.p; a.chunks = chunks;
    return a;
  }
  void Reset() {
    HIP_OK(hipMemcpy(state.p, state0.p, state.n * sizeof(float), hipMemcpyDeviceToDevice));
    y.Zero();
  }
};

template <class Launch> std::pair<std::vector<float>, std::vector<float>> Run(Inputs& in, Launch launch) {
  in.Reset();
  launch(in.Args());
  CheckLaunch();
  return {in.y.Get(), in.state.Get()};
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// GDN_HEADS=n (diagnostic) launches only the first n heads, to separate the
// per-wave row latency from the number of resident waves.
const int kHeads = std::getenv("GDN_HEADS") ? std::stoi(std::getenv("GDN_HEADS")) : 48;
// Below the kill line (2026-10-05 06:05); bench only.
// GdnScanLanesKernel<true> with one 32-row key slice per wave. In the lanes
// kernel the 8 column groups of a wave load the same k and q addresses, 16
// b128 loads per lane and row, and those loads bound the scan. Here wave w
// owns key rows 32w..32w+31 for 32 columns (one per lane), so its k and q are
// wave-uniform and come through scalar loads as SGPR operands. The four
// partial sums of a column meet in LDS and are added in the lanes kernel's
// shuffle tree, (p0+p1)+(p2+p3), so every value keeps its bits.
// The y sum of row r waits one row: wave r%4 adds it after the next barrier.
// The restrict kernel arguments let the compiler prove that the kernel's
// stores do not alias k, q, g and beta, which scalar loads require.
__global__ void __launch_bounds__(128) GdnScanSgprKernel(GdnPrefill a, const float* __restrict__ kin,
    const float* __restrict__ qin, const float* __restrict__ gin, const float* __restrict__ bin) {
  __shared__ float memp[2][4][32], ypart[4][4][32];
  const int h=blockIdx.x>>2, lane=threadIdx.x&31;
  const int sub=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int j=(blockIdx.x&3)*32+lane, R=a.input.R;
  float S[32];
  float* dst=a.input.state+(size_t)h*128*128+(size_t)sub*32*128+j;
#pragma unroll
  for (int i=0;i<32;++i) S[i]=dst[i*128];
  const size_t rows=(size_t)a.chunks*64, base=(size_t)h*rows, kbase=(size_t)(h/3)*rows;
  float scale=1.f, last=1.f;
  // Sums the y partials of row r (wave r%4) and stores y*scale, as sub 0 does.
  const auto finish=[&](int r, float s) {
    if ((r&3)!=sub) return;
    const float y=(ypart[r&3][0][lane]+ypart[r&3][1][lane])+(ypart[r&3][2][lane]+ypart[r&3][3][lane]);
    a.y[(base+r)*128+j]=y*s;
  };
  // The next row's k, decay inputs and v load right after the S update, the
  // last use of k, so they arrive during the y chain and the next barrier.
  float k[32],q[32];
  float gn,bn,vn;
  const auto load=[&](int r) {
    const size_t b=base+r;
    const float* kp=kin+(kbase+r)*128+sub*32;
#pragma unroll
    for (int i=0;i<32;++i) k[i]=kp[i];
    gn=gin[b]; bn=bin[b]; vn=a.v[b*128+j];
  };
  load(0);
#pragma unroll 1
  for (int r=0;r<R;++r) {
    const float* qp=qin+(kbase+r)*128+sub*32;
    const float decay=expf(gn),beta=bn,v=vn;
    scale*=decay;
    if (scale<0x1p-20f) {  // uniform: every lane of the block has head h
#pragma unroll
      for (int i=0;i<32;++i) S[i]*=scale;
      scale=1.f;
    }
    float m0=0,m1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) { m0+=S[i]*k[i];m1+=S[i+1]*k[i+1]; }
    memp[r&1][sub][lane]=m0+m1;
#pragma unroll
    for (int i=0;i<32;++i) q[i]=qp[i];
    __syncthreads();
    if (r>0) finish(r-1,last);
    const float mem=(memp[r&1][0][lane]+memp[r&1][1][lane])+(memp[r&1][2][lane]+memp[r&1][3][lane]);
    const float delta=beta*(v-mem*scale), d=delta/scale;
#pragma unroll
    for (int i=0;i<32;++i) S[i]+=k[i]*d;
    if (r+1<R) load(r+1);
    float y0=0,y1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) { y0+=S[i]*q[i];y1+=S[i+1]*q[i+1]; }
    ypart[r&3][sub][lane]=y0+y1;
    last=scale;
  }
  __syncthreads();
  finish(R-1,last);
#pragma unroll
  for (int i=0;i<32;++i) S[i]*=scale;
#pragma unroll
  for (int i=0;i<32;++i) dst[i*128]=S[i];
}

inline void LaunchGdnScanSgpr(GdnPrefill a, hipStream_t stream=nullptr) {
  GdnScanSgprKernel<<<48*4,128,0,stream>>>(a,a.k,a.q,a.g,a.beta);
}

const auto kBase = [](GdnPrefill a) { GdnScanLanesKernel<true><<<kHeads * 4, 128>>>(a); };
const auto kPfK = [](GdnPrefill a) { GdnScanLanesPfKernel<false><<<kHeads * 4, 128>>>(a); };
const auto kPfKQ = [](GdnPrefill a) { GdnScanLanesPfKernel<true><<<kHeads * 4, 128>>>(a); };
const auto kSgpr = [](GdnPrefill a) {
  GdnScanSgprKernel<<<kHeads * 4, 128>>>(a, a.k, a.q, a.g, a.beta);
};
}  // namespace

int main(int argc, char** argv) {
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i) sizes.push_back(std::stoi(argv[i]));
  if (sizes.empty()) sizes = {1, 17, 64, 2047, 2048};
  if (kHeads < 1 || kHeads > 48) return 2;
  bool ok = true;
  for (int R : sizes) {
    if (R < 1 || R > 2048) return 2;
    Inputs in(R);
    const auto base = Run(in, kBase), pk = Run(in, kPfK), pkq = Run(in, kPfKQ), sg = Run(in, kSgpr);
    const bool same_sgpr = Same(base.first, sg.first) && Same(base.second, sg.second);
    const bool same = Same(base.first, pk.first) && Same(base.second, pk.second)
        && Same(base.first, pkq.first) && Same(base.second, pkq.second) && same_sgpr;
    ok = ok && same;
    std::vector<double> rb, rk, rkq, rs;
    for (int rep = 0; rep < 4; ++rep) {  // ABCABC; the state keeps evolving, which does not change the work
      rb.push_back(Time([&] { kBase(in.Args()); }, 5));
      rk.push_back(Time([&] { kPfK(in.Args()); }, 5));
      rkq.push_back(Time([&] { kPfKQ(in.Args()); }, 5));
      rs.push_back(Time([&] { kSgpr(in.Args()); }, 5));
    }
    const double ts = bench::Median(rs);
    const double tb = bench::Median(rb), tk = bench::Median(rk), tkq = bench::Median(rkq);
    std::printf("R=%d bitwise=%s sgpr_bitwise=%s base_us=%.1f pf_k_us=%.1f (%.3f) pf_kq_us=%.1f (%.3f) sgpr_us=%.1f (%.3f)\n",
                R, same ? "EQUAL" : "DIFFER", same_sgpr ? "EQUAL" : "DIFFER", tb, tk, tk / tb, tkq, tkq / tb, ts, ts / tb);
    std::fflush(stdout);
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
