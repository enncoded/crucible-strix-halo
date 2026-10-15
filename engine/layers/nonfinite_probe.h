// Debug-build tools of an earlier change (CRUCIBLE_DEBUG_NONFINITE). The default build compiles nothing from this file.
// & 1, non-finite probes (layer B): kernels at sites that can drop a NaN without a trace set a site bit in one device
// word; the host reads and clears the word at the ids copy of each head (Engine::Head, AsyncHead::Finish, the prefill
// head) and ends the turn with an error that names the sites.
// & 4, inject probe: the plan that gen and batch --inject give to Engine::SetInject.
#pragma once
#if CRUCIBLE_DEBUG_NONFINITE & 1
#include <cstdio>
#include <stdexcept>
#include <string>
#include "../gfx1151/common.h"

namespace crucible {

enum NonfiniteSite : unsigned {
  kNfDecodeRoute = 1u,   // MoeRouteKernel: a NaN logit has rank 0 and races on sel[0] (ops.h)
  kNfPrefillRoute = 2u,  // PrefillMoeRouteWaveKernel: a NaN logit is never selected (prefill_moe.h)
  kNfQsaScore = 4u,      // QSA indexer: fmaxf(dot, 0) turns a NaN dot into 0 (qsa_long.h, kernels/prefill_qsa_*.h)
  kNfHidden = 8u,        // a hidden row after a layer (Engine::Dump hook)
};

__device__ unsigned g_nonfinite_sites;

// NaN or +Inf: the values that sampling.h rejects in logits.
__device__ inline bool NanOrPosInf(float x) { return !(x < INFINITY); }
// NaN, +Inf or -Inf.
__device__ inline bool Nonfinite(float x) { return !(fabsf(x) < INFINITY); }

__device__ inline void FlagNonfinite(unsigned site) { atomicOr(&g_nonfinite_sites, site); }

// One block per row: flags kNfHidden when a value of the row is not finite.
__global__ void __launch_bounds__(256) NonfiniteRowsKernel(const float* __restrict__ H, int ld, int width) {
  const float* row = H + (size_t)blockIdx.x * ld;
  bool bad = false;
  for (int i = threadIdx.x; i < width; i += 256) bad |= Nonfinite(row[i]);
  if (__syncthreads_or(bad) && threadIdx.x == 0) FlagNonfinite(kNfHidden);
}

inline void ScanNonfiniteRows(const float* H, int R, int ld, int width) {
  if (R > 0) NonfiniteRowsKernel<<<R, 256>>>(H, ld, width);
}

// Reads and clears the site word. The caller has synchronized the null stream.
inline unsigned TakeNonfiniteSites() {
  unsigned sites = 0;
  HIP_OK(hipMemcpyFromSymbol(&sites, HIP_SYMBOL(g_nonfinite_sites), sizeof(sites)));
  if (sites) {
    const unsigned zero = 0;
    HIP_OK(hipMemcpyToSymbol(HIP_SYMBOL(g_nonfinite_sites), &zero, sizeof(zero)));
  }
  return sites;
}

inline void RequireNoNonfiniteSites() {
  if (const unsigned sites = TakeNonfiniteSites()) {
    std::fprintf(stderr, "debug nonfinite probe: sites 0x%x\n", sites);
    throw std::invalid_argument("nonfinite model logits (debug probe sites " + std::to_string(sites) + ")");
  }
}

}  // namespace crucible
#endif

#if CRUCIBLE_DEBUG_NONFINITE & 4
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../model/flashnext/geometry.h"

namespace crucible {

// One NaN after layer `layer` of the first pass that holds a row of `stream` at a position of at least `pos`.
// hidden: the first value of that row. kv: the first K value at position `pos` of QSA layer `layer`. index: the first
// pooled indexer key of the block that ends at `pos` (pos % 4 == 3). gdn: the first value of the GDN state of layer
// `layer`. Spec: KIND,LAYER,POS[,STREAM]; an empty spec gives no plan.
struct InjectPlan { std::string kind; int layer = 0, pos = 0, stream = 0; bool done = false; };

inline std::optional<InjectPlan> ParseInjectPlan(const std::string& spec) {
  if (spec.empty()) return std::nullopt;
  std::stringstream in(spec);
  std::vector<std::string> f;
  for (std::string field; std::getline(in, field, ',');) f.push_back(field);
  if (f.size() < 3 || f.size() > 4) throw std::invalid_argument("--inject needs KIND,LAYER,POS[,STREAM]");
  InjectPlan p;
  p.kind = f[0];
  try {
    p.layer = std::stoi(f[1]);
    p.pos = std::stoi(f[2]);
    p.stream = f.size() == 4 ? std::stoi(f[3]) : 0;
  } catch (const std::logic_error&) {  // invalid_argument and out_of_range of std::stoi
    throw std::invalid_argument("--inject needs whole numbers for LAYER, POS and STREAM");
  }
  const bool known = p.kind == "hidden" || p.kind == "kv" || p.kind == "index" || p.kind == "gdn";
  if (!known || p.layer < 0 || p.layer >= kLayers || p.pos < 0 || p.stream < 0)
    throw std::invalid_argument("--inject: kind hidden|kv|index|gdn, layer 0.." + std::to_string(kLayers - 1));
  if (((p.kind == "kv" || p.kind == "index") && !IsQsa(p.layer)) || (p.kind == "gdn" && IsQsa(p.layer)))
    throw std::invalid_argument("--inject: kv and index need a QSA layer, gdn a GDN layer");
  if (p.kind == "index" && p.pos % 4 != 3) throw std::invalid_argument("--inject index: POS % 4 must be 3");
  return p;
}

}  // namespace crucible
#endif
