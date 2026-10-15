#pragma once
#include <string>
#include <utility>
#include <vector>

namespace crucible {
// The MoE prefill GEMMs read weight tiles in engine order: PrefillGemmJob and
// PrefillUniform(Exl3GemmJob) do not carry Exl3GemmJob::front_s, and no MoE
// body instantiates the FRONT walk. Returns the (name, front split) pairs that
// are not in engine order, so the GEMM prefill path rejects them instead of
// reading front-order tiles as engine order.
inline std::string PrefillMoeFrontIssue(const std::vector<std::pair<std::string, int>>& front) {
  std::string bad;
  for (const auto& [name, split] : front)
    if (split != 0) bad += " " + name + "(front " + std::to_string(split) + ")";
  return bad;
}
}  // namespace crucible
