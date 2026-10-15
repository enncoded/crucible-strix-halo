// CPU-visible decode capabilities. Dispatch and checkpoint validation share this list.
#pragma once
#include <array>
namespace crucible {
inline constexpr std::array kSupportedDecodeK{2, 3, 4, 5, 6};
constexpr bool SupportedDecodeK(int k) {
  for (const int supported : kSupportedDecodeK) if (k == supported) return true;
  return false;
}
}  // namespace crucible
