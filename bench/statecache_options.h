#pragma once
#include <exception>
#include <optional>

namespace crucible {
inline constexpr int kSnapshotSmallRows = 64;
inline constexpr int kSnapshotBulkRows = 4096;

template<class Parse>
std::optional<int> SnapshotPrefill(int argc, const char* raw, Parse parse) {
  if ((argc != 3 && argc != 4) || (argc == 4 && !raw)) return std::nullopt;
  try {
    const int rows = argc == 3 ? kSnapshotSmallRows : parse(raw);
    if (rows == kSnapshotSmallRows || rows == kSnapshotBulkRows) return rows;
  } catch (const std::exception&) { return std::nullopt; }
  return std::nullopt;
}
template<class Parse>
std::optional<bool> SnapshotDense(const char* raw, Parse parse) {
  if (!raw) return true;
  try {
    const int flag = parse(raw);
    if (flag == 0 || flag == 1) return flag != 0;
  } catch (const std::exception&) { return std::nullopt; }
  return std::nullopt;
}
}  // namespace crucible
