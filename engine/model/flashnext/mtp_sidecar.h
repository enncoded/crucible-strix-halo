// Log text for the opt-in MTP draft sidecar (CRUCIBLE_MTP_SIDECAR). The full path names the user's home
// directory, so logs and captured gate evidence carry only the file name and its size.
#pragma once
#include <filesystem>
#include <string>
#include <system_error>

namespace crucible {
inline std::string MtpSidecarLabel(const std::string& path) {
  std::error_code error;
  const auto bytes = std::filesystem::file_size(path, error);
  const std::string name = std::filesystem::path(path).filename().string();
  return name + (error ? " (size unknown)" : " (" + std::to_string(bytes) + " bytes)");
}
}  // namespace crucible
