// Checked output for residual-stack dumps. Buffered failures can appear at close.
#pragma once
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace crucible {
[[noreturn]] inline void DumpError(const std::string& path, const char* operation, int error) {
  std::fprintf(stderr, "dump error: %s: cannot %s: %s; check the directory and free disk space\n",
               path.c_str(), operation, error ? std::strerror(error) : "incomplete output");
  std::exit(1);
}
inline void AppendDump(const std::string& path, const float* data, size_t count) {
  FILE* f = std::fopen(path.c_str(), "ab");
  if (!f) DumpError(path, "open file", errno);
  if (std::fwrite(data, sizeof(float), count, f) != count) {
    const int error = errno;
    std::fclose(f);
    DumpError(path, "write all rows", error);
  }
  if (std::fclose(f) != 0) DumpError(path, "close file", errno);
}
}  // namespace crucible
