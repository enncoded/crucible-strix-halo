// Disk root for the disk-tier tests (engine/runtime/state_disk.h, engine/serve/batch_state_disk.h). They must run the real O_DIRECT
// path, so the root has to sit on a filesystem that takes O_DIRECT.
#pragma once
#include <fcntl.h>
#include <linux/magic.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

// A fresh, empty directory for one test process.
struct DiskTestDir {
  std::filesystem::path path;
  bool direct;  // true when `path` is on a filesystem that takes O_DIRECT
};

// True when `dir` is on a disk filesystem that takes O_DIRECT. tmpfs accepts O_DIRECT since Linux 6.6 but copies
// through its page cache, so it never pins the user pages as a disk does.
inline bool TakesDirect(const std::filesystem::path& dir) {
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  struct statfs info{};
  if (::statfs(dir.c_str(), &info) != 0 || info.f_type == TMPFS_MAGIC) return false;
  const auto probe = dir / "direct.probe";
  const int fd = ::open(probe.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0600);
  if (fd >= 0) ::close(fd);
  std::filesystem::remove(probe, error);
  return fd >= 0;
}

inline DiskTestDir FreshDiskTestDir(const std::filesystem::path& path, bool direct) {
  std::error_code error;
  std::filesystem::remove_all(path, error);
  std::filesystem::create_directories(path);
  return {path, direct};
}

// Picks the first base that takes O_DIRECT, in this order: STATE_DISK_TEST_DIR, TMPDIR, /tmp, $HOME/.cache.
// the test machine's /tmp is tmpfs, so a run with TMPDIR unset must not stop there. When no base takes O_DIRECT, the first
// base is used and `direct` is false. `name` keeps the directories of different tests apart.
inline DiskTestDir PickDiskTestDir(const std::string& name) {
  std::vector<std::filesystem::path> bases;
  for (const char* var : {"STATE_DISK_TEST_DIR", "TMPDIR"})
    if (const char* value = std::getenv(var)) bases.emplace_back(value);
  bases.emplace_back("/tmp");
  if (const char* home = std::getenv("HOME")) bases.push_back(std::filesystem::path(home) / ".cache");
  const std::string leaf = name + "." + std::to_string(::getpid());
  for (const auto& base : bases) {
    if (TakesDirect(base / leaf)) return FreshDiskTestDir(base / leaf, true);
    std::error_code error;
    std::filesystem::remove_all(base / leaf, error);
  }
  return FreshDiskTestDir(bases.front() / leaf, false);
}
