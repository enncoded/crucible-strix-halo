#include "../bench/statecache_options.h"
#include <cassert>
#include <charconv>
#include <cstring>
#include <stdexcept>

int main() {
  int calls = 0;
  const auto parse = [&](const char* raw) {
    ++calls;
    int value = 0;
    const char* end = raw + std::strlen(raw);
    const auto result = std::from_chars(raw, end, value);
    if (result.ec != std::errc{} || result.ptr != end) throw std::invalid_argument("invalid row count");
    return value;
  };
  using crucible::SnapshotPrefill;
  assert(SnapshotPrefill(3, nullptr, parse) == 64 && calls == 0);
  assert(SnapshotPrefill(3, "ignored", parse) == 64 && calls == 0);
  assert(SnapshotPrefill(4, "4096", parse) == 4096 && calls == 1);
  assert(SnapshotPrefill(4, "64", parse) == 64 && calls == 2);
  for (const char* raw : {"", "0", "-1", "16", "8192", "64x", "4096.0", "999999999999999999"})
    assert(!SnapshotPrefill(4, raw, parse));
  const int parsed = calls;
  assert(!SnapshotPrefill(2, "64", parse));
  assert(!SnapshotPrefill(5, "64", parse));
  assert(!SnapshotPrefill(4, nullptr, parse));
  assert(calls == parsed);
  using crucible::SnapshotDense;
  assert(SnapshotDense(nullptr, parse) == true);
  assert(SnapshotDense("0", parse) == false);
  assert(SnapshotDense("1", parse) == true);
  for (const char* raw : {"", "-1", "2", "true", "1.0", "1x"}) assert(!SnapshotDense(raw, parse));
}
