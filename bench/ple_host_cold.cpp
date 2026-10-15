// Times engine/model/flashnext/ple_host.h itself: a whole-file cold gather, the same gather hot,
// then decode-sized gathers. Usage: ple_host_cold FILE DATA_OFFSET ROW_BYTES ROWS IDS [drop]
// "drop" sends POSIX_FADV_DONTNEED for the whole file first (clean pages only).
#include "../engine/model/flashnext/ple_host.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace {
constexpr int kEos = 248044;
constexpr int64_t kMult[3] = {23703573157769, 20109073645365, 8052911324071};
constexpr int64_t kVocab[16] = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                                20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
constexpr int64_t kOffset[16] = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                                 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};

std::vector<int64_t> Rows(const std::vector<int>& hist, int pos0, int R) {
  std::vector<int64_t> rows((size_t)R * 16);
  for (int r = 0; r < R; ++r) {
    const int t = pos0 + r;
    int e = -1;
    for (int i = t - 1; i >= 0 && i >= t - 2; --i)
      if (hist[i] == kEos) { e = i; break; }
    auto tok = [&](int k) -> uint64_t {
      const int s = t - k;
      return (s < 0 || s <= e) ? (uint64_t)kEos : (uint64_t)hist[s];
    };
    const uint64_t h2 = (tok(0) * (uint64_t)kMult[0]) ^ (tok(1) * (uint64_t)kMult[1]);
    const uint64_t h3 = h2 ^ (tok(2) * (uint64_t)kMult[2]);
    for (int h = 0; h < 16; ++h) rows[r * 16 + h] = (int64_t)((h < 8 ? h2 : h3) % (uint64_t)kVocab[h]) + kOffset[h];
  }
  return rows;
}

double Ms(std::chrono::steady_clock::time_point a) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) return 2;
  const std::string path = argv[1];
  const uint64_t off = std::strtoull(argv[2], nullptr, 10);
  const size_t row = std::strtoull(argv[3], nullptr, 10);
  const int64_t nrows = std::strtoll(argv[4], nullptr, 10);
  std::ifstream f(argv[5]);
  std::stringstream ss;
  ss << f.rdbuf();
  std::vector<int> ids;
  std::string tok;
  for (char c : ss.str() + ",") {
    if (c >= '0' && c <= '9') tok += c;
    else if (!tok.empty()) { ids.push_back(std::stoi(tok)); tok.clear(); }
  }
  if (argc > 6) {
    const int fd = open(path.c_str(), O_RDONLY);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
  }
  crucible::PleHostTable t(path, row, {{0, nrows, off}});
  // Every prompt position except the last 64, which stand for decode steps.
  const int R = int(ids.size()) - 64;
  const auto rows = Rows(ids, 0, R);
  std::vector<uint8_t> out(rows.size() * row);
  for (const char* name : {"chunk first", "chunk again"}) {
    const auto a = std::chrono::steady_clock::now();
    t.Gather(rows.data(), rows.size(), out.data());
    std::printf("%s: %d rows in %.3f ms\n", name, R, Ms(a));
  }
  for (int i = 0; i < 64; ++i) {
    const auto d = Rows(ids, R + i, 1);
    const auto a = std::chrono::steady_clock::now();
    t.Gather(d.data(), d.size(), out.data());
    std::printf("decode %d: %.3f ms\n", i, Ms(a));
    usleep(20000);  // a decode step leaves the disk idle in between
  }
  return 0;
}
