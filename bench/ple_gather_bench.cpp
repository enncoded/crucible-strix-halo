// CPU cost of gathering PLE n-gram rows from the mmapped safetensors file.
// Build: g++ -O2 -std=c++20 -pthread bench/ple_gather_bench.cpp -o build/bench/ple_gather_bench
// Run:   ple_gather_bench FILE DATA_OFFSET ROW_BYTES IDS... (ids: comma-separated token files)
// It never writes the model file. "cold" drops only the touched pages with
// POSIX_FADV_DONTNEED, which other readers see as ordinary page-cache misses.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int kEos = 248044;
constexpr int64_t kMult[3] = {23703573157769, 20109073645365, 8052911324071};
constexpr int64_t kVocab[16] = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                                20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
constexpr int64_t kOffset[16] = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                                 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
constexpr size_t kPage = 4096;

std::vector<int> ReadIds(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  std::vector<int> ids;
  std::string tok;
  for (char c : ss.str()) {
    if (c >= '0' && c <= '9') tok += c;
    else if (!tok.empty()) { ids.push_back(std::stoi(tok)); tok.clear(); }
  }
  if (!tok.empty()) ids.push_back(std::stoi(tok));
  return ids;
}

// Same as Engine::PleRows.
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

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Table {
  int fd = -1;
  const uint8_t* base = nullptr;
  size_t size = 0, data = 0, row = 0;
  void Map() {
    base = (const uint8_t*)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); std::exit(1); }
    madvise((void*)base, size, MADV_RANDOM);
  }
  void Unmap() { munmap((void*)base, size); }
  size_t Off(int64_t r) const { return data + (size_t)r * row; }
};

void Gather(const Table& t, const std::vector<int64_t>& rows, uint8_t* out, int threads, bool willneed) {
  const size_t n = rows.size();
  auto work = [&](size_t b, size_t e) {
    if (willneed)
      for (size_t i = b; i < e; ++i) {
        const size_t o = t.Off(rows[i]), p0 = o / kPage * kPage, p1 = (o + t.row - 1) / kPage * kPage + kPage;
        madvise((void*)(t.base + p0), p1 - p0, MADV_WILLNEED);
      }
    for (size_t i = b; i < e; ++i) std::memcpy(out + i * t.row, t.base + t.Off(rows[i]), t.row);
  };
  if (threads <= 1) { work(0, n); return; }
  std::vector<std::thread> pool;
  for (int k = 0; k < threads; ++k) pool.emplace_back(work, n * k / threads, n * (k + 1) / threads);
  for (auto& th : pool) th.join();
}

void Drop(const Table& t, const std::vector<int64_t>& rows) {
  for (int64_t r : rows) {
    const size_t o = t.Off(r), p0 = o / kPage * kPage, p1 = (o + t.row - 1) / kPage * kPage + kPage;
    posix_fadvise(t.fd, (off_t)p0, (off_t)(p1 - p0), POSIX_FADV_DONTNEED);
  }
}

double Median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

void Case(Table& t, const char* name, const std::vector<int64_t>& rows, bool cold, int reps) {
  std::vector<uint8_t> out(rows.size() * t.row), ref(rows.size() * t.row);
  for (int threads : {1, 2, 4, 8, 16, 32}) {
    for (bool wn : {false, true}) {
      std::vector<double> ts;
      for (int k = 0; k < reps; ++k) {
        if (cold) { t.Unmap(); Drop(t, rows); t.Map(); }
        else {
          Gather(t, rows, ref.data(), 8, false);  // fault the pages in
          // Evict the rows from the CPU caches: page-cache hot, not L3 hot.
          static std::vector<uint8_t> flush(size_t(256) << 20);
          for (size_t i = 0; i < flush.size(); i += 64) flush[i] += 1;
        }
        const double a = Now();
        Gather(t, rows, out.data(), threads, wn);
        ts.push_back(Now() - a);
      }
      std::printf("%-26s rows %6zu %s threads %2d willneed %d: median %.3f ms\n", name, rows.size(),
                  cold ? "cold" : "hot ", threads, wn, Median(ts) * 1e3);
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: %s FILE DATA_OFFSET ROW_BYTES IDS...\n", argv[0]); return 2; }
  Table t;
  t.fd = open(argv[1], O_RDONLY);
  if (t.fd < 0) { perror("open"); return 1; }
  struct stat sb{};
  fstat(t.fd, &sb);
  t.size = sb.st_size;
  t.data = std::strtoull(argv[2], nullptr, 10);
  t.row = std::strtoull(argv[3], nullptr, 10);
  t.Map();
  for (int a = 4; a < argc; ++a) {
    const auto ids = ReadIds(argv[a]);
    const int R = std::min<int>(4096, ids.size());
    const auto rows = Rows(ids, 0, R);
    std::set<int64_t> distinct(rows.begin(), rows.end());
    std::set<size_t> pages;
    for (int64_t r : rows) {
      pages.insert(t.Off(r) / kPage);
      pages.insert((t.Off(r) + t.row - 1) / kPage);
    }
    std::printf("%s: %d tokens, %zu row reads, %zu distinct rows, %zu distinct pages\n", argv[a], R, rows.size(),
                distinct.size(), pages.size());
    const std::string chunk = std::string("chunk ") + std::to_string(R);
    Case(t, chunk.c_str(), rows, false, 9);
    Case(t, chunk.c_str(), rows, true, 3);
    // Decode: one token (16 rows) and an 8-row verify (128 rows), at the end of the prompt.
    for (int dr : {1, 8}) {
      const auto d = Rows(ids, R - dr, dr);
      const std::string name = std::string("decode R=") + std::to_string(dr);
      Case(t, name.c_str(), d, false, 51);
      Case(t, name.c_str(), d, true, 11);
    }
  }
  return 0;
}
