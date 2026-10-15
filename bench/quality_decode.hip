// Repeated serial trials within one resident model allocation.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
int main(int argc, char** argv) {
  if (argc != 4) return 2;
  std::ifstream file(argv[2]);
  std::stringstream text;
  text << file.rdbuf();
  const auto ids = ParseIds(text.str());
  const int trials = std::stoi(argv[3]);
  if (ids.empty() || trials < 1 || trials > 8) return 2;
  Engine eng(argv[1], 32768);
  for (int trial = 0; trial < trials; ++trial) {
    eng.Reset();
    std::vector<int> last;
    for (size_t i = 0; i < ids.size(); i += 8) {
      const int R = std::min<size_t>(8, ids.size() - i);
      last = eng.Forward(ids.data() + i, R, 0, R);
    }
    std::vector<int> gen{last.back()};
    const double start = Now();
    for (int i = 1; i < 160; ++i) gen.push_back(eng.Forward(&gen.back(), 1, 0, 1)[0]);
    const double seconds = Now() - start;
    std::printf("trial %d: %.3f s, %.3f tok/s\nids:", trial, seconds, 159 / seconds);
    for (int id : gen) std::printf(" %d", id);
    std::printf("\n");
    std::fflush(stdout);
  }
}
