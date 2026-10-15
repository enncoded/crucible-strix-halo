// Load weights once for both dump gates and the serial token reference.
#include <filesystem>
#include <stdexcept>

#define ENGINE_LIBRARY
#include "../engine/engine.hip"

namespace {
constexpr int kGenerated = 160;

std::vector<char> FileBytes(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

int Prefill(Engine& engine, const std::vector<int>& ids, int chunk, const std::string& dump) {
  engine.Reset();
  int last = -1;
  for (size_t i = 0; i < ids.size(); i += chunk) {
    const int rows = std::min<size_t>(chunk, ids.size() - i);
    last = engine.Forward(ids.data() + i, rows, 0, rows, dump).back();
  }
  return last;
}

void CompareDumps(const std::string& first, const std::string& second) {
  for (int i = 0; i <= kLayers; ++i) {
    char name[40];
    std::snprintf(name, sizeof(name), "/h_%02d.f32", i);
    if (FileBytes(first + name) != FileBytes(second + name))
      throw std::runtime_error("dump mismatch: " + first + name + " versus " + second + name);
  }
}

void Serial(Engine& engine, int next, const std::string& path) {
  std::vector<int> output{next};
  for (int i = 1; i < kGenerated; ++i)
    output.push_back(engine.Forward(&output.back(), 1, 0, 1)[0]);
  std::ofstream file(path);
  if (!file) throw std::runtime_error("cannot write " + path);
  file << "ids:";
  for (int token : output) file << ' ' << token;
  file << '\n';
  if (!file) throw std::runtime_error("write failed: " + path);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      std::fprintf(stderr, "usage: hc_engine_gates CKPT IDS OUTDIR BASELINE_DUMP\n");
      return 2;
    }
    const auto bytes = FileBytes(argv[2]);
    const auto ids = ParseIds(std::string(bytes.begin(), bytes.end()));
    if (ids.empty() || !ValidTokenIds(ids, kVocab)) throw std::runtime_error("invalid prompt tokens");
    const std::string out = argv[3], baseline = argv[4];
    if (std::filesystem::exists(out)) throw std::runtime_error("output directory already exists: " + out);
    std::filesystem::create_directories(out + "/c1");
    std::filesystem::create_directories(out + "/c8");
    Engine engine(argv[1], 32768);
    Prefill(engine, ids, 1, out + "/c1");
    const int next = Prefill(engine, ids, 8, out + "/c8");
    CompareDumps(out + "/c1", baseline);
    CompareDumps(out + "/c8", baseline);
    CompareDumps(out + "/c1", out + "/c8");
    Serial(engine, next, out + "/serial.stdout");
    std::printf("Gates 1a and 2: all 49 files match. Saved %d serial tokens.\n", kGenerated);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "gate error: %s\n", error.what());
    return 1;
  }
}
