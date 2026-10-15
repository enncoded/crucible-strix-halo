// Clipped last-row capture adapted from quality_probe, source 4bf0d90.
// The rename is kept for builds against engine sources older than train 20 (an unguarded main in gen_cli.h).
#define ENGINE_LIBRARY
#define main CrucibleCliMain
#ifndef CRUCIBLE_ENGINE_HEADER
#define CRUCIBLE_ENGINE_HEADER "../engine/engine.hip"
#endif
#include CRUCIBLE_ENGINE_HEADER
#undef main
#include "long_probe_runtime.h"
#include <bit>
#include <charconv>
#include <filesystem>
#include <fstream>

int StrictNumber(const char* value) {
  int result;
  const char* end = value+std::strlen(value);
  const auto parsed = std::from_chars(value,end,result);
  if (parsed.ec != std::errc{} || parsed.ptr != end) throw std::runtime_error("invalid probe integer");
  return result;
}

std::vector<int> ReadProbeIds(const char* path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open probe input");
  std::stringstream text; text << input.rdbuf();
  return ParseIds(text.str());
}

void WriteSchedule(const std::vector<crucible::longprobe::Block>& blocks) {
  std::printf("{\"event\":\"schedule\",\"kind\":\"clipped schedule\",\"blocks\":[");
  for (size_t i=0;i<blocks.size();++i)
    std::printf("%s[%zu,%d]",i ? "," : "",blocks[i].start,blocks[i].rows);
  std::printf("]}\n");
  std::fflush(stdout);
}

int main(int argc,char** argv) {
  try {
    if (argc != 7) return 2;
    static_assert(sizeof(float)==4 && std::endian::native==std::endian::little);
    const auto input = ReadProbeIds(argv[2]);
    const auto points = ReadProbeIds(argv[3]);
    const int bulk_rows = StrictNumber(argv[4]), context = StrictNumber(argv[5]);
    if (bulk_rows < 0 || bulk_rows > kMaxPrefillRows || context < 1 || context > 262144 ||
        !ValidTokenIds(input,kVocab) || !RequestFitsContext(input.size(),1,context))
      throw std::runtime_error("invalid probe dimensions or token IDs");
    crucible::longprobe::Validate(input,points,bulk_rows ? bulk_rows : 8,kVocab);
    const std::filesystem::path root(argv[6]);
    if (!std::filesystem::create_directory(root)) throw std::runtime_error("probe output must be new");
    std::ofstream binary(root/"logits.f32",std::ios::binary);
    if (!binary) throw std::runtime_error("cannot open probe output");
    Engine engine(argv[1],context,bulk_rows);
    engine.WarmPlans();
    const auto copy=[](const float* source,size_t count) {
      std::vector<float> values(count);
      HIP_OK(hipMemcpy(values.data(),source,count*sizeof(float),hipMemcpyDeviceToHost));
      return values;
    };
    const auto emit=[&](int position,int id,const std::vector<float>& values) {
      binary.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
      if (!binary) throw std::runtime_error("cannot write full probe row");
      std::printf("{\"event\":\"point\",\"position\":%d,\"argmax\":%d,\"copied_argmax\":true}\n",position,id);
      std::fflush(stdout);
    };
    const auto blocks=crucible::longprobe::Capture(engine,input,points,bulk_rows ? bulk_rows : 8,
                                                 bulk_rows != 0,kVocab,copy,emit);
    binary.close();
    if (!binary) throw std::runtime_error("cannot finish probe output");
    WriteSchedule(blocks);
  } catch(const std::exception& error) {
    std::fprintf(stderr,"long probe: %s\n",error.what());
    return 1;
  }
  return 0;
}
