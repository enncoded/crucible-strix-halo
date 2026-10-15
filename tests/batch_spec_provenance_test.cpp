#include "../bench/batch_spec_provenance.h"
#include <cassert>
#include <unistd.h>
int main(int argc,char** argv) {
  const auto base=std::filesystem::temp_directory_path()/(std::string("spec-provenance-")+std::to_string(getpid()));
  std::filesystem::create_directories(base);
  const auto root=base/"capture";
  const auto source=std::string(64,'a');
  assert(spec_probe::SelfSha256().size()==64);
  spec_probe::Provenance(root,"alone",source,argc,argv);
  std::ifstream file(base/"capture_provenance.json");
  const std::string payload((std::istreambuf_iterator<char>(file)),{});
  assert(payload.find(spec_probe::SelfSha256())!=std::string::npos);
  assert(payload.find(source)!=std::string::npos);
  bool rejected=false;
  try {spec_probe::Provenance(root,"alone",source,argc,argv);} catch(const std::runtime_error&) {rejected=true;}
  assert(rejected);
  std::filesystem::remove_all(base);
}
