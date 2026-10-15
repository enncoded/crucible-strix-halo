#include "prefill_common.h"
#include "../engine/layers/qsa_capture.h"
#include <filesystem>
#include <iterator>

using namespace crucible;
using namespace prefill_bench;
namespace fs = std::filesystem;

std::string Bytes(const fs::path& path) {
  std::ifstream in(path,std::ios::binary);
  if (!in) std::exit(1);
  return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}

int main() {
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM","0",0) != 0) return 2;
  char pattern[] = "/tmp/crucible-qsa-capture-XXXXXX";
  const char* tmp = mkdtemp(pattern);
  if (!tmp) return 2;
  const fs::path root(tmp), full=root/"full", index=root/"index";
  fs::create_directory(full); fs::create_directory(index);
  Buffer<float> q(Random(3*12288,1.f,31)), iq(Random(3*640,1.f,33));
  Buffer<float> pool(Random(QsaPoolFloats(4095),1.f,35));
  Buffer<__half> k(4095*512), v(4095*512), suh(6144);
  Buffer<int> sel(3*512), nsel(3);
  QsaArgs a{}; a.q=q.p; a.idx=iq.p; a.c={k.p,v.p,nullptr,pool.p};
  a.sel=sel.p; a.nsel=nsel.p; a.suh_o=suh.p; a.pos0=4092; a.R=3;
  setenv("CRUCIBLE_QSA_CAPTURE_AT","4095",1);
  setenv("CRUCIBLE_QSA_CAPTURE_INDEX_ONLY","1",1);
  unsetenv("CRUCIBLE_QSA_CAPTURE_DIR");
  CaptureQsa(a,3);
  if (!fs::is_empty(full) || !fs::is_empty(index)) return 1;
  setenv("CRUCIBLE_QSA_CAPTURE_DIR",full.c_str(),1);
  setenv("CRUCIBLE_QSA_CAPTURE_INDEX_ONLY","0",1);
  CaptureQsa(a,3);
  setenv("CRUCIBLE_QSA_CAPTURE_DIR",index.c_str(),1);
  setenv("CRUCIBLE_QSA_CAPTURE_INDEX_ONLY","1",1);
  CaptureQsa(a,3);
  for (const char* suffix : {".meta",".idx.f32",".pool.f32",".sel.i32",".nsel.i32"})
    if (Bytes(full/(std::string("qsa_3")+suffix)) != Bytes(index/(std::string("qsa_3")+suffix))) return 1;
  for (const char* suffix : {".q.f32",".k.f16",".v.f16",".suh.f16"})
    if (!fs::exists(full/(std::string("qsa_3")+suffix)) || fs::exists(index/(std::string("qsa_3")+suffix))) return 1;
  if (fs::exists(full/"qsa_3.capture-kind") || Bytes(index/"qsa_3.capture-kind") != "index-only\n") return 1;
  fs::remove_all(root);
  std::puts("PASS: index-only capture preserves shared file bytes and default capture files");
}
