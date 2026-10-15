// The engine's sidecar log line names the file and its size, never the directory (it names the user).
#include "../engine/model/flashnext/mtp_sidecar.h"
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>

int main() {
  char dir[] = "/tmp/crucible-sidecar-label-XXXXXX";
  assert(mkdtemp(dir));
  const std::string path = std::string(dir) + "/side-c18.safetensors";
  std::ofstream(path, std::ios::binary) << std::string(1234, 'x');
  const std::string label = crucible::MtpSidecarLabel(path);
  assert(label == "side-c18.safetensors (1234 bytes)");
  assert(label.find(dir) == std::string::npos);
  assert(crucible::MtpSidecarLabel("plain.safetensors").rfind("plain.safetensors (", 0) == 0);
  std::filesystem::remove_all(dir);
  std::puts("PASS: sidecar label is the file name and its byte count");
}
