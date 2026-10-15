#include "../engine/model/flashnext/weight_validation.h"
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  st::Checkpoint ck(argv[1]);
  ck.AddFile("ngram_embedding.safetensors");
  ck.AddFile("mtp_hyper_connection_mixer_patch.safetensors");
  crucible::ValidateCheckpoint(ck);
  std::puts("checkpoint CPU validation passed");
}
