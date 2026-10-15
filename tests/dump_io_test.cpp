#include "../engine/runtime/dump_io.h"
#include <cassert>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

static void Rejected(const char* path) {
  const pid_t pid = fork();
  assert(pid >= 0);
  if (pid == 0) { const float value = 1.f; crucible::AppendDump(path, &value, 1); _exit(0); }
  int status;
  assert(waitpid(pid, &status, 0) == pid);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 1);
}
int main() {
  Rejected("/this-directory-does-not-exist-crucible/h_00.f32");
#ifdef __linux__
  Rejected("/dev/full");
#endif
  char path[] = "/tmp/crucible-dump-test-XXXXXX";
  const int fd = mkstemp(path);
  assert(fd >= 0);
  close(fd);
  const float values[] = {1.f, -2.f, 3.f};
  crucible::AppendDump(path, values, 3);
  crucible::AppendDump(path, values, 3);
  FILE* f = fopen(path, "rb");
  assert(f);
  float got[6];
  assert(fread(got, sizeof(float), 6, f) == 6);
  assert(memcmp(got, values, sizeof(values)) == 0);
  assert(memcmp(got + 3, values, sizeof(values)) == 0);
  fclose(f);
  unlink(path);
}
