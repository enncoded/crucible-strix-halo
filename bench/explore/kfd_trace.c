// LD_PRELOAD shim: logs the KFD memory ioctls of one process and samples its
// KFD queue-eviction time and the global migration counters. Movable GPU
// mappings (USERPTR BOs, SVM ranges) are the only GPU-mapped host pages that
// compaction can migrate; this tool lists them with call stacks so each one
// can be tied to an engine allocation. Read-only: it changes no setting.
//
// Build: gcc -O2 -shared -fPIC -o build/bench/kfd_trace.so bench/explore/kfd_trace.c -ldl -lpthread
// Run:   KFD_TRACE=prefix LD_PRELOAD=$PWD/build/bench/kfd_trace.so CMD   (writes prefix.PID.log)
// Lines: A alloc, F free, V svm, Q queue, S sample (only when a counter moves),
// E KFD SMI event of this process (no root needed): "9 ns -pid node trigger" is
// a queue eviction with trigger 0 SVM, 1 USERPTR, 2 TTM, 3 SUSPEND; "a" a restore.
// Stack frames are "module+offset"; resolve with addr2line -f -C -e MODULE OFFSET.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <glob.h>
#include <linux/kfd_ioctl.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int (*real_ioctl)(int, unsigned long, void*);
static FILE* out;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned long other[256];
static volatile int stop;
static pthread_t sampler, smi;
static int kfd_fd = -1, smi_started;

static double Now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static void Frames(void) {
  void* f[28];
  int n = backtrace(f, 28);
  for (int i = 2; i < n; ++i) {
    Dl_info d;
    if (dladdr(f[i], &d) && d.dli_fname) {
      const char* s = strrchr(d.dli_fname, '/');
      fprintf(out, " %s+0x%lx", s ? s + 1 : d.dli_fname, (unsigned long)((char*)f[i] - (char*)d.dli_fbase));
      if (d.dli_sname) fprintf(out, "(%s)", d.dli_sname);
    } else {
      fprintf(out, " %p", f[i]);
    }
  }
  fputc('\n', out);
}

static const char* Kind(uint32_t flags) {
  if (flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) return "USERPTR";
  if (flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT) return "GTT";
  if (flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM) return "VRAM";
  if (flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) return "DOORBELL";
  if (flags & KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP) return "MMIO";
  return "OTHER";
}

static void Log(unsigned long req, void* arg, int ret) {
  pthread_mutex_lock(&mu);
  if (!out) {
    pthread_mutex_unlock(&mu);
    return;
  }
  if (req == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
    struct kfd_ioctl_alloc_memory_of_gpu_args* a = arg;
    fprintf(out, "A %.6f ret=%d kind=%s va=0x%llx size=%llu flags=0x%x handle=0x%llx off=0x%llx",
            Now(), ret, Kind(a->flags), (unsigned long long)a->va_addr, (unsigned long long)a->size,
            a->flags, (unsigned long long)a->handle, (unsigned long long)a->mmap_offset);
    if (a->flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM) fputc('\n', out); else Frames();
  } else if (req == AMDKFD_IOC_FREE_MEMORY_OF_GPU) {
    struct kfd_ioctl_free_memory_of_gpu_args* a = arg;
    fprintf(out, "F %.6f ret=%d handle=0x%llx\n", Now(), ret, (unsigned long long)a->handle);
  } else if (_IOC_NR(req) == _IOC_NR(AMDKFD_IOC_SVM)) {
    // libhsakmt adds the attribute array to the size field of the request.
    struct kfd_ioctl_svm_args* a = arg;
    fprintf(out, "V %.6f ret=%d op=%u start=0x%llx size=%llu nattr=%u", Now(), ret, a->op,
            (unsigned long long)a->start_addr, (unsigned long long)a->size, a->nattr);
    for (uint32_t i = 0; i < a->nattr && i < 8; ++i) fprintf(out, " %u:0x%x", a->attrs[i].type, a->attrs[i].value);
    Frames();
  } else if (req == AMDKFD_IOC_CREATE_QUEUE) {
    struct kfd_ioctl_create_queue_args* a = arg;
    fprintf(out, "Q %.6f ret=%d type=%u ring=0x%llx ring_size=%u\n", Now(), ret, a->queue_type,
            (unsigned long long)a->ring_base_address, a->ring_size);
  } else if (req == AMDKFD_IOC_SET_XNACK_MODE) {
    struct kfd_ioctl_set_xnack_mode_args* a = arg;
    fprintf(out, "X %.6f ret=%d xnack=%d\n", Now(), ret, a->xnack_enabled);
  } else {
    other[_IOC_NR(req) & 255]++;
  }
  fflush(out);
  pthread_mutex_unlock(&mu);
}

// Reads this process's KFD SMI events (evictions with their trigger, restores,
// SVM migrations and unmaps, VM faults) until the library unloads.
static void* Smi(void* gpu) {
  struct kfd_ioctl_smi_events_args a = {.gpuid = (uint32_t)(uintptr_t)gpu};
  if (real_ioctl(kfd_fd, AMDKFD_IOC_SMI_EVENTS, &a) != 0) {
    pthread_mutex_lock(&mu);
    if (out) fprintf(out, "# smi open failed\n");
    pthread_mutex_unlock(&mu);
    return NULL;
  }
  const uint64_t mask = KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_VMFAULT) |
                        KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_MIGRATE_START) |
                        KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_MIGRATE_END) |
                        KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_PAGE_FAULT_START) |
                        KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_QUEUE_EVICTION) |
                        KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_QUEUE_RESTORE) |
                        KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_UNMAP_FROM_GPU);
  if (write(a.anon_fd, &mask, sizeof mask) != (ssize_t)sizeof mask) return NULL;
  char buf[8192];
  while (!stop) {
    struct pollfd pf = {.fd = (int)a.anon_fd, .events = POLLIN};
    if (poll(&pf, 1, 200) <= 0) continue;
    const ssize_t n = read(a.anon_fd, buf, sizeof buf - 1);
    if (n <= 0) continue;
    buf[n] = 0;
    const double t = Now();
    pthread_mutex_lock(&mu);
    for (char* line = strtok(buf, "\n"); line && out; line = strtok(NULL, "\n")) fprintf(out, "E %.6f %s\n", t, line);
    if (out) fflush(out);
    pthread_mutex_unlock(&mu);
  }
  close(a.anon_fd);
  return NULL;
}

int ioctl(int fd, unsigned long req, ...) {
  va_list ap;
  va_start(ap, req);
  void* arg = va_arg(ap, void*);
  va_end(ap);
  if (!real_ioctl) real_ioctl = (int (*)(int, unsigned long, void*))dlsym(RTLD_NEXT, "ioctl");
  int ret = real_ioctl(fd, req, arg);
  if (out && _IOC_TYPE(req) == AMDKFD_IOCTL_BASE) {
    kfd_fd = fd;
    if (req == AMDKFD_IOC_ACQUIRE_VM && ret == 0 && !__atomic_exchange_n(&smi_started, 1, __ATOMIC_SEQ_CST)) {
      const uint32_t gpu = ((struct kfd_ioctl_acquire_vm_args*)arg)->gpu_id;
      pthread_create(&smi, NULL, Smi, (void*)(uintptr_t)gpu);
    }
    Log(req, arg, ret);
  }
  return ret;
}

static long long ReadLong(const char* path) {
  FILE* f = fopen(path, "r");
  long long v = -1;
  if (f) {
    if (fscanf(f, "%lld", &v) != 1) v = -1;
    fclose(f);
  }
  return v;
}

enum { kVm = 7 };
static const char* vm_names[kVm] = {"pgmigrate_success", "compact_migrate_scanned", "compact_daemon_migrate_scanned",
                                    "compact_stall", "compact_isolated", "thp_collapse_alloc", "thp_fault_alloc"};

static void ReadVm(long long* v) {
  char name[64];
  long long x;
  for (int i = 0; i < kVm; ++i) v[i] = -1;
  FILE* f = fopen("/proc/vmstat", "r");
  if (!f) return;
  while (fscanf(f, "%63s %lld", name, &x) == 2)
    for (int i = 0; i < kVm; ++i)
      if (!strcmp(name, vm_names[i])) v[i] = x;
  fclose(f);
}

// Samples every KFD_TRACE_MS (default 20) ms; logs a line when eviction time or migrations move.
static void* Sample(void* unused) {
  (void)unused;
  const char* ms_env = getenv("KFD_TRACE_MS");
  const int ms = ms_env ? atoi(ms_env) : 20;
  char evicted[256] = "";
  long long last_ev = -2, last[kVm], v[kVm];
  ReadVm(last);
  while (!stop) {
    if (!evicted[0]) {
      char pattern[128];
      glob_t g;
      snprintf(pattern, sizeof pattern, "/sys/class/kfd/kfd/proc/%d/stats_*/evicted_ms", getpid());
      if (glob(pattern, 0, NULL, &g) == 0 && g.gl_pathc > 0) snprintf(evicted, sizeof evicted, "%s", g.gl_pathv[0]);
      globfree(&g);
    }
    long long ev = evicted[0] ? ReadLong(evicted) : -1;
    ReadVm(v);
    if (ev != last_ev || v[0] != last[0] || v[3] != last[3] || v[5] != last[5]) {
      pthread_mutex_lock(&mu);
      fprintf(out, "S %.6f evicted_ms=%lld", Now(), ev);
      for (int i = 0; i < kVm; ++i) fprintf(out, " %s=%lld", vm_names[i], v[i] - last[i]);
      fputc('\n', out);
      fflush(out);
      pthread_mutex_unlock(&mu);
      last_ev = ev;
      memcpy(last, v, sizeof last);
    }
    usleep(ms * 1000);
  }
  return NULL;
}

__attribute__((constructor)) static void Start(void) {
  const char* prefix = getenv("KFD_TRACE");
  if (!prefix) return;
  char path[512];
  snprintf(path, sizeof path, "%s.%d.log", prefix, getpid());
  out = fopen(path, "w");
  if (!out) return;
  struct timespec b;
  clock_gettime(CLOCK_BOOTTIME, &b);
  fprintf(out, "# kfd_trace pid=%d t0=%.6f boottime=%.6f\n", getpid(), Now(), b.tv_sec + b.tv_nsec * 1e-9);
  pthread_create(&sampler, NULL, Sample, NULL);
}

__attribute__((destructor)) static void Finish(void) {
  if (!out) return;
  stop = 1;
  pthread_join(sampler, NULL);
  if (smi_started) pthread_join(smi, NULL);
  pthread_mutex_lock(&mu);
  fprintf(out, "# other kfd ioctls (nr:count)");
  for (int i = 0; i < 256; ++i)
    if (other[i]) fprintf(out, " 0x%x:%lu", i, other[i]);
  fprintf(out, "\n# end t=%.6f\n", Now());
  fclose(out);
  out = NULL;
  pthread_mutex_unlock(&mu);
}
