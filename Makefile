# Public build entry point. GPU execution is always explicit.
.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
ROCM_PATH ?= /opt/rocm
HIPCC ?= $(ROCM_PATH)/bin/hipcc
GPU_ARCH ?= gfx1151
PYTHON ?= python3
CXXFLAGS ?= -O2 -g
HIPFLAGS ?= -O3
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?= -pthread
SANITIZERS ?=

# These probes require an archived reference header, not the current engine.
PINNED_HC_SOURCES := bench/isa_hc_regs.hip bench/isa_hc_pitch.hip bench/isa_hc_bounds.hip
PINNED_HC_BINARIES := $(patsubst %.hip,build/%,$(PINNED_HC_SOURCES))
# bench/split2048.hip includes split2048_prompt.h, which tools/split2048_probe.py generates.
# The probe refuses an existing --output dir, so make cannot run it. Build that binary with the probe's --build path.
HIP_SOURCES := engine/engine.hip $(filter-out $(PINNED_HC_SOURCES) bench/split2048.hip,$(wildcard bench/*.hip bench/suite/*.hip tests/*.hip))
HIP_BINARIES := $(patsubst %.hip,build/%,$(filter-out engine/engine.hip,$(HIP_SOURCES))) build/engine/crucible
# The Python runner generates the driver header and supplies its CPU engine stubs.
CPU_SOURCES := $(filter-out tests/gen_cli_cpu_test.cpp tests/gen_jobs_cpu_test.cpp,$(wildcard tests/*.cpp bench/*.cpp bench/*.cc))
CPU_BINARIES := $(addprefix build/,$(basename $(CPU_SOURCES)))
CPU_TESTS := $(filter-out build/tests/draft_vocab_test build/tests/gen_cli_cpu_test build/tests/gen_jobs_cpu_test,$(patsubst %.cpp,build/%,$(wildcard tests/*_test.cpp)))
CPU_TESTS += build/bench/static_layout build/bench/test_migration_wait
# Recursive, so that the module layout of tools/layout_map.tsv needs no edit here.
rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))
HEADERS := $(call rwildcard,engine kernels bench/kernels,*.h *.inc) $(wildcard bench/*.h bench/suite/*.h)
DEPFILES := $(addsuffix .d,$(HIP_BINARIES) $(PINNED_HC_BINARIES) $(CPU_BINARIES))

.PHONY: all engine binaries benchmarks cpu-tests gpu-tests test test-cpu test-reference test-gpu help
all: engine
engine: build/engine/crucible
binaries: $(HIP_BINARIES) $(CPU_BINARIES)
benchmarks: $(filter build/bench/%,$(HIP_BINARIES) $(CPU_BINARIES))
cpu-tests: $(filter build/tests/%,$(CPU_BINARIES)) $(CPU_TESTS)
gpu-tests: $(filter build/tests/%,$(HIP_BINARIES))

# The bench provenance hash uses OpenSSL; production binaries do not link it.
build/tests/batch_spec_provenance_test build/bench/batch_spec_probe: LDLIBS += -lcrypto
# The probe stops at #error without its source digest. Use the digest that tools/build_batch_gates.py records for
# the same source set, so the probe and the gates report one digest. A caller can pin another digest in the environment.
# The shell runs only when a recipe needs the digest.
SPEC_SOURCE_DIGEST = $(strip $(or $(CRUCIBLE_BATCH_SPEC_SOURCE_SHA),$(shell $(PYTHON) -I -c 'import sys;sys.path.insert(0,"tools");import build_batch_gates as g;print(g.source_digest(g.source_hashes(g.ROOT)))')))
# An empty or malformed digest stops the build here. Otherwise the probe compiles with a bad provenance value.
SPEC_SOURCE_DIGEST_CHECKED = $(if $(shell printf '%s\n' '$(SPEC_SOURCE_DIGEST)' | grep -Ex '[0-9a-f]{64}'),$(SPEC_SOURCE_DIGEST),$(error CRUCIBLE_BATCH_SPEC_SOURCE_SHA must be 64 lowercase hex characters, got '$(SPEC_SOURCE_DIGEST)'))
build/bench/batch_spec_probe: CPPFLAGS += -DCRUCIBLE_BATCH_SPEC_SOURCE_SHA='"$(SPEC_SOURCE_DIGEST_CHECKED)"'

build/engine/crucible: engine/engine.hip $(HEADERS) Makefile
	@mkdir -p $(@D)
	$(HIPCC) $(CPPFLAGS) $(HIPFLAGS) -std=c++20 --offload-arch=$(GPU_ARCH) --rocm-path=$(ROCM_PATH) -I$(ROCM_PATH)/include -MMD -MP -MF $@.d $< $(LDFLAGS) $(LDLIBS) -o $@

build/%: %.hip $(HEADERS) engine/engine.hip Makefile
	@mkdir -p $(@D)
	$(HIPCC) $(CPPFLAGS) $(HIPFLAGS) -std=c++20 --offload-arch=$(GPU_ARCH) --rocm-path=$(ROCM_PATH) -I$(ROCM_PATH)/include -MMD -MP -MF $@.d $< $(LDFLAGS) $(LDLIBS) -o $@

build/%: %.cpp $(HEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SANITIZERS) -std=c++20 -UNDEBUG -MMD -MP -MF $@.d $< $(LDFLAGS) $(LDLIBS) -o $@

build/%: %.cc $(HEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SANITIZERS) -std=c++20 -UNDEBUG -MMD -MP -MF $@.d $< $(LDFLAGS) $(LDLIBS) -o $@

test: test-cpu
test-cpu: cpu-tests
	@set -eu; for test in $(CPU_TESTS); do echo "TEST $$test"; "$$test"; done
	build/tests/draft_vocab_test tests/data/draft/blocks65536.txt
	CXX="$(CXX)" $(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) -m pytest -q server/tests
	$(PYTHON) -m unittest discover -s bench/suite -p 'test_*.py'
	$(PYTHON) -m pytest -q ref/test_exl3_ref.py ref/test_prefill.py ref/test_vision_ref.py

# PyTorch CPU is an optional dependency. No fp64 model oracle runs here.
test-reference:
	$(PYTHON) -m pytest -q ref/test_quality_formats.py ref/test_vision_hf.py

# The gate tool owns GPU scheduling, fresh artifacts, and the CPU oracle lock.
test-gpu: engine
	@test -n "$(GATE_ARGS)" || { echo 'Set GATE_ARGS; see python3 tools/gates.py --help.' >&2; exit 2; }
	$(PYTHON) tools/gates.py $(GATE_ARGS)

help:
	@printf '%s\n' 'make: build the engine' 'make binaries: build standard C++ and HIP binaries' 'make test-cpu: run offline CPU tests' 'make test-reference: run optional PyTorch CPU codec tests' 'make test-gpu GATE_ARGS="...": run shared manual gates' 'See docs/BUILD.md for dependencies and options.'

-include $(DEPFILES)
