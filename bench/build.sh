#!/bin/bash
# Build the calibration probes for gfx1151. Usage: bench/build.sh [name ...]
# Binaries go to build/bench/, device ISA to build/isa/ for instruction checks.
set -euo pipefail
cd "$(dirname "$0")/.."
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
mkdir -p build/bench build/isa
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  for f in bench/*.hip; do
    # These probes must not silently use a different production reference.
    case "$f" in bench/isa_hc_regs.hip|bench/isa_hc_pitch.hip|bench/isa_hc_bounds.hip) continue ;; esac
    names+=("$(basename "$f" .hip)")
  done
fi
for n in "${names[@]}"; do
  case "$n" in
    isa_hc_regs|isa_hc_pitch|isa_hc_bounds)
      echo "$n requires a pinned ISA_HC_REFERENCE. Set it to the quoted path of a pinned engine/layers/prefill_hc.h (see bench/isa_hc_ws.h)." >&2
      exit 2 ;;
  esac
  "$HIPCC" -O3 -std=c++20 --offload-arch=gfx1151 --rocm-path=/opt/rocm -I/opt/rocm/include -Wall -Wno-unused-result \
    -o "build/bench/$n" "bench/$n.hip"
  "$HIPCC" -O3 -std=c++20 --offload-arch=gfx1151 --rocm-path=/opt/rocm -I/opt/rocm/include --cuda-device-only -S \
    -o "build/isa/$n.s" "bench/$n.hip" 2>/dev/null || true
  echo "built $n"
done
