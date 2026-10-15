#!/bin/bash
# Build one bench probe without the ISA dump: bench/build_one.sh NAME [OUT] [extra hipcc flags...]
set -euo pipefail
cd "$(dirname "$0")/.."
n=$1; out=${2:-$1}; shift; [ $# -gt 0 ] && shift
mkdir -p build/bench
nice -n 10 /opt/rocm/bin/hipcc -O3 -std=c++20 --offload-arch=gfx1151 --rocm-path=/opt/rocm -I/opt/rocm/include -Wall \
  -Wno-unused-result "$@" -o "build/bench/$out" "bench/$n.hip"
echo "built $out"
