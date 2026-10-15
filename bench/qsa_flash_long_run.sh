#!/usr/bin/env bash
# Kernel-only batch: union flash on real 3.05 selections at 16K, 64K, and 128K.
set -euo pipefail
bin=${1:?binary}
captures=${2:?capture root containing main90-idx-CTX directories}
for ctx in 16384 65536 131072; do
  for layer in 3 47; do
    "$bin" "$captures/main90-idx-$ctx/capture/qsa_$layer" 1024
  done
done
"$bin" "$captures/main90-idx-131072/capture/qsa_3" 8192
echo batch_complete
