#!/usr/bin/env bash
# Kernel-only batch: half-q union flash bitwise and ABBA timing against production.
set -euo pipefail
bin=${1:?binary}
captures=${2:?capture root containing main90-idx-CTX directories}
for past in 0 2048 14336; do
  "$bin" "synthetic:$past" 1024
done
for ctx in 16384 65536 131072; do
  for layer in 3 47; do
    "$bin" "$captures/main90-idx-$ctx/capture/qsa_$layer" 1024
  done
done
echo batch_complete
