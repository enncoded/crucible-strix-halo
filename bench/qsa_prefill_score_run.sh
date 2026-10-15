#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
bin=${1:?binary}
for ctx in 16384 65536 131072 262144; do
  "$bin" "$ctx" 2048 0 3
  "$bin" "$ctx" 37 1 2
  "$bin" "$ctx" 37 2 2
done
"$bin" 2052 2048 2 2
for ctx in 2048 2051 2052 2055 2056 32768 32769 32772 32775 65539 131071 262143; do
  "$bin" "$ctx" 7 2 2
done
for tier in 305 405; do
  for layer in 3 47; do
    "$bin" capture "$WORK_DIR/wt/longctx/results/qsa_audit.cQyrGL/$tier/n4095/qsa_$layer"
  done
done
