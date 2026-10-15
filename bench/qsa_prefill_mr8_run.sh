#!/usr/bin/env bash
# One short kernel batch: OOB probe, the engine shape proof, and MR8 candidates.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
bin=${1:?binary}
probe=${2:?soffset probe}
"$probe"
for ctx in 16384 65536 131072 262144; do
  "$bin" "$ctx" 2048 0 3 mr8
  "$bin" "$ctx" 37 1 2 mr8
  "$bin" "$ctx" 37 2 2 mr8
done
"$bin" 2052 2048 2 2 mr8
for ctx in 2048 2051 2052 2055 2056 32768 32769 32772 32775 65539 131071 262143; do
  "$bin" "$ctx" 7 2 2 mr8
done
for tier in 305 405; do
  for layer in 3 47; do
    "$bin" capture "$WORK_DIR/wt/longctx/results/qsa_audit.cQyrGL/$tier/n4095/qsa_$layer" mr8
  done
done
for ctx in 16384 65536 131072; do
  for layer in 3 23 47; do
    "$bin" capture "results/engine/305/baseline82-$ctx/capture/qsa_$layer" mr8
  done
done
echo batch_complete
