#!/usr/bin/env bash
# Run this batch under the shared GPU lock and a 300-second timeout.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cd "$(dirname "$0")/.."
checkpoint=~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
ids=$WORK_FIB_IDS
out=$(mktemp -d results/pinned_gates.XXXXXX)
echo "$out"
binary=${1:-./build/crucible_pinned}
./build/pinned_transfer_test
for chunk in 1 8; do
  mkdir "$out/chunk$chunk"
  "$binary" gen "$checkpoint" --ids-file "$ids" --n 1 --chunk "$chunk" \
    --dump "$out/chunk$chunk" >"$out/chunk$chunk.log" 2>&1
done
count=0
for file in "$out/chunk1"/h_*.f32; do
  cmp "$file" "$out/chunk8/${file##*/}"
  cmp "$file" "results/cached_chunk1/${file##*/}"
  count=$((count+1))
done
[[ $count == 49 ]]
for spec in 0 2; do
  "$binary" gen "$checkpoint" --ids-file "$ids" --n 128 --chunk 8 \
    --spec "$spec" >"$out/spec$spec.log" 2>&1
done
grep '^ids:' "$out/spec0.log" >"$out/spec0.ids"
grep '^ids:' "$out/spec2.log" >"$out/spec2.ids"
cmp "$out/spec0.ids" "$out/spec2.ids"
grep '^prefill' "$out"/spec*.log
echo 'PASS: all 49 baseline and row-count dumps; all 128 spec IDs'
