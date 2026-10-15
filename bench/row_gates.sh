#!/usr/bin/env bash
# Run under the shared GPU lock. Each checkpoint uses two short prompt dump runs.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cd "$(dirname "$0")/.."
engine=${1:-build/engine/crucible-v2}
checkpoint=${2:?checkpoint path required}
baseline=${3:?baseline dump directory required}
tag=${4:?unique result tag required}
ids=$WORK_FIB_IDS
out=results/row-gates-$tag
if [[ -e $out ]]; then echo "Use a new gate tag: $out already exists" >&2; exit 2; fi
mkdir -p "$out/chunk1" "$out/chunk8"
for rows in 1 8; do
  CRUCIBLE_TUNE_ROWS=1 "$engine" gen "$checkpoint" --ids-file "$ids" --n 1 --chunk "$rows" \
    --ctx 512 --dump "$out/chunk$rows" > "$out/chunk$rows.txt" 2>&1
done
n=0
for index in $(seq -w 0 48); do
  file=h_$index.f32
  test -s "$baseline/$file"
  cmp "$baseline/$file" "$out/chunk1/$file"
  cmp "$out/chunk1/$file" "$out/chunk8/$file"
  n=$((n+1))
done
echo "Gate1(a): $n/49 baseline files match; Gate2: $n/49 chunk1/chunk8 files match"
