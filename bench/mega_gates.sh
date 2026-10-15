#!/bin/bash
# Run through gpu-run. Each checkpoint uses its own baseline and output tag.
# Usage: bash bench/mega_gates.sh CKPT BASE TAG
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cd "$(dirname "$0")/.."
CKPT=$1
BASE=$2
TAG=$3
IDS=$WORK_FIB_IDS
OUT=results/gate-$TAG
export CRUCIBLE_DECODE_RING=1
for CHUNK in 1 8; do
  mkdir "$OUT-$CHUNK"
  build/engine/crucible gen "$CKPT" --ids-file "$IDS" --n 1 --chunk "$CHUNK" --dump "$OUT-$CHUNK" > "$OUT-$CHUNK.log" 2>&1
done
COUNT=0
for F in "$BASE"/h_*.f32; do
  NAME=$(basename "$F")
  cmp "$F" "$OUT-1/$NAME"
  cmp "$OUT-1/$NAME" "$OUT-8/$NAME"
  COUNT=$((COUNT + 1))
done
test "$COUNT" = 49
echo "Gate 1(a): 49/49 baseline files equal. Gate 2: 49/49 row-count files equal."
printf 'serial --ids-file %s --n 160 --spec 0\nspec2 --ids-file %s --n 160 --spec 2\n' "$IDS" "$IDS" > "$OUT-spec.cfg"
build/bench/spec_sweep "$CKPT" "$OUT-spec.cfg" > "$OUT-spec.txt" 2>&1
grep '^==' "$OUT-spec.txt"
test "$(grep -c ', exact$' "$OUT-spec.txt")" = 1
! grep -q MISMATCH "$OUT-spec.txt"
echo "Gate 3: speculative IDs equal serial IDs."
