#!/bin/bash
# Run under gpu-run. One model load covers all three gates for one checkpoint.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cd "$(dirname "$0")/.."
if [[ $# != 4 ]]; then
  echo 'usage: quality_gates.sh TAG CKPT BASE SWEEP' >&2
  exit 2
fi
TAG=$1 C=$2 BASE=$3 SWEEP=$4
OUT="results/quality-gates-$TAG"
if [[ ! $TAG =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'use a nonempty gate tag with only letters, digits, dots, underscores, or hyphens' >&2
  exit 2
fi
mkdir -p results
# Reserve the tag before writing any captures or logs, including concurrent runs.
mkdir "$OUT" || exit 2
mkdir "$OUT/c1" "$OUT/c8"
IDS=$WORK_FIB_IDS
check_dumps() {
  python3 - "$IDS" "$@" <<'PY'
from pathlib import Path
import re
import sys
ids = [int(x) for x in re.split(r"[,\s]+", Path(sys.argv[1]).read_text().strip()) if x]
size = len(ids) * 10240 * 4
if not size:
    raise SystemExit("the gate prompt must contain token IDs")
expected = [f"h_{i:02d}.f32" for i in range(49)]
for directory in sys.argv[2:]:
    files = sorted(Path(directory).glob("h_*.f32"))
    if [file.name for file in files] != expected or any(
            not file.is_file() or file.is_symlink() or file.stat().st_size != size for file in files):
        raise SystemExit(f"expected 49 complete residual dumps of {size} bytes: {directory}")
PY
}
check_dumps "$BASE"
printf '%s\n' \
  "c1 --ids-file $IDS --n 1 --chunk 1 --spec 0 --dump $OUT/c1" \
  "c8 --ids-file $IDS --n 1 --chunk 8 --spec 0 --dump $OUT/c8" \
  "warm --ids-file $IDS --n 160 --spec 0" \
  "s0 --ids-file $IDS --n 160 --spec 0" \
  "s2 --ids-file $IDS --n 160 --spec 2" > "$OUT/config.txt"
"$SWEEP" "$C" "$OUT/config.txt" > "$OUT/run.log" 2>&1
check_dumps "$BASE" "$OUT/c1" "$OUT/c8"
for layer in $(seq -w 0 48); do
  cmp "$BASE/h_$layer.f32" "$OUT/c1/h_$layer.f32"
  cmp "$OUT/c1/h_$layer.f32" "$OUT/c8/h_$layer.f32"
done
if grep -q MISMATCH "$OUT/run.log"; then
  echo 'Gate 3 failed' >&2
  exit 1
fi
grep -E '^== s2: .* exact$' "$OUT/run.log"
echo 'Gate 1(a): 49/49. Gate 2: 49/49. Gate 3: exact.'
grep '^==' "$OUT/run.log"
