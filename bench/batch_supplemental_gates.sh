#!/usr/bin/env bash
# Queue this complete short job with gpu-run. Shared primary gates run separately.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cd "$(dirname "$0")/.."
if (( $# != 3 )); then
  echo 'usage: batch_supplemental_gates.sh CHECKPOINT NAMED_BASELINE OUT' >&2
  exit 2
fi
checkpoint=$1
baseline=$2
output=$3
export HSA_USERPTR_FOR_PAGED_MEM=0
mkdir "$output"
mkdir "$output/baseline"
"$baseline" gen "$checkpoint" --ids-file "$WORK_FIB_IDS" \
  --n 1 --chunk 8 --ctx 4096 --spec 0 --dump "$output/baseline" \
  > "$output/baseline.log" 2>&1
build/bench/batch_gates-f4f8e97 "$checkpoint" results/routing-paths.txt "$output/resident" "$output/baseline" \
  > "$output/resident.log" 2>&1
build/bench/batch_snapshot_gate-f4f8e97 "$checkpoint" "$WORK_FIB_IDS" "$output/snapshot" \
  > "$output/snapshot.log" 2>&1
# The parent permits profile collection before primary acceptance. Claims stay gated.
build/bench/batch_profile-f4f8e97 "$checkpoint" results/routing-paths.txt \
  > "$output/profile.log" 2>&1
