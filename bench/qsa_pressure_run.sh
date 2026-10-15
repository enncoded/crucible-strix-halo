#!/usr/bin/env bash
# Call under the shared GPU lock and timeout. One continuous run has no restores.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cd "$(dirname "$0")/.."
[[ $# == 3 && "$2" =~ ^[a-zA-Z0-9_]+$ && "$3" =~ ^[01]$ ]] || exit 2
binary=$1
tag=$2
pressure=$3
checkpoint=~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
ready="results/${tag}.ready.$$"
python bench/qsa_memwatch.py "results/${tag}_memory.jsonl" & monitor=$!
oracle=
cleanup() {
  kill "$monitor" 2>/dev/null || true
  wait "$monitor" 2>/dev/null || true
  if [[ -n "$oracle" ]]; then
    kill "$oracle" 2>/dev/null || true
    wait "$oracle" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if [[ "$pressure" == 1 ]]; then
  (
    for i in $(seq 1 1200); do [[ -f "$ready" ]] && break; sleep .1; done
    [[ -f "$ready" ]] || exit 1
    date -Is
    nice -n 5 env OMP_NUM_THREADS=8 $WORK_REFENV_PYTHON \
      ref/oracle.py "$checkpoint" results/cached_chunk1 3 --pos 0,14,23 \
      --ids $WORK_FIB_IDS
    date -Is
  ) >"results/${tag}_oracle.log" 2>&1 & oracle=$!
fi
CRUCIBLE_QSA_MIGRATION_SETTLE=1 CRUCIBLE_QSA_READY_FILE="$ready" \
  "$binary" "$checkpoint" results/states/context253952.bin 1024 1 \
  "results/${tag}_trace.csv" --stages >"results/${tag}_trace.log" 2>&1
if [[ -n "$oracle" ]]; then wait "$oracle"; oracle=; fi
