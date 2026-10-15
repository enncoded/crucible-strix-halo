#!/usr/bin/env bash
# Serial long-context driver: baseline and candidate pairs on both checkpoints.
set -euo pipefail
base=${1:?baseline engine with index capture}
cand=${2:?candidate engine}
btag=${3:?baseline tag}
ctag=${4:?candidate tag}
for tier in 305 405; do
  bash bench/qsa_prefill_long_run.sh "$tier" 16384 "$btag=$base" "$ctag=$cand"
  bash bench/qsa_prefill_long_run.sh "$tier" 65536 "$btag=$base" "$ctag=$cand"
  bash bench/qsa_prefill_long_run.sh "$tier" 131072 "$btag=$base"
  bash bench/qsa_prefill_long_run.sh "$tier" 131072 "$ctag=$cand"
done
echo long_complete
