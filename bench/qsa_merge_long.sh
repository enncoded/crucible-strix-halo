#!/usr/bin/env bash
# Serial driver for the merge evidence: the pfbench job, then the long pairs.
set -euo pipefail
cand=${1:?candidate engine}
base=${2:?named main engine}
idx=${3:?main plus index capture engine}
ctag=${4:?candidate tag}
btag=${5:?baseline tag}
bash bench/qsa_merge_pf.sh "$cand" "$base" "results/pf-$ctag"
for tier in 305 405; do
  bash bench/qsa_prefill_long_run.sh "$tier" 16384 "$btag=$idx" "$ctag=$cand"
  bash bench/qsa_prefill_long_run.sh "$tier" 65536 "$btag=$idx" "$ctag=$cand"
  bash bench/qsa_prefill_long_run.sh "$tier" 131072 "$btag=$idx"
  bash bench/qsa_prefill_long_run.sh "$tier" 131072 "$ctag=$cand"
done
echo merge_long_complete
