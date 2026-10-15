#!/usr/bin/env bash
# Serial profile driver: one queued job at a time, baseline first in each job.
set -euo pipefail
base=${1:?baseline engine}
cand=${2:?candidate engine}
tag=${3:?unique tag}
bash bench/qsa_prefill_profile_run.sh 305 16384 "$tag" "$base" "$cand"
bash bench/qsa_prefill_profile_run.sh 305 65536 "$tag" "$base" "$cand"
bash bench/qsa_prefill_profile_run.sh 305 131072 "$tag-base" "$base"
bash bench/qsa_prefill_profile_run.sh 305 131072 "$tag-cand" "$cand"
echo profile_complete
