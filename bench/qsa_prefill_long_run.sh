#!/usr/bin/env bash
# Runs one or more engines back to back in one queued job, so a timing pair
# sees the same box state. Each run records full final logits, generated IDs,
# all 12 index-only QSA captures, and the CRUCIBLE_PROFILE stage times.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
tier=${1:?305 or 405}
ctx=${2:?context}
shift 2
(( $# >= 1 )) || { echo 'need TAG=ENGINE arguments' >&2; exit 2; }
case "$tier" in
  305) ckpt="$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5" ;;
  405) ckpt="$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6" ;;
  *) echo 'invalid tier: use 305 or 405' >&2; exit 2 ;;
esac
priority=2 deadline=300 capacity=$((ctx+4))
case "$ctx" in
  16384|65536|131072) ;;
  # The first output does not consume a cache slot. Keep the supported 256K bound.
  262144) priority=3 deadline=600 capacity=$ctx ;;
  *) echo 'invalid context: use 16384, 65536, 131072, or 262144' >&2; exit 2 ;;
esac
ids="results/prompts/long_$ctx.txt"
job=$(mktemp "results/long-job.XXXXXX.sh")
outs=()
echo 'set -u' >"$job"
for pair in "$@"; do
  tag=${pair%%=*} binary=${pair#*=}
  if [[ ! "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ || ${#tag} -gt 80 || ! -x "$binary" ]]; then
    echo "invalid TAG=ENGINE: $pair" >&2; exit 2
  fi
  out="results/engine/$tier/$tag-$ctx"
  mkdir -p "$(dirname "$out")"
  mkdir "$out" "$out/capture"
  sha256sum "$binary" "$ids" >"$out/input.sha256"
  printf '%s\n' "env HSA_USERPTR_FOR_PAGED_MEM=0 CRUCIBLE_PROFILE=1 CRUCIBLE_QSA_CAPTURE_DIR='$out/capture' \
CRUCIBLE_QSA_CAPTURE_AT=$ctx CRUCIBLE_QSA_CAPTURE_INDEX_ONLY=1 '$binary' gen '$ckpt' --ids-file '$ids' \
--n 1 --prefill 4096 --spec 0 --ctx $capacity --dump-logits '$out/logits.f32' >'$out/run.log' 2>&1; echo '$tag' rc \$?" >>"$job"
  outs+=("$out")
done
"$WORK_GPU_RUN" -p "$priority" -w 7200 -- timeout -k 10 "$deadline" bash "$job"
for out in "${outs[@]}"; do
  sha256sum --check "$out/input.sha256" >"$out/inputs-checked.log"
  sha256sum "$out/logits.f32" "$out"/capture/*.i32 >"$out/output.sha256"
done
