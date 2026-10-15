#!/usr/bin/env bash
# One queued job: pfbench last-position logits at 1K and 4K for the named main
# baseline and the candidate, default and exact-split paths on 3.05, default
# on 4.05, with CRUCIBLE_PROFILE on the default 3.05 runs.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cand=${1:?candidate engine}
base=${2:?named main engine}
out=${3:?new output directory}
C3=$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
C4=$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6
P=$WORK_DIR/wt/prefill-scan/results/prompts
runs="n1k:$P/source-1024.ids:4096:0:0:0:0:0:0 n4k:$P/source-4096.ids:4096:0:0:0:0:0:0"
mkdir "$out"
sha256sum "$cand" "$base" "$P"/source-{1024,4096}.ids >"$out/inputs.sha256"
job="$out/job.sh"
echo 'set -u; export HSA_USERPTR_FOR_PAGED_MEM=0' >"$job"
for who in base cand; do
  bin=$base; [[ $who == cand ]] && bin=$cand
  for mode in d305 s305 d405; do
    mkdir "$out/$who-$mode"
    env=""; ck=$C3
    [[ $mode == d305 ]] && env="CRUCIBLE_PROFILE=1"
    [[ $mode == s305 ]] && env="CRUCIBLE_PF_EXACT_SPLIT=1"
    [[ $mode == d405 ]] && ck=$C4
    echo "env $env CRUCIBLE_PF_OUT='$out/$who-$mode' '$bin' pfbench '$ck' --prefill 4096 $runs >'$out/$who-$mode/log.txt' 2>&1; echo $who-$mode rc \$?" >>"$job"
  done
done
"$WORK_GPU_RUN" -p 2 -w 7200 -- timeout -k 10 300 bash "$job" >"$out/job.log" 2>&1
sha256sum --check "$out/inputs.sha256" >"$out/inputs-checked.log"
for mode in d305 s305 d405; do
  for t in n1k n4k; do
    cmp -s "$out/base-$mode/$t.logits.f32" "$out/cand-$mode/$t.logits.f32" \
      && echo "$mode $t EQUAL" || echo "$mode $t DIFFERS"
  done
done | tee "$out/compare.txt"
