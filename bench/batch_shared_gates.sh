#!/usr/bin/env bash
# The shared tool queues each capture. Do not wrap this script with gpu-run.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cd "$(dirname "$0")/.."
if (( $# < 4 || $# > 5 )); then
  echo 'usage: batch_shared_gates.sh ENGINE BASELINE BASELINE_NAME OUT [MODEL_ROOT]' >&2
  exit 2
fi
engine=$1
baseline=$2
name=$3
output=$4
models=${5:-$HOME/models/exl3}
mkdir "$output"
sha256sum "$engine" "$baseline" > "$output/binary-sha256.txt"
status=0
for item in '305:3.05bpw_h5_ng5' '405:4.05bpw_h6_ng6'; do
  tag=${item%%:*}
  pack=${item#*:}
  if ! HSA_USERPTR_FOR_PAGED_MEM=0 python3 tools/gates.py \
      "$models/Qwen3.8-Flash-Next-exl3-$pack" --engine "$engine" \
      --baseline-engine "$baseline" --baseline-name "$name" \
      --ids "$WORK_FIB_IDS" \
      --output "$output/$tag" --priority 3 > "$output/$tag.log" 2>&1; then
    status=1
  fi
done
exit "$status"
