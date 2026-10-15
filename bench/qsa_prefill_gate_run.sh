#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
set -o noclobber
candidate=${1:?candidate binary}
baseline=${2:?independent baseline binary}
tag=${3:?unique evidence tag}
name=${4:?baseline name, for example origin/main-45c3f0e}
if [[ ! "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ || ${#tag} -gt 80 ]]; then
  echo 'invalid tag: use 1 to 80 letters, digits, periods, underscores, or hyphens' >&2
  exit 2
fi
for tier in 305 405; do
  case "$tier" in
    305) ckpt="$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5" ;;
    405) ckpt="$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6" ;;
  esac
  output="results/gates/$tier/$tag"
  mkdir -p "$(dirname "$output")"
  sha256sum "$candidate" "$baseline" tools/gates.py >"$output.inputs.sha256"
  python3 tools/gates.py "$ckpt" --engine "$candidate" --baseline-engine "$baseline" \
    --baseline-name "$name" \
    --ids "$WORK_FIB_IDS" \
    --prefill-rows 24 --baseline-prefill-rows 24 --tokens 160 --priority 2 \
    --output "$output"
done
