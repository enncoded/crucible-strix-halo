#!/bin/bash
# One A/B lease: spec_sweep on CFG with the stock head, then with SIDECAR (if given).
#   measure.sh CKPT CFG OUTPREFIX [SIDECAR]
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
unset CRUCIBLE_MTP_SIDECAR
build/bench/spec_sweep "$1" "$2" > "$3.stock.log" 2>&1
if [ -n "${4:-}" ]; then
  CRUCIBLE_MTP_SIDECAR="$4" build/bench/spec_sweep "$1" "$2" > "$3.tuned.log" 2>&1
fi
