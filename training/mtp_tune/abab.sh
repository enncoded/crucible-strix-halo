#!/bin/bash
# ABAB timing lease: PAIRS x (stock, tuned) spec_sweep runs of CFG, with the GPU clock sampled every
# 0.2 s (hwmon freq1_input) during each run. Logs: OUT.<i>.<arm>.log and OUT.<i>.<arm>.sclk.
#   abab.sh CKPT CFG OUT SIDECAR PAIRS
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
ck=$1 cfg=$2 out=$3 side=$4 pairs=$5
freq=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/freq1_input | head -1)
echo "platform_profile $(cat /sys/firmware/acpi/platform_profile)" > "$out.env"
for i in $(seq 1 "$pairs"); do
  for arm in stock tuned; do
    ( while :; do cat "$freq"; sleep 0.2; done ) > "$out.$i.$arm.sclk" &
    sampler=$!
    if [ $arm = tuned ]; then
      CRUCIBLE_MTP_SIDECAR="$side" build/bench/spec_sweep "$ck" "$cfg" > "$out.$i.$arm.log" 2>&1
    else
      env -u CRUCIBLE_MTP_SIDECAR build/bench/spec_sweep "$ck" "$cfg" > "$out.$i.$arm.log" 2>&1
    fi
    kill "$sampler"
  done
done
