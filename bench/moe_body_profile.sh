#!/usr/bin/env bash
# The caller holds gpu-run. Disable engine warmup because rocprofv3 stalls there.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
[[ $# -eq 4 ]] || { echo "usage: $0 ENGINE CHECKPOINT OUTPUT GROUP" >&2; exit 2; }
engine=$1 checkpoint=$2 out=$3
shift 3
mkdir -p "$out"
sha256sum "$engine" > "$out/binary.sha256"
for group in "$@"; do
  case "$group" in
    occupancy) counters=(MeanOccupancyPerCU MeanOccupancyPerActiveCU OccupancyPercent GRBM_COUNT GRBM_GUI_ACTIVE CU_NUM SIMD_NUM) ;;
    lds) counters=(LDSBankConflict SQ_INSTS_LDS GRBM_GUI_ACTIVE) ;;
    issue) counters=(SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_TEX_LOAD SQ_INSTS_TEX_STORE GRBM_GUI_ACTIVE) ;;
    reads) counters=(FETCH_SIZE GRBM_GUI_ACTIVE) ;;
    writes) counters=(WRITE_SIZE L2CacheHit MemUnitBusy GRBM_GUI_ACTIVE) ;;
    *) echo "unknown counter group: $group" >&2; exit 2 ;;
  esac
  mkdir "$out/$group"
  CRUCIBLE_WARMUP=0 CRUCIBLE_PROFILE=1 /opt/rocm/bin/rocprofv3 \
    --disable-signal-handlers true --process-sync false \
    -f csv --pmc "${counters[@]}" \
    --kernel-include-regex 'PrefillMoe(Fx|Gemm)Kernel' \
    --kernel-iteration-range 49-96 -d "$out/$group" -o "$group" \
    -- "$engine" pfbench "$checkpoint" --prefill 4096 \
    "base:$WORK_DIR/wt/prefill-scan/results/prompts/source-4096.ids:4096:0:0:0:0:0:0" \
    > "$out/$group/run.log" 2>&1
  test -s "$out/$group/${group}_counter_collection.csv"
done
