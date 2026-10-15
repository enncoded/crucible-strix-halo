#!/usr/bin/env bash
# mem_floor.sh FLOOR_GB LOG: append MemAvailable to LOG and exit 76 (the cause on stderr) when it is below FLOOR_GB.
# FLOOR_GB is decimal: FLOOR_GB x 10^6 of the kB unit of /proc/meminfo (kB there is KiB). Exit 76 means that the
# floor was not met and no engine started; the caller can run the job again in a new lease.
# The engine load takes about 92.6 GB, and the short profiles need about 34 GB at engine ready. The recommended floor
# is 126 (128 is too close to an idle test machine: 3 of 8 engine-free samples were below 128 x 10^6 kB).
# Right after an earlier lease, its engine memory can still be in use (109.9 GB, then 128.4 GB a few minutes later).
# Below the floor the script reads MemAvailable again every MEM_FLOOR_POLL_S (10) seconds for up to
# MEM_FLOOR_WAIT_S (600; 0 = one check) before it gives exit 76.
# mem_floor.sh FLOOR_GB LOG -- COMMAND...: the same check, then exec COMMAND (exit 76 without it). GPU leases that
# start an engine directly use it (tools/long_capture.py).
# MEM_FLOOR_MEMINFO replaces /proc/meminfo in the tests.
set -u
floor=${1:-} log=${2:?usage: mem_floor.sh FLOOR_GB LOG [-- COMMAND...]}
shift 2
if (( $# )); then
  [[ $1 == -- && $# -gt 1 ]] || { echo "mem_floor.sh: after LOG give -- COMMAND..." >&2; exit 2; }
  shift
fi
wait_s=${MEM_FLOOR_WAIT_S:-600} poll_s=${MEM_FLOOR_POLL_S:-10}
[[ $floor =~ ^[1-9][0-9]{0,5}$ ]] || { echo "mem_floor.sh: SOAK_MIN_MEM_GB must be a whole number of GB from 1 to 999999 (found '$floor')" >&2; exit 2; }
[[ $wait_s =~ ^(0|[1-9][0-9]{0,4})$ ]] && (( wait_s <= 86400 )) ||
  { echo "mem_floor.sh: MEM_FLOOR_WAIT_S must be a whole number of seconds from 0 to 86400 (found '$wait_s')" >&2; exit 2; }
[[ $poll_s =~ ^[0-9]+(\.[0-9]+)?$ && ! $poll_s =~ ^0+(\.0+)?$ ]] ||
  { echo "mem_floor.sh: MEM_FLOOR_POLL_S must be a positive number of seconds (found '$poll_s')" >&2; exit 2; }
available() { awk '/^MemAvailable:/ {print $2}' "${MEM_FLOOR_MEMINFO:-/proc/meminfo}"; }
# Microseconds: $SECONDS counts whole seconds from the shell start, so a 1 s wait could end at once.
now_us() { echo $(( ${EPOCHREALTIME//[.,]/} )); }
start=$(now_us)
kb=$(available)
if [[ $kb =~ ^[0-9]+$ ]] && (( kb < floor * 1000000 && wait_s > 0 )); then
  echo "$(date +%T) MemAvailable $kb kB floor $floor GB below: wait up to $wait_s s" >> "$log"
  while (( $(now_us) - start < wait_s * 1000000 )); do
    sleep "$poll_s"
    kb=$(available)
    [[ $kb =~ ^[0-9]+$ ]] && (( kb >= floor * 1000000 )) && break
  done
fi
waited=$(( ($(now_us) - start) / 1000000 ))
after=""; (( wait_s > 0 )) && after=" after $waited s"
if [[ ! $kb =~ ^[0-9]+$ ]]; then
  line="$(date +%T) MemAvailable unreadable floor $floor GB below: no start$after"
elif (( kb < floor * 1000000 )); then
  line="$(date +%T) MemAvailable $kb kB floor $floor GB below: no start$after"
else
  echo "$(date +%T) MemAvailable $kb kB floor $floor GB start$after" >> "$log"
  (( $# )) && exec "$@"
  exit 0
fi
echo "$line" >> "$log"; echo "mem_floor.sh: $line" >&2; exit 76
