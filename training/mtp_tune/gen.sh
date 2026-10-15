#!/bin/bash
# One generation lease: spec_sweep (tree-draft 1dd67a2 build, it knows --stop-ids) with the
# production policy, printing output IDs.
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
CRUCIBLE_SWEEP_IDS=1 build/bench/spec_sweep_td "$CK305" "$1"
