#!/bin/bash
# One dump lease: mtp_dump of one LIST shard into DATA/<set>/stk-<shard>/.
#   dump.sh SET SHARD
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
mkdir -p "$DATA/$1/stk-$2"
build/bench/mtp_dump "$CK305" "$DATA/$1/$2.list" "$DATA/$1/stk-$2" --ctx 4096 2>&1 | grep -v "^prefill plan"
