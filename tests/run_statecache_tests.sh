#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CXX=${CXX:-clang++}
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
for source in conversation_cache batch_snapshot draft_prompt statecache_options; do
  "$CXX" -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
    "tests/${source}_test.cpp" -o "$build/$source"
  "$build/$source"
done
printf 'PASS: cache policy, pinned snapshots, pairing, and fixture options.\n'
