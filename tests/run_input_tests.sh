#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
compiler=${CXX:-clang++}
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
export CXX="$compiler"
python3 -m unittest discover -s tests -p test_checkpoint.py
python3 -m unittest discover -s tests -p test_prefill_h_default.py
for test in engine_bounds_test retained_prefix_test dump_io_test checkpoint_raii_test; do
  "$compiler" -std=c++20 -O1 -g -fsanitize=address,undefined "tests/$test.cpp" -o "$build/$test"
  "$build/$test"
done
# A supplied local checkpoint adds a full CPU geometry check. This does not use HIP.
if [[ $# -gt 0 ]]; then
  "$compiler" -std=c++20 -O1 -g -fsanitize=address,undefined tests/checkpoint_validate.cpp -o "$build/validate"
  "$build/validate" "$1"
fi
