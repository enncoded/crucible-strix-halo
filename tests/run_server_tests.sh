#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
compiler=${CXX:-clang++}
python=${PYTHON:-python3}
if [[ $# -gt 0 ]]; then
  export CRUCIBLE_TEST_CKPT="$1"
fi
"$python" -m pytest -q server/tests
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
for test in serve_protocol_test serve_policy_test generation_test sampling_test sampling_generation_test sampling_filter_reference_test \
            nonfinite_inject_test; do
  "$compiler" -std=c++20 -O1 -g -fsanitize=address,undefined "tests/$test.cpp" -o "$build/$test"
  "$build/$test"
done
