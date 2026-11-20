#!/usr/bin/env bash
# Build the max-tree Python module mta.so from max-tree-trie/ and copy it next
# to the scripts. Incremental: fast when nothing changed.
#
# Usage: ./build_mta.sh
#
# Needs CMake (3.8 or newer), a C++14 compiler, and the Python development
# headers and NumPy for the `python` on your PATH (the module is built for it).
set -euo pipefail

here=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)

cmake -S "$here/max-tree-trie" -B "$here/max-tree-trie/build" \
    -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v python)"
cmake --build "$here/max-tree-trie/build" --target mta -j
cp "$here/max-tree-trie/build/mta.so" "$here/mta.so"

echo "Built $here/mta.so"
