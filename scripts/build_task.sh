#!/usr/bin/env bash
# Copyright (c) 2026 Niobium Microsystems, Inc.
# SPDX-License-Identifier: Apache-2.0
# build_task.sh — the one build. The niobium-client submodule builds its own
# OpenFHE and libnbfhetch, and every stage is built against those. Run from
# anywhere; binaries land in build/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== [1/3] sync niobium-client (niobium-fhetch and the OpenFHE it carries) ==="
git submodule update --init niobium-client
# niobium-client's only submodule is niobium-fhetch (OpenFHE nested in it);
# cpp-httplib is vendored in its tree.
make -C niobium-client sync

echo "=== [2/3] build niobium-client: OpenFHE, libnbfhetch, simulator, transport ==="
make -C niobium-client release

OPENFHE_PREFIX="$ROOT/niobium-client/vendor/lib/openfhe"
[[ -d "$OPENFHE_PREFIX" ]] || { echo "error: no OpenFHE at $OPENFHE_PREFIX after 'make release'" >&2; exit 1; }

echo "=== [3/3] build the diabetes stages against the client's OpenFHE ==="
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNB_CLIENT_DIR="$ROOT/niobium-client" \
  -DCMAKE_PREFIX_PATH="$OPENFHE_PREFIX"
cmake --build build -j

echo "=== done: binaries in $ROOT/build ==="
ls -la build/keygen build/encrypt build/decrypt build/diabetes_compute_sdk
