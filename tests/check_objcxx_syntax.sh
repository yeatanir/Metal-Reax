#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
# Syntax/type check of src/metal/metal_backend.mm on a machine WITHOUT the Apple SDK, against the hand-written stubs in
# tests/objc_stub (fidelity to the real headers is NOT guaranteed; a pass is necessary, not sufficient).
# usage: tests/check_objcxx_syntax.sh [clang++]   (exit 0 pass, 77 skipped: no clang++ / no Objective-C support)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cxx="${1:-clang++}"
command -v "$cxx" >/dev/null 2>&1 || { echo "skip: $cxx not found"; exit 77; }
gen="$(mktemp -d)"; trap 'rm -rf "$gen"' EXIT
echo '' > "$gen/empty.mm"
"$cxx" -x objective-c++ -std=c++20 -fobjc-runtime=gnustep-2.0 -fobjc-arc -fsyntax-only "$gen/empty.mm" 2>/dev/null || { echo "skip: $cxx cannot parse Objective-C++"; exit 77; }
"$cxx" -x objective-c++ -std=c++20 -fobjc-runtime=gnustep-2.0 -fobjc-arc -fsyntax-only -Wall -Wextra -Wno-unused-parameter \
  -DREAXMETAL_OBJCXX_STUB_CHECK=1 -I"$here/tests/objc_stub" -I"$here/include" -I"$here/src/metal/shaders" "$here/src/metal/metal_backend.mm"
echo "metal_backend.mm: syntax OK against the stub headers (NOT the real Metal SDK)"
