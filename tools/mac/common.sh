#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
# Shared helpers for the Apple-machine check scripts (sourced, not executed). Works with macOS' stock bash 3.2 and zsh-launched bash.
# Everything is recorded in mac-reports/stepN-<timestamp>.txt: environment, build log, check output. Paste that file back.

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${REAXMETAL_MAC_BUILD:-$here/build-mac}"
REPORT_DIR="$here/mac-reports"
mkdir -p "$REPORT_DIR"
stamp="$(date +%Y%m%d-%H%M%S)"

mac_env_report() {
  echo "=== ENVIRONMENT ($(date))"
  echo "--- sw_vers:";           sw_vers 2>&1
  echo "--- uname -a:";          uname -a 2>&1
  echo "--- cpu:";               sysctl -n machdep.cpu.brand_string 2>&1
  echo "--- xcode-select -p:";   xcode-select -p 2>&1
  echo "--- xcrun --show-sdk-path:"; xcrun --show-sdk-path 2>&1
  echo "--- xcrun --find metal (EXPECTED to fail on a Command-Line-Tools-only machine):"; xcrun --find metal 2>&1
  echo "--- Metal.framework headers present:"; ls "$(xcrun --show-sdk-path 2>/dev/null)/System/Library/Frameworks/Metal.framework/Headers" 2>&1 | head -3
  echo "--- clang++:";           clang++ --version 2>&1 | head -2
  echo "--- cmake:";             cmake --version 2>&1 | head -1
  echo "--- GPU:";               system_profiler SPDisplaysDataType 2>&1 | grep -E "Chipset Model|Total Number of Cores|Metal Support|Metal Family" | head -5
  echo "--- repository:";        git -C "$here" rev-parse HEAD 2>&1; git -C "$here" status --short 2>&1 | head -5
  echo
}

# usage: run_step <n> <extra args for reaxmetal_metal_check...>
run_step() {
  local n="$1"; shift
  local report="$REPORT_DIR/step${n}-${stamp}.txt"
  {
    mac_env_report
    echo "=== CONFIGURE + BUILD ($BUILD)"
    if ! cmake -S "$here" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DREAXMETAL_ENABLE_METAL=ON -DREAXMETAL_BUILD_TESTS=OFF 2>&1; then
      echo "STEP $n RESULT: FAIL (cmake configure failed)"; return 1
    fi
    if ! cmake --build "$BUILD" -j "$(sysctl -n hw.ncpu 2>/dev/null || echo 4)" --target reaxmetal_metal_check 2>&1; then
      echo "STEP $n RESULT: FAIL (build failed - the first error above is the one to send)"; return 1
    fi
    echo
    echo "=== RUN: reaxmetal_metal_check --step $n $*"
    "$BUILD/reaxmetal_metal_check" --step "$n" "$@" 2>&1
  } 2>&1 | tee "$report"
  local rc=${PIPESTATUS[0]}
  echo
  echo ">>> Report saved to: $report"
  echo ">>> Please send me the contents of that file (or paste the terminal output)."
  return "$rc"
}
