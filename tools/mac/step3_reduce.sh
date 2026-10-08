#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
# STEP 3 (FORCE-2, MET-4): fixed-order float reductions on the GPU are bitwise equal to the CPU twin, repeat exactly over 50 launches,
# and work on real neighbor-row data. ~1-3 minutes.   usage: tools/mac/step3_reduce.sh [--quick]
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_step 3 "$@"
