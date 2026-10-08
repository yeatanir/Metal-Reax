#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
# STEP 1 (MET-1): does the Metal backend build against the Command-Line-Tools SDK, does the runtime shader compiler work
# without Xcode, and do trivial kernels + buffer round trips give exact results? Takes ~1-2 minutes.
#   usage (from the repository root):  tools/mac/step1_bringup.sh
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_step 1
