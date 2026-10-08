#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
# STEP 2 (NBR-1/NBR-2, FORCE-2): device neighbor rows over owned+ghost atoms equal the CPU-64 list (superset/bounded contract of
# include/reaxmetal/neighbor.hpp) on 7 shared geometries, byte-identical across launches, and grow-and-retry works. ~2-5 minutes.
# Run step 1 first; do not run this one if step 1 failed.   usage: tools/mac/step2_neighbor.sh [--quick]
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_step 2 "$@"
