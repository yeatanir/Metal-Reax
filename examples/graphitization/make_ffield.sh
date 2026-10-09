#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
# Extracts the carbon ReaxFF "Reactive MD-force field: 2013 C" of S. G. Srinivasan, A. C. T. van Duin, P. Ganesh, J. Phys. Chem. A 119, 571 (2015) from its
# Supporting Information (jp510274e_si_001.pdf, which you must have obtained from the journal) into a LAMMPS ffield file. The parameters are NOT redistributed
# with this repository.   usage: make_ffield.sh <jp510274e_si_001.pdf> <ffield.reax.C2013>
set -euo pipefail
pdf="${1:?SI pdf}"; out="${2:?output ffield}"
command -v pdftotext >/dev/null || { echo "pdftotext (poppler) is required" >&2; exit 1; }
pdftotext -layout "$pdf" "$out"
grep -q "Reactive MD-force field: 2013 C" "$out" || { echo "unexpected content: this is not the 2013 C force field" >&2; exit 2; }
echo "wrote $out ($(wc -l < "$out") lines)"
