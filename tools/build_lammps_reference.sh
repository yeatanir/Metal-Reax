#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# Build the STOCK (unmodified) pinned LAMMPS used as the validation oracle and as the host for the
# reaxff/metal plugin. Reusable by M1. The source tree must come from `tools/fetch_lammps.sh --full`.
#   usage: tools/build_lammps_reference.sh <lammps-source-tree> <build-dir> <install-dir> [extra cmake args...]
# Serial (no MPI), shared library (required so plugins resolve LAMMPS symbols from the loaded library),
# packages: REAXFF (oracle), QEQ (qeq/shielded), PLUGIN (plugin command). No other package is enabled.
set -euo pipefail
src="${1:?lammps source tree}"; bld="${2:?build dir}"; ins="${3:?install dir}"; shift 3
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pin_commit="$(grep -E '^lammps_commit=' "$here/third_party/lammps/PIN.txt" | cut -d= -f2)"
[[ "$(git -C "$src" rev-parse HEAD)" == "$pin_commit" ]] || { echo "source tree is not the pinned commit" >&2; exit 3; }
[[ -z "$(git -C "$src" status --porcelain)" ]] || { echo "source tree is modified; the stock reference build must be pristine" >&2; exit 3; }
cmake -S "$src/cmake" -B "$bld" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ins" \
  -DBUILD_MPI=off -DBUILD_OMP=off -DBUILD_SHARED_LIBS=on -DLAMMPS_EXCEPTIONS=on \
  -DPKG_REAXFF=on -DPKG_QEQ=on -DPKG_PLUGIN=on "$@"
cmake --build "$bld" -j "${JOBS:-4}"
cmake --install "$bld"
echo "stock LAMMPS built from $pin_commit into $ins"
