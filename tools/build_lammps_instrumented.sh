#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
# Build the INSTRUMENTED pinned LAMMPS: the pristine pinned tree + third_party/lammps/patches/0001-*.patch.
# The pristine tree is never touched: the patch is applied to a private clone (work dir), whose diff hash is verified
# against third_party/lammps/patches/PATCHES.sha256 before building.
#   usage: tools/build_lammps_instrumented.sh <pristine-lammps-tree> <work-tree> <build-dir> <install-dir> [cmake args...]
set -euo pipefail
src="${1:?pristine lammps tree}"; work="${2:?work tree (will be created)}"; bld="${3:?build dir}"; ins="${4:?install dir}"; shift 4
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patch="$here/third_party/lammps/patches/0001-reaxmetal-diagnostics.patch"
want="$(grep -F "0001-reaxmetal-diagnostics.patch" "$here/third_party/lammps/patches/PATCHES.sha256" | cut -d' ' -f1)"
have="$(sha256sum "$patch" | cut -d' ' -f1)"
[[ "$want" == "$have" ]] || { echo "patch hash mismatch: $have != $want" >&2; exit 3; }
pin_commit="$(grep -E '^lammps_commit=' "$here/third_party/lammps/PIN.txt" | cut -d= -f2)"
[[ "$(git -C "$src" rev-parse HEAD)" == "$pin_commit" ]] || { echo "source tree is not the pinned commit" >&2; exit 3; }
[[ -z "$(git -C "$src" status --porcelain)" ]] || { echo "pristine tree is modified" >&2; exit 3; }
[[ ! -e "$work" ]] || { echo "work tree $work already exists" >&2; exit 3; }
git clone -q --shared "$src" "$work"
git -C "$work" checkout -q "$pin_commit"
git -C "$work" apply "$patch"
export REAXMETAL_PATCHED_TREE_SHA256="$want"
exec "$here/tools/build_lammps_reference.sh" "$work" "$bld" "$ins" "$@"
