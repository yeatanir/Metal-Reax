#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
# Reproduce every M1 measurement from scratch (or resume: each step is skipped if its output exists).
#   usage: tools/m1_reproduce.sh <work-root> [--no-build] [--no-mpi-kokkos] [--refreeze]
# Needs: git, cmake >= 3.20, g++, clang++, python3 + numpy; MPICH (mpicc/mpiexec) for the MPI probe; network access to github.com
# (git over HTTPS) for the pinned LAMMPS tree. Wall time on 4 cores: ~1.5 h cold (12 LAMMPS builds), minutes warm.
# Outputs under <work-root>: lammps/ (pristine pinned tree), ff/, matrix/ (builds), runs/ (suites), reports/, exp/, gate.json
# The frozen tolerance file (tolerances/) is NOT regenerated unless --refreeze (it is hashed and tested).
set -uo pipefail
root="${1:?work root}"; shift
build=1; mk=1; refreeze=0
for a in "$@"; do case "$a" in --no-build) build=0;; --no-mpi-kokkos) mk=0;; --refreeze) refreeze=1;; esac; done
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$here/tools/reaxref"
mkdir -p "$root"; root="$(cd "$root" && pwd)"
J="${JOBS:-4}"
step() { echo; echo "=== $* ==="; }
cd "$here"

step "1. pristine pinned LAMMPS"
[[ -d "$root/lammps/.git" ]] || tools/fetch_lammps.sh --full "$root/lammps" || exit 1

step "2. force fields (verified against tests/fixtures/ffield_manifest.tsv)"
mkdir -p "$root/ff"
while IFS=$'\t' read -r name sha rel; do
  [[ "$name" == \#* || -z "$name" ]] && continue
  cp "$root/lammps/$rel" "$root/ff/$name"
  got="$(sha256sum "$root/ff/$name" | cut -d' ' -f1)"
  [[ "$got" == "$sha" ]] || { echo "force field hash mismatch: $name" >&2; exit 1; }
done < tests/fixtures/ffield_manifest.tsv
echo "force fields OK"

if [[ $build -eq 1 ]]; then
  step "3. build matrix"
  variants=(gcc clang gcc-native clang-native gcc-O0 gcc-O2-nofma)
  [[ $mk -eq 1 ]] && variants+=(mpi kokkos-serial)
  JOBS=$J tools/build_reference_matrix.sh "$root/lammps" "$root/matrix" "${variants[@]}" || exit 1
fi

M="$root/matrix"; RUNS="$root/runs"; REP="$root/reports"; EXP="$root/exp"; mkdir -p "$RUNS" "$REP" "$EXP"
suite() { # label binary [diag|nodiag] [extra run_suite args...]
  local label="$1" bin="$2" mode="$3"; shift 3
  [[ -f "$RUNS/$label/summary.json" ]] && { echo "[skip] $label"; return; }
  local nd=(); [[ "$mode" == nodiag ]] && nd=(--no-diag)
  python3 "$R/run_suite.py" --lmp "$bin" --label "$label" --cases tests/fixtures/cases --ffield-dir "$root/ff" --out "$RUNS/$label" --jobs 2 "${nd[@]}" "$@" > "$RUNS/$label.log" 2>&1
  echo "$label: $(tail -1 "$RUNS/$label.log")"
}

step "4. suites on every build"
for v in gcc clang gcc-native clang-native; do
  suite "stock-$v" "$M/stock-$v/install/bin/lmp" nodiag
  suite "inst-$v" "$M/inst-$v/install/bin/lmp" diag
  suite "inst-$v-nodiag" "$M/inst-$v/install/bin/lmp" nodiag
done
suite stock-gcc-O0 "$M/stock-gcc-O0/install/bin/lmp" nodiag
suite stock-gcc-O2-nofma "$M/stock-gcc-O2-nofma/install/bin/lmp" nodiag
if [[ $mk -eq 1 && -x "$M/stock-mpi/install/bin/lmp" ]]; then
  for np in 1 2 4; do suite "stock-mpi-np$np" "$M/stock-mpi/install/bin/lmp" nodiag --launcher "mpiexec -np $np"; done
fi
if [[ $mk -eq 1 && -x "$M/stock-kokkos-serial/install/bin/lmp" ]]; then
  suite stock-kokkos-serial "$M/stock-kokkos-serial/install/bin/lmp" nodiag --lmp-args "-k on t 1 -sf kk"
fi

step "5. reports: instrumentation equivalence, conditioning, coverage, noise floor"
for v in gcc clang; do
  python3 "$R/compare_builds.py" "$RUNS/stock-$v" "$RUNS/inst-$v" --bitwise --json "$REP/equiv_stock-${v}_inst.json"
  python3 "$R/compare_builds.py" "$RUNS/stock-$v" "$RUNS/inst-$v-nodiag" --bitwise --json "$REP/equiv_stock-${v}_inst-nodiag.json"
done
for v in gcc-native clang-native; do
  python3 "$R/compare_builds.py" "$RUNS/stock-$v" "$RUNS/inst-$v" --json "$REP/equiv_stock-${v}_inst.json"
done
python3 "$R/conditioning.py" "$RUNS/inst-gcc" tests/fixtures/cases --json "$REP/conditioning.json" | tail -3
python3 "$R/coverage.py" "$RUNS/inst-gcc" --require | tail -4
nb=(); for d in "$RUNS"/stock-*/; do b="$(basename "$d")"; [[ -f "$RUNS/$b/summary.json" ]] && nb+=("$b"); done
python3 "$R/noise_floor.py" "$RUNS" "$REP/conditioning.json" --out "$REP/noise_floor.json" "${nb[@]}"
python3 "$R/check_nonbonded.py" "$RUNS/inst-gcc" tests/fixtures/cases --json "$REP/nonbonded_vs_independent.json" | tail -1

step "6. experiments"
LMP="$M/inst-gcc/install/bin/lmp"
[[ -f "$EXP/periodic.json" ]] || python3 "$R/exp_periodic.py" --lmp "$LMP" --ffield-dir "$root/ff" --workdir "$EXP/periodic" --out "$EXP/periodic.json" --label inst-gcc > "$EXP/periodic.log" 2>&1
[[ -f "$EXP/ghost.json" ]] || python3 "$R/exp_ghost.py" --lmp "$LMP" --ffield-dir "$root/ff" --workdir "$EXP/ghost" --out "$EXP/ghost.json" --label inst-gcc > "$EXP/ghost.log" 2>&1
[[ -f "$EXP/quirks.json" ]] || python3 "$R/exp_ffield_quirks.py" --lmp "$LMP" --ffield-dir "$root/ff" --workdir "$EXP/quirks" --out "$EXP/quirks.json" > "$EXP/quirks.log" 2>&1
if [[ ! -f "$EXP/q32_hbond_identity.json" ]]; then
  # experimental builds: patch 0001 + one-line experiment patch (throwaway, never used for fixtures)
  for e in 1 2; do
    pf=$(ls third_party/lammps/patches/experiments/EXP-000${e}-*.patch); w="$EXP/exp$e-work"
    if [[ ! -x "$EXP/exp$e-install/bin/lmp" ]]; then
      rm -rf "$w"; git clone -q --shared "$root/lammps" "$w" && git -C "$w" checkout -q "$(grep '^lammps_commit=' third_party/lammps/PIN.txt | cut -d= -f2)" &&
      git -C "$w" apply third_party/lammps/patches/0001-reaxmetal-diagnostics.patch && git -C "$w" add -A &&
      git -C "$w" -c user.email=x@y -c user.name=x commit -q -m base0001 && git -C "$w" apply "$pf" &&
      cmake -S "$w/cmake" -B "$EXP/exp$e-b" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$EXP/exp$e-install" -DBUILD_MPI=off -DBUILD_OMP=off \
        -DBUILD_SHARED_LIBS=on -DLAMMPS_EXCEPTIONS=on -DPKG_REAXFF=on -DPKG_QEQ=on -DPKG_PLUGIN=on > "$EXP/exp$e-cfg.log" 2>&1 &&
      cmake --build "$EXP/exp$e-b" -j "$J" > "$EXP/exp$e-build.log" 2>&1 && cmake --install "$EXP/exp$e-b" > "$EXP/exp$e-inst.log" 2>&1
    fi
  done
  python3 "$R/exp_q32_q34.py" --stock "$LMP" --exp1 "$EXP/exp1-install/bin/lmp" --exp2 "$EXP/exp2-install/bin/lmp" --ffield-dir "$root/ff" \
    --cases tests/fixtures/cases --workdir "$EXP/q3234" --out32 "$EXP/q32_hbond_identity.json" --out34 "$EXP/q34_ovun.json"
fi
[[ -f "$REP/fd_all.json" ]] || python3 "$R/fd_check.py" --lmp "$M/stock-gcc/install/bin/lmp" --ffield-dir "$root/ff" --cases tests/fixtures/cases --workdir "$EXP/fd" --out "$REP/fd_all.json" --jobs 3 > "$REP/fd_all.log" 2>&1

step "7. tolerances"
if [[ $refreeze -eq 1 ]]; then
  python3 "$R/make_tolerances.py" --noise "$REP/noise_floor.json" --conditioning "$REP/conditioning.json" --fd "$REP/fd_all.json" --runs "$RUNS/inst-gcc" \
    --cases tests/fixtures/cases --out tolerances --patch-sha256 "$(sha256sum third_party/lammps/patches/0001-reaxmetal-diagnostics.patch | cut -d' ' -f1)"
else
  echo "(frozen file kept; use --refreeze to regenerate)"
fi

step "8. gate"
python3 "$R/m1_gate.py" --root "$root" --out "$root/gate.json" --skip-git
