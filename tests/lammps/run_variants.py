#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""VAR-1: option and force-field variants that no bundled fixture exercises, as in-LAMMPS A/B against stock `reaxff` (same machinery as INT-2):
  (a) van der Waals type 2 (inner wall without shielding): ffield.reax.cho edited so every element has gamma_w = 0.3 and rcore2/ecore2/acore2 > 0;
  (b) `enobonds no` on atoms with and without bonds (isolated atoms, H2, CO, water).
usage: run_variants.py --lmp BIN --plugin SO --ffield-dir DIR [--backend cpu64|metal]"""
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VDW2_CASES = ["cho_water_dimer", "cho_ethane_staggered", "cho_co2", "cho_methanol", "cho_benzene", "cho_water_box_8"]
ENOB_CASES = ["cho_atom_C", "cho_atom_H", "cho_atom_O", "cho_h2", "cho_co", "cho_water", "cho_o2"]


def make_vdw2_ffield(src, dst):
    L = Path(src).read_text().splitlines()
    k = [i for i, l in enumerate(L) if "Nr of atoms" in l][0]
    n = int(L[k].split()[0])
    first = k + 4                       # first element line
    for e in range(n):
        b = first + 4 * e
        t2 = L[b + 1].split(); t2[1] = "0.3000"                         # gamma_w <= 0.5: no shielding
        t4 = L[b + 3].split(); t4[5], t4[6], t4[7] = "1.5000", "0.1000", "8.0000"   # rcore2 ecore2 acore2: inner wall
        L[b + 1] = "      " + "  ".join(t2)
        L[b + 3] = "      " + "  ".join(t4)
    Path(dst).write_text("\n".join(L) + "\n")


def int2(a, fixtures, ffdir, only):
    cmd = [sys.executable, "-I", str(ROOT / "tests/lammps/run_int2.py"), "--lmp", a.lmp, "--plugin", a.plugin, "--ffield-dir", str(ffdir),
           "--backend", a.backend, "--fixtures", str(fixtures), "--only", only]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-3:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--backend", default="cpu64")
    a = ap.parse_args()
    fails = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # (a) vdW type 2
        ffd = td / "ff"; ffd.mkdir()
        make_vdw2_ffield(Path(a.ffield_dir) / "ffield.reax.cho", ffd / "ffield.reax.cho")
        for c in VDW2_CASES:
            ok, tail = int2(a, ROOT / "tests/fixtures", ffd, c)
            print(("ok   " if ok else "FAIL ") + f"vdW type 2: {c}")
            if not ok: fails.append(("vdw2", c, tail))
        # (b) enobonds no
        fx = td / "fx"; (fx / "cases").mkdir(parents=True)
        (fx / "reference").symlink_to(ROOT / "tests/fixtures/reference")
        for c in ENOB_CASES:
            case = json.loads((ROOT / "tests/fixtures/cases" / f"{c}.json").read_text())
            case.setdefault("pair", {})["enobonds"] = False
            (fx / "cases" / f"{c}.json").write_text(json.dumps(case))
            ok, tail = int2(a, fx, a.ffield_dir, c)
            print(("ok   " if ok else "FAIL ") + f"enobonds no: {c}")
            if not ok: fails.append(("enobonds", c, tail))
    for f in fails: print("FAIL", f)
    print(f"VAR-1 backend {a.backend}: {len(VDW2_CASES) + len(ENOB_CASES)} runs, {len(fails)} failures")
    print("RESULT: PASS" if not fails else "RESULT: FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
