#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""HYB-1: `pair_style hybrid/overlay` with `reaxff/metal` as a sub-style, on examples/reaxff/ci-reaxFF (charge-implicit ReaxFF for C/H plus a tabulated ZBL
correction per type pair, Berendsen thermostat, 300 steps): stock `reaxff` against the plugin (cpu64 and metal). The example's stock run has a ghost shell of
9.5 A, narrower than the 2*bond_cut = 10 A that the reference never checks; the plugin run therefore uses `shellcheck no` (the check itself is exercised
elsewhere). Compared: PE at step 0 (cpu64 2e-9 relative, metal 1e-3 kcal/mol/atom) and PE / T every 100 steps (1e-6 relative cpu64; 1e-3 relative metal).
usage: run_hybrid.py --lmp BIN --plugin SO --example DIR [--backend cpu64|metal]"""
import argparse, os, shutil, subprocess, sys, tempfile
from pathlib import Path


def run(lmp, wd, text, env):
    (wd / "in.lmp").write_text(text)
    r = subprocess.run([lmp, "-in", "in.lmp", "-log", "none", "-nocite"], cwd=wd, env=env, capture_output=True, text=True, timeout=900)
    if r.returncode: raise RuntimeError((r.stdout + r.stderr)[-800:])
    rows, on = [], False
    for line in r.stdout.splitlines():
        if line.startswith("   Step"): on = True; continue
        if line.startswith("Loop time"): on = False
        if on:
            try: rows.append([float(x) for x in line.split()])
            except ValueError: pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--example", required=True); ap.add_argument("--backend", default="cpu64")
    a = ap.parse_args()
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"))
    fails = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for f in Path(a.example).iterdir():
            if f.is_file() and not f.name.startswith(("log.", "in.")): shutil.copy(f, td / f.name)
        env["LAMMPS_POTENTIALS"] = str(td)
        src = (Path(a.example) / "in.ci-reax.CH").read_text().replace("run             3000", "thermo 100\nthermo_style custom step temp pe ke press\nthermo_modify format float %.12g\nrun 300")
        ours = src.replace("pair_style      hybrid/overlay reaxff control checkqeq no", f"plugin load {a.plugin}\npair_style hybrid/overlay reaxff/metal control checkqeq no shellcheck no backend {a.backend}")
        ours = ours.replace("pair_coeff      * * reaxff ffield", "pair_coeff * * reaxff/metal ffield")
        s, o = run(a.lmp, td, src, env), run(a.lmp, td, ours, env)
        nat = 315
        for k, (rs, ro) in enumerate(zip(s, o)):
            dpe, dT = abs(ro[2] - rs[2]), abs(ro[1] - rs[1]) / rs[1]
            print(f"step {int(rs[0]):4d}  PE stock {rs[2]:.6f} ours {ro[2]:.6f}  dPE/atom {dpe / nat:.2e}  dT/T {dT:.2e}")
            if a.backend == "metal":
                if (k == 0 and dpe / nat > 1e-3) or (k > 0 and (dpe / abs(rs[2]) > 1e-3 or dT > 1e-3)): fails.append((int(rs[0]), "metal"))
            else:
                if (k == 0 and dpe > 2e-9 * abs(rs[2])) or (k > 0 and (dpe > 1e-6 * abs(rs[2]) or dT > 1e-6)): fails.append((int(rs[0]), "cpu64"))
        if len(s) != len(o) or not s: fails.append(("rows", len(s), len(o)))
    for f in fails: print("FAIL", f)
    print(f"HYB-1 backend {a.backend}: {len(fails)} failures")
    print("RESULT: PASS" if not fails else "RESULT: FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
