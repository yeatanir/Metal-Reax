#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""MIN-1: LAMMPS-hosted `minimize` with `reaxff/metal` reaches the same stationary point as stock `reaxff`. Per perturbed fixture: minimise with the
stock style and with the plugin from the same start; compare the final energies (C3: <= 1e-3 kcal/mol/atom), the energy of the plugin's final geometry re-evaluated
with STOCK reaxff (same bound) and the largest atom displacement between the two minima (<= 0.1 A). The residual force is printed but not gated: ReaxFF has kinks, so neither
style reaches the requested ftol on every fixture (stock stops at fmax up to 0.5 kcal/mol/A on these).
usage: run_min.py --lmp BIN --plugin SO --ffield-dir DIR [--backend metal] [--gpu-qeq]"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "reaxref"))
import runner  # noqa: E402


def last_row(text, ncols):
    rows, on = [], False
    for line in text.splitlines():
        if line.startswith("   Step"): on = True; continue
        if line.startswith("Loop time"): on = False
        if on:
            try:
                v = [float(x) for x in line.split()]
                if len(v) == ncols: rows.append(v)
            except ValueError: pass
    return rows[-1] if rows else None


def read_atoms(path):
    L = Path(path).read_text().splitlines()
    box = np.array([float(L[i].split()[1]) - float(L[i].split()[0]) for i, l in enumerate(L) if l.endswith(("xlo xhi", "ylo yhi", "zlo zhi"))])
    k = [i for i, l in enumerate(L) if l.startswith("Atoms")][0] + 2
    rows = []
    while k < len(L) and L[k].strip():
        v = L[k].split(); rows.append((int(v[0]), [float(x) for x in v[3:6]])); k += 1
    return box, np.array([r[1] for r in sorted(rows)])


def max_displacement(a, b):
    box, xa = read_atoms(a); _, xb = read_atoms(b)
    d = xa - xb
    d -= box * np.round(d / box)
    return np.linalg.norm(d, axis=1).max()


def run(lmp, cwd, env, text):
    (cwd / "in.lmp").write_text(text)
    r = subprocess.run([lmp, "-in", "in.lmp", "-log", "none", "-nocite"], cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode: raise RuntimeError((r.stdout + r.stderr)[-500:])
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--backend", default="metal"); ap.add_argument("--gpu-qeq", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"), LAMMPS_POTENTIALS=a.ffield_dir)
    tail = "thermo_style custom step pe fmax fnorm\nthermo_modify format float %.12g\nthermo 20\nmin_style cg\nminimize 1e-14 1e-5 3000 30000\nwrite_data {out} nocoeff\n"
    fails, worst_e, worst_f = [], 0.0, 0.0
    cases = sorted((ROOT / "tests" / "fixtures" / "cases").glob("*_pert.json"))
    with tempfile.TemporaryDirectory() as td:
        for cf in cases:
            case = json.loads(cf.read_text())
            ffpath = Path(a.ffield_dir) / case["ffield"]["name"]
            d = Path(td) / cf.stem; d.mkdir()
            runner.write_data(case, runner.parse_ffield(ffpath), d / "data.lmp")
            case["charge"] = dict(case.get("charge", {"model": "qeq/reaxff", "swb": 10.0, "maxiter": 500}), tolerance=1e-10)
            head = runner.build_input(case, ffpath, d / "data.lmp", d / "x.dump").split("fix integ all nve")[0]
            n = len(case["atoms"])
            try:
                s_out = run(a.lmp, d, env, head + tail.format(out="stock.data"))
                ours = head.replace("pair_style reaxff NULL", f"plugin load {a.plugin}\npair_style reaxff/metal NULL backend {a.backend}", 1)
                if a.gpu_qeq: ours = ours.replace(" qeq/reaxff ", " qeq/reaxff/metal ")
                o_out = run(a.lmp, d, env, ours + tail.format(out="ours.data"))
                chk = head.replace("data.lmp", "ours.data") + "thermo_style custom step pe fmax fnorm\nthermo_modify format float %.12g\nrun 0\n"
                c_out = run(a.lmp, d, env, chk)
            except RuntimeError as e:
                fails.append(f"{cf.stem}: {e}"); continue
            s, o, c = last_row(s_out, 4), last_row(o_out, 4), last_row(c_out, 4)
            if not (s and o and c): fails.append(f"{cf.stem}: no thermo"); continue
            de, dc = abs(o[1] - s[1]) / n, abs(c[1] - s[1]) / n
            disp = max_displacement(d / 'stock.data', d / 'ours.data')
            worst_e, worst_f = max(worst_e, de, dc), max(worst_f, c[2])
            ok = de <= 1e-3 and dc <= 1e-3 and disp <= 0.1   # fmax is reported, not gated: ReaxFF has kinks (bond-order cutoffs), stock itself stops at fmax up to ~0.5 on these
            print(f"{'ok  ' if ok else 'FAIL'} {cf.stem:28s} E/atom diff {de:.2e} (stock-evaluated {dc:.2e})  max atom displacement {disp:.1e} A  fmax at ours' geometry {c[2]:.2e} (stock's own {s[2]:.1e}, plugin's own {o[2]:.1e})")
            if not ok: fails.append(cf.stem)
    print(f"MIN-1 backend {a.backend}: {len(cases)} fixtures, {len(fails)} failures; worst E/atom diff {worst_e:.2e}, worst fmax {worst_f:.2e}")
    print("RESULT:", "PASS" if not fails and cases else "FAIL")
    return 0 if not fails and cases else 1


if __name__ == "__main__":
    sys.exit(main())
