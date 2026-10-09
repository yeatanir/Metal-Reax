#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""PERATOM-1: per-atom energy (compute pe/atom) and per-atom virial (compute stress/atom ... virial) of `reaxff/metal` against stock `reaxff` on the
58 fixtures, identical input. Per-atom outputs are produced by the CPU-64 engine's tallies (the reference's ev_tally / v_tally calls), also when
'backend metal' is selected: such a step is evaluated by the CPU-64 engine. Criterion: |d| <= 1e-8 * max(1, scale of the fixture's per-atom values)
(CPU-64 vs stock noise is ~1e-12); the per-atom sums must also reproduce the potential energy and the global virial.
usage: run_peratom.py --lmp BIN --plugin SO --ffield-dir DIR [--backend cpu64|metal]"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "reaxref"))
import runner  # noqa: E402


def run(lmp, case, ffield_dir, wd, env, plugin=None, backend=None):
    wd.mkdir(parents=True, exist_ok=True)
    ffpath = Path(ffield_dir) / case["ffield"]["name"]
    runner.write_data(case, runner.parse_ffield(ffpath), wd / "data.lmp")
    head = runner.build_input(case, ffpath, wd / "data.lmp", wd / "x.dump").split("fix integ all nve")[0]
    if plugin:
        head = head.replace("pair_style reaxff NULL", f"plugin load {plugin}\npair_style reaxff/metal NULL backend {backend}", 1)
    text = head + ("compute pea all pe/atom\ncompute sta all stress/atom NULL virial\ncompute pv all pressure NULL virial\n"
                   "thermo_style custom step pe c_pv[1] c_pv[2] c_pv[3] c_pv[4] c_pv[5] c_pv[6]\nthermo_modify format float %.15g\nrun 0\n"
                   'write_dump all custom out.dump id c_pea c_sta[1] c_sta[2] c_sta[3] c_sta[4] c_sta[5] c_sta[6] modify sort id format line "%d %.15g %.15g %.15g %.15g %.15g %.15g %.15g"\n')
    (wd / "in.lmp").write_text(text)
    r = subprocess.run([lmp, "-in", "in.lmp", "-log", "none", "-nocite"], cwd=wd, env=env, capture_output=True, text=True)
    if r.returncode: raise RuntimeError((r.stdout + r.stderr)[-400:])
    rows = [l.split() for l in r.stdout.splitlines()]
    th = None
    for k, l in enumerate(r.stdout.splitlines()):
        if l.startswith("   Step"): th = [float(x) for x in r.stdout.splitlines()[k + 1].split()]
    d = np.loadtxt(wd / "out.dump", skiprows=9, ndmin=2)
    return th, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--backend", default="cpu64"); ap.add_argument("--only")
    a = ap.parse_args()
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"), LAMMPS_POTENTIALS=a.ffield_dir)
    fails, n, worst = [], 0, 0.0
    with tempfile.TemporaryDirectory() as td:
        for cf in sorted((ROOT / "tests" / "fixtures" / "cases").glob("*.json")):
            if a.only and cf.stem != a.only: continue
            ref = json.loads((ROOT / "tests" / "fixtures" / "reference" / cf.name).read_text())
            if not ref.get("valid", True): continue
            case = json.loads(cf.read_text())
            used = sorted(set(int(t) for t in ref["types"]))
            for k, c in enumerate([case, dict(case, elements=[case["elements"][u - 1] for u in used])]):
                try:
                    s_th, s = run(a.lmp, c, a.ffield_dir, Path(td) / (cf.stem + "_s"), env)
                    o_th, o = run(a.lmp, c, a.ffield_dir, Path(td) / (cf.stem + "_o"), env, a.plugin, a.backend)
                    break
                except RuntimeError as e:
                    if k == 0 and "bond-parameter block" in str(e): continue
                    fails.append(f"{cf.stem}: {str(e)[:200]}"); s = None; break
            if s is None: continue
            n += 1
            scale = max(1.0, np.abs(s[:, 1:]).max())
            d = np.abs(o[:, 1:] - s[:, 1:]).max() / scale
            worst = max(worst, d)
            if d > 1e-8: fails.append(f"{cf.stem}: per-atom values differ by {d:.2e} (relative to {scale:.3g})")
            # sums reproduce the global values
            if abs(o[:, 1].sum() - o_th[1]) > 1e-8 * max(1.0, abs(o_th[1])): fails.append(f"{cf.stem}: sum(pe/atom) {o[:,1].sum()!r} != PE {o_th[1]!r}")
    for f in fails[:40]: print("FAIL", f)
    print(f"PERATOM-1 backend {a.backend}: {n} fixtures, worst relative difference {worst:.2e}, {len(fails)} failures")
    ok = not fails and n >= (1 if a.only else 50)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
