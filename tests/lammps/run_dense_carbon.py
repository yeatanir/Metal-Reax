#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""DENSE-1: dense, highly coordinated, nearly collinear structures that no small fixture contains: random carbon at 1 g/cm3 (random sequential packing) as
packed, after a stock `minimize`, and after a short stock NVT run at 4000 K. On each of these fixed configurations the forces and energy of `reaxff/metal`
(cpu64 and metal) are compared with stock `reaxff` (single point). This test found the float `acos(cos)` angle of the first GPU kernels failing for nearly
collinear atom triples (forces wrong by 1e4 kcal/mol/A while the energy was right).
Atoms of nearly linear bonded triples (sin(angle) < 1e-3, the ill-conditioned class) are excluded from the force-max criterion and reported.
Criteria: cpu64: force max 1e-6 and relative energy 1e-9; metal: owner-set C3 (force max 0.05, RMS 5e-3 kcal/mol/A, energy 1e-3 kcal/mol/atom).
usage: run_dense_carbon.py --lmp BIN --plugin SO --ffield FILE [--n 1000] [--seeds 3]"""
import argparse, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def make_box(path, n, seed, density=1.0, dmin=1.2):
    mass, amu = 12.0107, 1.66053906660e-24
    L = (n * mass * amu / density * 1e24) ** (1.0 / 3.0)
    rng = np.random.default_rng(seed)
    x = rng.random((n, 3)) * L
    for _ in range(1000):
        pairs = cKDTree(x, boxsize=L).query_pairs(dmin, output_type="ndarray")
        if len(pairs) == 0: break
        bad = np.unique(pairs[:, 1]); x[bad] = rng.random((len(bad), 3)) * L
    with open(path, "w") as f:
        f.write(f"random carbon\n\n{n} atoms\n1 atom types\n\n0 {L:.8f} xlo xhi\n0 {L:.8f} ylo yhi\n0 {L:.8f} zlo zhi\n\nMasses\n\n1 {mass}\n\nAtoms # charge\n\n")
        for i, p in enumerate(x): f.write(f"{i + 1} 1 0.0 {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def ill_conditioned_atoms(data, rcut=2.6, sin_min=1e-3):
    """atoms that belong to a bonded triple with sin(angle) < sin_min (nearly linear): the angle forces carry 1/sin(theta) and FP32 error grows like 1/sin^2"""
    L = Path(data).read_text().splitlines()
    n = int([l for l in L if l.endswith("atoms")][0].split()[0])
    box = float([l for l in L if l.endswith("xlo xhi")][0].split()[1])
    k = [i for i, l in enumerate(L) if l.startswith("Atoms")][0] + 2
    rows = sorted([L[k + i].split() for i in range(n)], key=lambda r: int(r[0]))
    x = np.array([[float(v) for v in r[3:6]] for r in rows])
    tree = cKDTree(x, boxsize=box)
    flagged = set()
    for j in range(n):
        nb = [i for i in tree.query_ball_point(x[j], rcut) if i != j]
        for u in range(len(nb)):
            for v in range(u + 1, len(nb)):
                a_, b_ = x[nb[u]] - x[j], x[nb[v]] - x[j]
                a_ -= box * np.round(a_ / box); b_ -= box * np.round(b_ / box)
                if np.linalg.norm(np.cross(a_, b_)) / (np.linalg.norm(a_) * np.linalg.norm(b_)) < sin_min: flagged.update((j, nb[u], nb[v]))
    return np.array(sorted(flagged), dtype=int)


def lmp(a, env, wd, name, text):
    (wd / name).write_text(text)
    r = subprocess.run([a.lmp, "-in", name, "-log", "none", "-nocite"], cwd=wd, env=env, capture_output=True, text=True, timeout=1200)
    if r.returncode: raise RuntimeError((r.stdout + r.stderr)[-600:])
    return r.stdout


def static(a, env, wd, data, style, tag):
    out = lmp(a, env, wd, f"static_{tag}.in", f"""units real
atom_style charge
read_data {data}
{style}
pair_coeff * * {a.ffield} C
neighbor 2.0 bin
thermo_style custom step pe
thermo_modify format float %.15g
dump d all custom 1 f_{tag}.dump id fx fy fz
dump_modify d sort id format float %.15g
run 0
""")
    pe = [l.split() for l in out.splitlines() if l.strip().startswith("0 ")][0]
    return float(pe[1]), np.loadtxt(wd / f"f_{tag}.dump", skiprows=9, ndmin=2)[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--ffield", required=True)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"))
    fails, worst = [], {"c": 0.0, "m": 0.0, "m_rms": 0.0, "e": 0.0}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for seed in range(1, a.seeds + 1):
            make_box(td / f"c{seed}.data", a.n, seed)
            lmp(a, env, td, "prep.in", f"""units real
atom_style charge
read_data c{seed}.data
pair_style reaxff NULL checkqeq no
pair_coeff * * {a.ffield} C
neighbor 2.0 bin
min_style cg
minimize 0.0 1.0e-3 200 10000
write_data c{seed}.min.data
velocity all create 4000.0 {seed} dist gaussian mom yes rot yes
fix n all nvt temp 4000.0 4000.0 25.0
timestep 0.25
run 300
write_data c{seed}.hot.data
""")
            for kind, data in (("packed", f"c{seed}.data"), ("minimised", f"c{seed}.min.data"), ("hot 4000 K", f"c{seed}.hot.data")):
                try:
                    pe_s, f_s = static(a, env, td, data, "pair_style reaxff NULL checkqeq no", "s")
                    res = {}
                    for key, backend in (("c", "cpu64"), ("m", "metal")):
                        res[key] = static(a, env, td, data, f"plugin load {a.plugin}\npair_style reaxff/metal NULL checkqeq no backend {backend}", key)
                except RuntimeError as e:
                    fails.append((seed, kind, str(e))); continue
                for key in "cm":
                    pe, f = res[key]; d = f - f_s
                    ill = ill_conditioned_atoms(td / data)
                    keep = np.ones(len(d), bool); keep[ill] = False
                    fm, fr, de = abs(d[keep]).max(), np.sqrt((d ** 2).mean()), abs(pe - pe_s)
                    if key == "m" and len(ill): worst["ill"] = max(worst.get("ill", 0.0), abs(d[ill]).max())
                    if key == "c":
                        worst["c"] = max(worst["c"], fm)
                        if fm > 1e-6 or de > 1e-9 * abs(pe_s): fails.append((seed, kind, f"cpu64 force max {fm:.2e} energy diff {de:.2e}"))
                    else:
                        worst["m"] = max(worst["m"], fm); worst["m_rms"] = max(worst["m_rms"], fr); worst["e"] = max(worst["e"], de / a.n)
                        if fm > 0.05 or fr > 5e-3 or de / a.n > 1e-3: fails.append((seed, kind, f"metal force max {fm:.2e} rms {fr:.2e} energy/atom {de / a.n:.2e}"))
                print(f"seed {seed} {kind:11s} |F| rms {np.sqrt((f_s ** 2).mean()):8.2f} max {abs(f_s).max():8.1f}")
    for f in fails: print("FAIL", f)
    print(f"DENSE-1: worst force max cpu64 {worst['c']:.2e}, metal {worst['m']:.2e} (rms {worst['m_rms']:.2e}) on well-conditioned atoms; atoms in nearly linear triples (sin < 1e-3) up to {worst.get('ill', 0.0):.2e}; metal energy/atom {worst['e']:.2e}; {len(fails)} failures")
    print("RESULT: PASS" if not fails else "RESULT: FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
