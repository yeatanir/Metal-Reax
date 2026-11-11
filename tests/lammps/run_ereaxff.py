#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""eReaxFF through LAMMPS (`pair_style reaxff/metal ... ereaxff yes`) against exact reference data from the rebuilt legacy executable.

Single case:  run_ereaxff.py --lmp BIN --plugin SO --ffield F --geo GEO --export ereac_export.out [--periodic] [--taper 10]
Frames:       run_ereaxff.py --lmp BIN --plugin SO --ffield F --geo GEO --frames frames.json  --periodic

Compared per case/frame: the 14 energy terms (pair pvector slots; ev+epen merged as in the legacy ev+epen column), the total energy, and the total
gradient (LAMMPS force = -gradient). Atom ids follow the order of the geo file; charges come from its MOLCHARGE lines (legacy icharg 8), type 1 = C, type 2 = El.
Reference units: kcal/mol, kcal/mol/A. Torsion and torsion_conj are reported separately (known stock-vs-legacy residual, task 0.4).
Exit status 0 when every compared quantity is within the tolerances: terms --tol-term (relative, torsion excluded), --tol-torsion, forces --tol-force (relative to the largest reference force)."""
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path

NAMES = ["bond", "over+under", "lone_pair", "molecule", "valence+penalty", "angle_conj", "hbond", "torsion", "torsion_conj", "vdw", "coulomb", "charge", "field", "electron"]
# index in the 15-slot LAMMPS pvector (eb ea elp emol ev epen ecoa ehb et eco ew ep efi eqeq eelect) for each of the 14 reference terms
SLOTS = [[0], [1], [2], [3], [4, 5], [6], [7], [8], [9], [10], [11], [13], [12], [14]]
CHECK = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13]   # emol, ech, efi are zero in both codes (legacy icharg 8)


def read_geo(path):
    sym, pos, groups, cell = [], [], [], None
    for line in Path(path).read_text().splitlines():
        w = line.split()
        if not w: continue
        if w[0] == "HETATM":   # fixed format (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5): element = atom-type field, x y z = columns 31-60
            raw = line.ljust(60); sym.append(raw[25:30].strip()); pos.append([float(raw[30:40]), float(raw[40:50]), float(raw[50:60])])
        elif w[0] == "MOLCHARGE": groups.append((int(w[1]), int(w[2]), float(w[3])))
        elif w[0] == "CRYSTX": cell = [float(x) for x in w[1:7]]
    q = [0.0] * len(sym)
    for a, b, c in groups:
        for i in range(a - 1, b): q[i] = c / (b - a + 1)
    return sym, pos, q, cell


def ffield_masses(ffield):
    """legacy masses (atom line 1, column 3, f9.4) by element name, via the independent fixed-column slicer"""
    sl = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve().parents[1] / "python" / "legacy_ffield_slice.py"), ffield], capture_output=True, text=True, check=True)
    return {w[1].upper(): float(w[4]) for w in (l.split() for l in sl.stdout.splitlines()) if w[0] == "ATOM"}


def elements_in_order(sym):
    return list(dict.fromkeys(sym))   # LAMMPS type k = k-th distinct element in file order


def write_data(path, sym, pos, q, cell, periodic, masses):   # masses: dict element(upper) -> amu, read from the ffield (legacy values)
    ext = [(0.0, cell[0]), (0.0, cell[1]), (0.0, cell[2])] if periodic else [(-60.0, 70.0)] * 3
    lines = ["LAMMPS data file (eReaxFF case)", "", "%d atoms" % len(sym), "%d atom types" % len(elements_in_order(sym)), "",
             "%.12f %.12f xlo xhi" % ext[0], "%.12f %.12f ylo yhi" % ext[1], "%.12f %.12f zlo zhi" % ext[2], "",
             "Masses", ""] + ["%d %.6f" % (k + 1, masses[e.upper()]) for k, e in enumerate(elements_in_order(sym))] + ["", "Atoms # charge", ""]
    for i, (s, p, c) in enumerate(zip(sym, pos, q)):
        t = elements_in_order(sym).index(s) + 1
        lines.append("%d %d %.10f %.15f %.15f %.15f" % (i + 1, t, c, p[0], p[1], p[2]))
    Path(path).write_text("\n".join(lines) + "\n")


def run_lammps(lmp, plugin, ffield, workdir, periodic, taper, elems, extra=""):
    cutoff = 11.5 if "Av" not in elems else 12.0
    inp = f"""plugin load {plugin}
units real
atom_style charge
boundary {'p p p' if periodic else 'f f f'}
newton on
read_data data.lmp
pair_style reaxff/metal NULL ereaxff yes electron_taper {taper} {extra}
pair_coeff * * {ffield} {' '.join(elems)}
comm_modify cutoff {cutoff}
compute pp all pair reaxff/metal
thermo_style custom step pe c_pp[1] c_pp[2] c_pp[3] c_pp[4] c_pp[5] c_pp[6] c_pp[7] c_pp[8] c_pp[9] c_pp[10] c_pp[11] c_pp[12] c_pp[13] c_pp[14] c_pp[15]
thermo_modify format float %.16g
dump d all custom 1 forces.dump id fx fy fz
dump_modify d format float %.16g sort id
run 0
"""
    (Path(workdir) / "in.lmp").write_text(inp)
    env = dict(os.environ, OMP_NUM_THREADS="1", DYLD_LIBRARY_PATH=str(Path(lmp).resolve().parents[1] / "lib"))
    r = subprocess.run([lmp, "-in", "in.lmp", "-log", "none", "-nocite"], cwd=workdir, env=env, capture_output=True, text=True)
    if r.returncode: raise RuntimeError((r.stdout + r.stderr)[-1500:])
    rows, on = [], False
    for line in r.stdout.splitlines():
        if line.lstrip().startswith("Step"): on = True; continue
        if line.startswith("Loop time"): on = False
        if on:
            v = line.split()
            if len(v) == 17: rows.append([float(x) for x in v])
    if not rows: raise RuntimeError("no thermo row:\n" + r.stdout[-1500:])
    row = rows[-1]; pe, pv = row[1], row[2:]
    lines = (Path(workdir) / "forces.dump").read_text().splitlines()
    k = [i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS")][0] + 1
    f = [[float(x) for x in l.split()[1:4]] for l in lines[k:k + len(lines) - k] if l.strip()]
    return pe, pv, f


def read_export(path):
    terms, est, grad, step = None, None, [], None
    for line in Path(path).read_text().splitlines():
        w = line.split()
        if w[0] == "STEP":
            step = int(w[1])
            if step > 0: break
        elif step == 0 and w[0] == "TERMS": terms = [float(x) for x in w[1:15]]
        elif step == 0 and w[0] == "ESTRC": est = float(w[1])
        elif step == 0 and w[0] == "GRAD": grad.append([float(x) for x in w[2:5]])
    return terms, est, grad


def compare(pe, pv, f, terms, est, grad):
    got = [sum(pv[s] for s in sl) for sl in SLOTS]
    rel = {}
    for i in CHECK:
        rel[NAMES[i]] = abs(got[i] - terms[i]) / max(1.0, abs(terms[i]))
    fmax = max(abs(x) for g in grad for x in g)
    fdiff = max(abs(-g[c] - f[i][c]) for i, g in enumerate(grad) for c in range(3))
    return {"dE_total": pe - est, "terms": rel, "fdiff": fdiff, "fmax": fmax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--ffield", required=True)
    ap.add_argument("--geo", required=True); ap.add_argument("--export"); ap.add_argument("--frames")
    ap.add_argument("--periodic", action="store_true"); ap.add_argument("--zero-charges", action="store_true", help="use q=0 (references whose geo has END before the MOLCHARGE lines: the legacy reader then sees no charges)"); ap.add_argument("--taper", type=float, default=10.0)
    ap.add_argument("--tol-term", type=float, default=1e-8); ap.add_argument("--tol-torsion", type=float, default=2e-6)
    ap.add_argument("--tol-force", type=float, default=1e-6); ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    sym, pos, q, cell = read_geo(a.geo)
    if a.periodic and cell is None: sys.exit("--periodic needs a CRYSTX line in the geo")
    if a.zero_charges: q = [0.0] * len(q)
    masses = ffield_masses(a.ffield); elems = elements_in_order(sym)
    cases = []
    if a.frames:
        d = json.loads(Path(a.frames).read_text())
        for fr in d["frames"][: a.max_frames or None]:
            cases.append(("step %d" % fr["step"], fr["pos"], fr["terms"], fr["estrc"], fr["grad"]))
    else:
        t, e, g = read_export(a.export); cases.append(("case", pos, t, e, g))
    worst = dict(term=0.0, torsion=0.0, force=0.0, energy=0.0); ok = True
    with tempfile.TemporaryDirectory() as td:
        for name, p, t, e, g in cases:
            write_data(Path(td) / "data.lmp", sym, p, q, cell, a.periodic, masses)
            pe, pv, f = run_lammps(a.lmp, a.plugin, a.ffield, td, a.periodic, a.taper, elems)
            c = compare(pe, pv, f, t, e, g)
            tors = max(c["terms"]["torsion"], c["terms"]["torsion_conj"])
            other_name, other = max(((k, v) for k, v in c["terms"].items() if not k.startswith("torsion")), key=lambda kv: kv[1])
            fr = c["fdiff"] / c["fmax"]
            worst["term"] = max(worst["term"], other); worst["torsion"] = max(worst["torsion"], tors)
            worst["force"] = max(worst["force"], fr); worst["energy"] = max(worst["energy"], abs(c["dE_total"]))
            good = other <= a.tol_term and tors <= a.tol_torsion and fr <= a.tol_force
            ok &= good
            print("%-10s dE_total %+10.3e  max term rel %.2e (%s)  torsion rel %.2e  force max|dF| %.2e (%.1e of %.3f)  %s" % (name, c["dE_total"], other, other_name, tors, c["fdiff"], fr, c["fmax"], "ok" if good else "FAIL"))
    print("WORST over %d case(s): |dE_total| %.3e kcal/mol, terms (no torsion) %.2e, torsion %.2e, force %.2e relative" % (len(cases), worst["energy"], worst["term"], worst["torsion"], worst["force"]))
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
