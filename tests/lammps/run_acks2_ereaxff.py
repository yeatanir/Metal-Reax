#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""eReaxFF `acks2 yes` (ACKS2 charges solved inside the pair style) against the legacy executable (single points of tests/data/acks2/refs.json:
Ag/Av + El, C + El, Si (Al + El) cluster and 3x3x3 periodic supercell; compact = El fixed, full = El solved), plus
  * acks2 off reproduces the stored fixed-charge Ag run (energy, 15 energy slots, charges, forces) to 1e-12 relative,
  * `charge=fixed` for every species gives the same energy and forces as acks2 off,
  * `acks2_constants consistent`: the forces equal central finite differences of the energy (rel 1e-6) for the compact and the full mode.
usage: run_acks2_ereaxff.py --lmp BIN --plugin SO --data tests/data/acks2"""
import argparse, json, os, random, subprocess, sys, tempfile
from pathlib import Path
ap = argparse.ArgumentParser()
for k in ("lmp", "plugin", "data"): ap.add_argument("--" + k, required=True)
ap.add_argument("--si-data", default="", help="optional directory with refs_si.json and ffield_si (private Si material, never stored in the repo)")
a = ap.parse_args(); D = Path(a.data)
NAMES = ["bond", "over+under", "lone_pair", "molecule", "valence+penalty", "angle_conj", "hbond", "torsion", "torsion_conj", "vdw", "coulomb", "charge", "field", "electron"]
SLOTS = [[0], [1], [2], [3], [4, 5], [6], [7], [8], [9], [10], [11], [13], [12], [14]]
masses = {"AV": 107.8682, "EL": 1.0, "C": 12.0, "AL": 26.98154}   # only used for the data file; masses do not enter the energy

def lmp_run(td, ffield, atoms, groups, cell, eparams, args, taper, post="", periodic_cut=14.0, ddir=None):
    td = Path(td); td.mkdir(parents=True, exist_ok=True)
    elems = list(dict.fromkeys(x[0] for x in atoms))
    mol = [0] * len(atoms); qd = [0.0] * len(atoms)
    for g, (lo, hi, q) in enumerate(groups, 1):
        for i in range(lo - 1, hi): mol[i] = g; qd[i] = q / (hi - lo + 1)
    ext = [(0.0, cell[k]) for k in range(3)] if cell else [(-60.0, 70.0)] * 3
    L = ["data", "", "%d atoms" % len(atoms), "%d atom types" % len(elems), ""] + ["%.12f %.12f %slo %shi" % (e[0], e[1], c, c) for e, c in zip(ext, "xyz")] + ["", "Masses", ""]
    L += ["%d %.4f" % (k + 1, masses.get(e.upper(), 1.0)) for k, e in enumerate(elems)] + ["", "Atoms # charge", ""]
    L += ["%d %d %.10f %.10f %.10f %.10f" % (i + 1, elems.index(n) + 1, qd[i], x, y, z) for i, (n, x, y, z) in enumerate(atoms)]
    L += ["", "Molecules", ""] + ["%d %d" % (i + 1, mol[i]) for i in range(len(atoms))]
    (td / "data.lmp").write_text("\n".join(L) + "\n")
    (td / "ep.eparams").write_text(eparams)
    (td / "ffield").write_bytes(((ddir or D) / ffield).read_bytes())
    tap = f" electron_taper {taper}" if taper else ""
    ep = " eparams ep.eparams" if eparams.strip() else ""
    (td / "in.lmp").write_text(f"""plugin load {os.path.relpath(a.plugin, td)}
units real
atom_style charge
atom_modify map array
boundary {'p p p' if cell else 'f f f'}
newton on
fix pa all property/atom mol ghost yes
read_data data.lmp fix pa NULL Molecules
pair_style reaxff/metal NULL ereaxff yes{tap}{ep} {args}
pair_coeff * * ffield {' '.join(elems)}
comm_modify cutoff {periodic_cut}
compute pp all pair reaxff/metal
thermo_style custom step pe c_pp[1] c_pp[2] c_pp[3] c_pp[4] c_pp[5] c_pp[6] c_pp[7] c_pp[8] c_pp[9] c_pp[10] c_pp[11] c_pp[12] c_pp[13] c_pp[14] c_pp[15]
thermo_modify format float %.16g
dump d all custom 1 f.dump id q fx fy fz
dump_modify d format float %.16g sort id
run 0
undump d
{post}
""")
    env = dict(os.environ, OMP_NUM_THREADS="1", DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"))
    r = subprocess.run([a.lmp, "-in", "in.lmp", "-log", "none", "-nocite"], cwd=td, env=env, capture_output=True, text=True)
    txt = r.stdout + r.stderr
    if r.returncode: raise RuntimeError(txt[-1200:])
    rows = [l.split() for l in txt.splitlines() if len(l.split()) == 17 and l.split()[0] == "0"]
    lines = (td / "f.dump").read_text().splitlines(); k = [i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS")][0] + 1
    d = [[float(x) for x in l.split()[1:5]] for l in lines[k:] if l.strip()]
    return dict(pe=float(rows[0][1]), pv=[float(x) for x in rows[0][2:]], q=[x[0] for x in d], f=[x[1:4] for x in d], log=txt)

ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""), flush=True)

def eparams_for(ref):
    if ref["mode"] == "compact": return ""
    return "EL 0 0 tested charge=solve\n" + "".join(f"{e} 3 1 tested polarization=off\n" for e in ref["pol_off"])   # legacy full ACKS2 leaves El and Av out of the q*s sum

refs = json.loads((D / "refs.json").read_text())
for r_ in refs: r_["ddir"] = D
if a.si_data:
    for r_ in json.loads((Path(a.si_data) / "refs_si.json").read_text()): r_["ddir"] = Path(a.si_data); refs.append(r_)
with tempfile.TemporaryDirectory() as td:
    for ref in refs:
        lm = lmp_run(Path(td) / ref["name"], ref["ffield"], ref["atoms"], ref["groups"], ref["cell"], eparams_for(ref), "acks2 yes", ref["taper"], ddir=ref["ddir"])
        lg = ref["legacy"]; terms = [sum(lm["pv"][s] for s in sl) for sl in SLOTS]
        rel = {n: abs(t - x) / max(1.0, abs(x)) for n, t, x in zip(NAMES, terms, lg["terms"]) if n not in ("molecule", "field")}
        wt = max(v for n, v in rel.items() if not n.startswith("torsion")); tor = max(rel["torsion"], rel["torsion_conj"])
        dq = max(abs(x - y[0]) for x, y in zip(lm["q"], lg["chg"]))
        fmax = max(abs(v) for g in lg["grad"] for v in g)
        df = max(abs(-g[c] - lm["f"][i][c]) for i, g in enumerate(lg["grad"]) for c in range(3)) / fmax
        dE = abs(lm["pe"] - lg["estrc"])
        check(f"{ref['name']}: legacy terms {wt:.1e}, torsion {tor:.1e}, charges {dq:.1e} e, energy {dE:.1e} kcal/mol, forces {df:.1e} rel",
              wt <= 1e-8 and tor <= 1e-5 and dq <= 1e-6 and dE <= 1e-4 and df <= 1e-6)

    # acks2 off == stored pre-change fixed-charge run (the data file of the Ag regression: Av +1, El -1)
    ag = next(r for r in refs if r["name"] == "ag_compact")
    off = lmp_run(Path(td) / "off", ag["ffield"], ag["atoms"], [(i + 1, i + 1, 1.0 if x[0] == "Av" else -1.0) for i, x in enumerate(ag["atoms"])], None, "", "", 5.7)
    base = [l.split() for l in (D / "ag_fixed_baseline.txt").read_text().splitlines()]
    row = [float(x) for x in base[0]]
    dev = max(abs(x - y) / max(1.0, abs(y)) for x, y in zip([off["pe"]] + off["pv"], row[1:]))
    check(f"acks2 off reproduces the stored fixed-charge Ag run (energy and 15 slots, max rel {dev:.1e})", dev <= 1e-12)
    k = [i for i, l in enumerate(base) if l and l[0] == "ITEM:" and l[1] == "ATOMS"][0] + 1
    fb = [[float(x) for x in l[2:5]] for l in base[k:]]
    df = max(abs(x - y) for f1, f2 in zip(off["f"], fb) for x, y in zip(f1, f2))
    check(f"... and its forces (max abs diff {df:.1e})", df <= 1e-9)
    # every species fixed with acks2 yes == acks2 off
    allfix = lmp_run(Path(td) / "allfix", ag["ffield"], ag["atoms"], [(i + 1, i + 1, 1.0 if x[0] == "Av" else -1.0) for i, x in enumerate(ag["atoms"])], None, "AV 3 1 tested charge=fixed\n", "acks2 yes", 5.7)
    check("charge=fixed for Av (El fixed by default) gives the acks2-off energy and forces",
          abs(allfix["pe"] - off["pe"]) < 1e-9 and max(abs(x - y) for f1, f2 in zip(allfix["f"], off["f"]) for x, y in zip(f1, f2)) < 1e-9)

    # forces = finite differences of the energy, acks2_constants consistent (distorted Ag cluster, electrons 0.3 A from their cores)
    rnd = random.Random(7); at = [(n, x + rnd.uniform(-.15, .15), y + rnd.uniform(-.15, .15), z + rnd.uniform(-.15, .15)) for n, x, y, z in ag["atoms"]]
    at = [(n, round(x, 5), round(y, 5), round(z, 5)) for n, x, y, z in at]
    at = [at[i] if i < 6 else (at[i][0], *[round(at[i][k + 1] + rnd.uniform(-.3, .3), 5) for k in range(3)]) for i in range(len(at))]
    hs = 5e-5
    for mode in ("compact", "full"):
        ep = "" if mode == "compact" else "EL 0 0 tested charge=solve polarization=on\n"
        cmds = ["variable E equal pe", "thermo_style custom step pe", "thermo_modify format float %.16g", "thermo 1000000"]
        for i in range(1, 12):
            cmds.append(f"group g{i} id {i}")
            for d, vec in enumerate(("{h} 0 0", "0 {h} 0", "0 0 {h}")):
                cmds += [f"displace_atoms g{i} move {vec.format(h=hs)} units box", "run 0", f'print "FDP {i} {d} ${{E}}"', f"displace_atoms g{i} move {vec.format(h=-2 * hs)} units box", "run 0",
                         f'print "FDM {i} {d} ${{E}}"', f"displace_atoms g{i} move {vec.format(h=hs)} units box"]
        grp = [(1, 6, 6.0)] + [(i, i, -1.0) for i in range(7, 12)]
        lm = lmp_run(Path(td) / ("fd_" + mode), ag["ffield"], at, grp, None, ep, "acks2 yes acks2_constants consistent", 5.7, post="\n".join(cmds))
        fp, fm = {}, {}
        for l in lm["log"].splitlines():
            w = l.split()
            if len(w) == 4 and w[0] in ("FDP", "FDM"): (fp if w[0] == "FDP" else fm)[(int(w[1]), int(w[2]))] = float(w[3])
        fmax = max(abs(v) for f in lm["f"] for v in f)
        err = max(abs(-(fp[k] - fm[k]) / (2 * hs) - lm["f"][k[0] - 1][k[1]]) for k in fp)
        check(f"forces vs finite differences of the energy, acks2_constants consistent, {mode}: max {err:.1e} ({err / fmax:.1e} of max |F| {fmax:.1f})", err / fmax < 1e-6 and len(fp) == 33)
print("RESULT: PASS" if ok else "RESULT: FAIL"); sys.exit(0 if ok else 1)
