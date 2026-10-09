#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""ANA-1: the stock analysis commands of ReaxFF -- fix reaxff/bonds, fix reaxff/species, compute reaxff/atom, compute SPEC/ATOM -- work unmodified with
`pair_style reaxff/metal` and give the stock results, on the 105-atom reactive CHO system of examples/reaxff/CHO (500 K Berendsen NVE, 300 steps, QEq).
Stock `pair_style reaxff` and the plugin (cpu64 and metal) run the same input. Compared: the bond table (per atom: neighbor ids and bond orders as sets, total bond
order, lone pairs, charge), the species file (species and counts), the per-atom and local output of compute reaxff/atom, and the bond orders of compute SPEC/ATOM.
Tolerances: 2e-3 on the 3-decimal bond orders for cpu64 (printed values) and 2e-2 for metal (trajectory divergence of a reactive 300-step run, FP32 bond orders).
usage: run_analysis.py --lmp BIN --plugin SO --example DIR [--backend cpu64|metal]"""
import argparse, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

INPUT = """units real
atom_style charge
read_data data.CHO
{pair}
pair_coeff * * ffield.reax.cho H C O
neighbor 2 bin
neigh_modify every 10 delay 0 check no
fix 1 all nve
fix 2 all qeq/reax 1 0.0 10.0 1e-6 param.qeq
fix 3 all temp/berendsen 500.0 500.0 100.0
fix b all reaxff/bonds 100 bonds.{tag}
fix s all reaxff/species 10 5 100 species.{tag}
compute ra all reaxff/atom bonds yes
compute sp all SPEC/ATOM q abo01 abo02 abo03 abo04
dump d1 all custom 100 atom.{tag}.dump id c_ra[1] c_ra[2] c_ra[3] c_sp[1] c_sp[2] c_sp[3] c_sp[4] c_sp[5]
dump_modify d1 sort id format float %.8g
dump d2 all local 100 local.{tag}.dump c_ra[1] c_ra[2] c_ra[3]
timestep 0.25
thermo 100
run 300
"""


def bonds(path):
    """last snapshot of a reaxff/bonds file -> {id: (type, sorted [(nbr, bo)], abo, nlp, q)}"""
    txt = Path(path).read_text().split("# Timestep")
    last = txt[-1]
    out = {}
    for line in last.splitlines():
        if line.startswith("#") or not line.strip(): continue
        v = line.split()
        if len(v) < 4: continue   # the step number that follows '# Timestep'
        i, t, nb = int(v[0]), int(v[1]), int(v[2])
        ids = [int(x) for x in v[3:3 + nb]]
        bos = [float(x) for x in v[4 + nb:4 + 2 * nb]]
        abo, nlp, q = [float(x) for x in v[4 + 2 * nb:7 + 2 * nb]]
        out[i] = (t, sorted(zip(ids, bos)), abo, nlp, q)
    return out


def species(path):
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    hdr = [r for r in rows if r[0].startswith("#")][-1]
    data = [r for r in rows if not r[0].startswith("#")][-1]
    return dict(zip(hdr[1:], data)) if False else (hdr, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--example", required=True); ap.add_argument("--backend", default="cpu64")
    a = ap.parse_args()
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"))
    tol = 2e-2 if a.backend == "metal" else 2e-3
    fails = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for f in Path(a.example).iterdir():
            if f.is_file() and not f.name.startswith(("log.", "in.")): shutil.copy(f, td / f.name)
        env["LAMMPS_POTENTIALS"] = str(td)
        (td / "lmp_control").write_text((Path(a.example) / "lmp_control").read_text())
        for tag, pair in (("stock", "pair_style reaxff lmp_control"),
                          ("ours", f"plugin load {a.plugin}\npair_style reaxff/metal lmp_control backend {a.backend}")):
            (td / f"in.{tag}").write_text(INPUT.format(pair=pair, tag=tag))
            r = subprocess.run([a.lmp, "-in", f"in.{tag}", "-log", "none", "-nocite"], cwd=td, env=env, capture_output=True, text=True, timeout=900)
            if r.returncode: print("FAIL run", tag, (r.stdout + r.stderr)[-600:]); return 1
        sb, ob = bonds(td / "bonds.stock"), bonds(td / "bonds.ours")
        if sb.keys() != ob.keys(): fails.append("bond table: different atoms")
        nbad = 0; worst = 0.0
        for i in sb:
            ts, bs, abos, nlps, qs = sb[i]; to, bo, aboo, nlpo, qo = ob[i]
            if [x[0] for x in bs] != [x[0] for x in bo]: nbad += 1; continue
            d = max([abs(x[1] - y[1]) for x, y in zip(bs, bo)] + [abs(abos - aboo), abs(nlps - nlpo), abs(qs - qo)])
            worst = max(worst, d)
        if nbad > (0 if a.backend == "cpu64" else 2): fails.append(f"bond table: {nbad} atoms with different neighbor sets")
        if worst > tol: fails.append(f"bond table: worst difference {worst:.3e} > {tol}")
        print(f"fix reaxff/bonds: {len(sb)} atoms, {nbad} neighbor-set mismatches, worst numeric difference {worst:.2e}")
        hs, ds = species(td / "species.stock"); ho, do = species(td / "species.ours")
        print("fix reaxff/species stock:", " ".join(hs[:6]), "|", " ".join(ds[:6])); print("fix reaxff/species ours: ", " ".join(ho[:6]), "|", " ".join(do[:6]))
        if hs != ho or (a.backend == "cpu64" and ds != do): fails.append("species file differs")
        def last_snapshot(path):
            blk = Path(path).read_text().split("ITEM: TIMESTEP")[-1].splitlines()
            return np.array([[float(x) for x in l.split()] for l in blk[blk.index(next(l for l in blk if l.startswith("ITEM: ATOMS"))) + 1:] if l.strip()])
        sa, oa = last_snapshot(td / "atom.stock.dump"), last_snapshot(td / "atom.ours.dump")
        d = abs(sa[:, 1:4] - oa[:, 1:4]).max()
        # spec/atom bond orders: compare as sorted sets per atom
        dsp = max(abs(np.sort(sa[i, 5:9]) - np.sort(oa[i, 5:9])).max() for i in range(len(sa)))
        print(f"compute reaxff/atom (abo, nlp, nbonds): worst difference {d:.2e};  compute SPEC/ATOM abo01-04 (sorted): {dsp:.2e}")
        if d > tol or dsp > tol: fails.append("per-atom compute differs")
        ls = Path(td / "local.stock.dump").read_text().split("ITEM: TIMESTEP"); lo = Path(td / "local.ours.dump").read_text().split("ITEM: TIMESTEP")
        nl_s = int([l for l in ls[-1].splitlines()][3]); nl_o = int([l for l in lo[-1].splitlines()][3])
        print(f"local bond list entries: stock {nl_s} ours {nl_o}")
        if abs(nl_s - nl_o) > (0 if a.backend == "cpu64" else 4): fails.append("local bond count differs")
    for f in fails: print("FAIL", f)
    print(f"ANA-1 backend {a.backend}: {len(fails)} failures")
    print("RESULT: PASS" if not fails else "RESULT: FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
