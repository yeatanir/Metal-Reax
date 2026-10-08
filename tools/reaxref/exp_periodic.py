#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""M1 periodic-image accounting experiments against pinned LAMMPS (instrumented build).

E1 supercell invariance : replicate a (perturbed) periodic cell n x n x n; every energy slot must scale by n^3 and per-atom
                          charges / forces must repeat exactly, if the pinned LAMMPS counts images correctly.
E2 translation/wrap     : rigidly translate all atoms by a fraction of the cell (wrapped and unwrapped input); results invariant.
E3 ghost-cutoff sweep   : (see exp_ghost.py)
Output: JSON to --out with per-case deviations. This script only MEASURES; classification of any deviation is documented in
VALIDATION.md. usage: exp_periodic.py --lmp BIN --ffield-dir DIR --workdir DIR --out FILE"""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner, geometries as g  # noqa: E402

CHO = ["C", "H", "O"]


def replicate(at, cell, n):
    lo = np.array(cell["lo"], float); L = np.array(cell["hi"], float) - lo
    out = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                sh = np.array([i, j, k]) * L
                out += [{"el": a["el"], "xyz": [float(v) for v in np.array(a["xyz"]) + sh]} for a in at]
    return out, {"lo": cell["lo"], "hi": [float(v) for v in lo + L * np.array(n)]}


def mk(cid, ff, els, at, cell, periodic, tol=1e-12, extra=None):
    c = {"id": cid, "ffield": {"name": "ffield.reax." + ff}, "elements": els, "cell": cell, "periodic": list(periodic), "atoms": at,
         "charge": {"model": "qeq/reaxff", "swb": 10.0, "tolerance": tol, "maxiter": 1000}}
    if extra: c.update(extra)
    return c


def run(case, a, tag):
    return runner.run_case(case, a.lmp, a.ffield_dir, Path(a.workdir) / tag, a.label, diag=True)


def slots_vec(r):
    return np.array([r["energy"]["slots"][s] for s in runner.SLOT_NAMES])


def supercell_experiment(a, name, ff, els, at, cell, periodic, reps, tol=1e-12, extra=None):
    base = None; rows = []
    for n in reps:
        at_n, cell_n = replicate(at, cell, n)
        r = run(mk(f"{name}_{n[0]}{n[1]}{n[2]}", ff, els, at_n, cell_n, periodic, tol, extra), a, f"{name}_{n[0]}{n[1]}{n[2]}")
        mult = n[0] * n[1] * n[2]
        row = {"reps": n, "natoms": len(at_n), "valid": r["valid"], "reasons": r["invalid_reasons"]}
        if r["valid"]:
            if base is None:
                base = (mult, r)
            nat0 = len(at)
            e = slots_vec(r) / mult
            # per-replica charges / forces: compare every replica copy with the first
            q = np.array(r["charges"]).reshape(mult, nat0); F = np.array(r["forces"]).reshape(mult, nat0, 3)
            row["energy_total_per_cell"] = r["energy"]["total_pe"] / mult
            row["slots_per_cell"] = dict(zip(runner.SLOT_NAMES, e.tolist()))
            row["max_charge_spread_over_replicas"] = float(np.abs(q - q[0]).max())
            row["max_force_spread_over_replicas"] = float(np.abs(F - F[0]).max())
            row["_q0"] = q[0].tolist(); row["_f0"] = F[0].tolist()     # first replica, for cross-run comparison
            row["tallies"] = {k: v["count"] for k, v in r["diag"]["tallies"].items()}
        rows.append(row)
    ref = next(r_ for r_ in rows if r_["valid"])
    for r_ in rows:
        if r_["valid"]:
            r_["max_slot_dev_vs_first_per_cell"] = max(abs(r_["slots_per_cell"][s] - ref["slots_per_cell"][s]) for s in runner.SLOT_NAMES)
            r_["total_dev_vs_first_per_cell"] = abs(r_["energy_total_per_cell"] - ref["energy_total_per_cell"])
            r_["max_charge_dev_vs_first"] = r_["_q0"] - ref["_q0"] if False else float(np.abs(np.array(r_["_q0"]) - np.array(ref["_q0"])).max())
            r_["max_force_dev_vs_first"] = float(np.abs(np.array(r_["_f0"]) - np.array(ref["_f0"])).max())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--workdir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label", default="exp")
    a = ap.parse_args()
    res = {}
    # perturbed primitive cells (seeded) so that forces are non-zero and nothing cancels by symmetry
    rng = np.random.default_rng(7)
    def pert(at, amp):
        return [{"el": q["el"], "xyz": [float(v + rng.uniform(-amp, amp)) for v in q["xyz"]]} for q in at]
    dia, diac = g.diamond(3.567, 1)
    res["diamond_8atoms"] = supercell_experiment(a, "dia", "cho", CHO, pert(dia, 0.07), diac, (True,) * 3, [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (2, 2, 2), (3, 3, 3)])
    gr, grc = g.graphene_sheet(1, 1)
    res["graphene_4atoms_2d"] = supercell_experiment(a, "gra", "cho", CHO, pert(gr, 0.04), grc, (True, True, False), [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (3, 3, 1), (4, 4, 1)])
    au, auc = g.fcc("Au", 4.078, 1)
    res["au_fcc_4atoms"] = supercell_experiment(a, "au", "AuO", ["H", "O", "Au"], pert(au, 0.05), auc, (True,) * 3, [(1, 1, 1), (2, 2, 2), (3, 3, 3)], tol=1e-11)
    # one-atom-per-cell chains: bonds/angles/torsions where every participating atom is an image of the same atom
    chain = [{"el": "C", "xyz": [0.5, 5.0, 5.0]}]
    chc = {"lo": [0, 0, 0], "hi": [1.30, 20.0, 20.0]}
    res["c_chain_1atom_period1.30"] = supercell_experiment(a, "chn", "cho", CHO, chain, chc, (True, False, False), [(1, 1, 1), (2, 1, 1), (3, 1, 1), (5, 1, 1), (8, 1, 1)], extra={"neighbor": {"skin": 1.0}})
    zz = [{"el": "C", "xyz": [0.4, 5.0, 5.0]}, {"el": "C", "xyz": [1.55, 5.7, 5.0]}]
    zzc = {"lo": [0, 0, 0], "hi": [2.5, 20.0, 20.0]}
    res["c_zigzag_2atom_period2.5"] = supercell_experiment(a, "zig", "cho", CHO, pert(zz, 0.03), zzc, (True, False, False), [(1, 1, 1), (2, 1, 1), (3, 1, 1), (6, 1, 1)], extra={"neighbor": {"skin": 1.0}})
    w = g.water(); box = 6.2
    wat = [dict(q_) for q_ in g.translate(w, [3.0, 3.0, 3.0])]
    res["water_1mol_periodic_6.2"] = supercell_experiment(a, "w1", "cho", CHO, pert(wat, 0.05), {"lo": [0, 0, 0], "hi": [box] * 3}, (True,) * 3, [(1, 1, 1), (2, 2, 2), (3, 3, 3)], tol=1e-11)
    # ---- charged periodic systems (single-element crystals have q = 0 by symmetry and cannot test QEq images)
    me = g.translate(g.methane(), [3.0, 3.0, 3.0])
    res["methane_1mol_periodic_6.0"] = supercell_experiment(a, "me1", "cho", CHO, pert(me, 0.05), {"lo": [0, 0, 0], "hi": [6.0] * 3}, (True,) * 3, [(1, 1, 1), (2, 2, 2), (3, 3, 3)], tol=1e-11)
    wd = g.translate(g.water_dimer(), [3.0, 3.0, 3.0])
    res["water_dimer_periodic_7.0"] = supercell_experiment(a, "wd1", "cho", CHO, pert(wd, 0.05), {"lo": [0, 0, 0], "hi": [7.0] * 3}, (True,) * 3, [(1, 1, 1), (2, 2, 2), (3, 3, 3)], tol=1e-11)
    wd9 = g.translate(g.water_dimer(), [4.5, 4.5, 4.5])
    res["water_dimer_periodic_9.0"] = supercell_experiment(a, "wd9", "cho", CHO, pert(wd9, 0.05), {"lo": [0, 0, 0], "hi": [9.0] * 3}, (True,) * 3, [(1, 1, 1), (2, 2, 2)], tol=1e-11)
    pe = []   # all-trans polyethylene, 2 CH2 per period along x
    for cx, cz, sg in ((0.0, 0.4307, 1.0), (1.2765, -0.4307, -1.0)):
        pe.append({"el": "C", "xyz": [cx, 5.0, 5.0 + cz]})
        for sy in (1.0, -1.0):
            pe.append({"el": "H", "xyz": [cx, 5.0 + sy * 1.09 * np.sin(54.75 * np.pi / 180), 5.0 + cz + sg * 1.09 * np.cos(54.75 * np.pi / 180)]})
    res["polyethylene_6atoms_period2.553"] = supercell_experiment(a, "pe", "cho", CHO, pert(pe, 0.03), {"lo": [0, 0, 0], "hi": [2.553, 10.0, 10.0]}, (True, False, False), [(1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1)], tol=1e-11, extra={"neighbor": {"skin": 1.0}})
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, rows in res.items():
        print("==", k)
        for r_ in rows:
            if not r_["valid"]: print("  ", r_["reps"], "INVALID", r_["reasons"]); continue
            print(f"   reps={r_['reps']} N={r_['natoms']:4d} E/cell={r_['energy_total_per_cell']:.9f} max|dSlot|={r_['max_slot_dev_vs_first_per_cell']:.2e}"
                  f" dE_tot={r_['total_dev_vs_first_per_cell']:.2e} q-dev={r_['max_charge_dev_vs_first']:.1e} F-dev={r_['max_force_dev_vs_first']:.1e}  max|q|={max(abs(v) for v in r_['_q0']):.2e}")


if __name__ == "__main__":
    main()
