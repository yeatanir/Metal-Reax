#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""M1 experiments E2 (rigid translation / wrap invariance) and E3 (ghost-range sensitivity: neighbor skin and QEq taper radius).
Measurement only; interpretation is recorded in docs/VALIDATION.md.
usage: exp_ghost.py --lmp BIN --ffield-dir DIR --workdir DIR --out FILE [--label L]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner, geometries as g, ref_nonbonded  # noqa: E402

CHO = ["C", "H", "O"]


def mk(cid, at, cell, periodic, swb=10.0, tol=1e-12, skin=2.0, extra_cmds=None):
    c = {"id": cid, "ffield": {"name": "ffield.reax.cho"}, "elements": CHO, "cell": cell, "periodic": list(periodic), "atoms": at,
         "charge": {"model": "qeq/reaxff", "swb": swb, "tolerance": tol, "maxiter": 1000}, "neighbor": {"skin": skin}}
    if extra_cmds: c["extra_commands"] = extra_cmds
    return c


def nb_dev(case, r):
    """deviation of LAMMPS vdW/Coulomb energy from the independent explicit-image reference"""
    P = runner.read_params_dump(Path(r["diag"]["file"]).parent / "params.txt")
    ref = ref_nonbonded.nonbonded_reference(np.array(r["positions"]), np.array(runner.ffield_indices(P, case["elements"]))[np.array(r["types"]) - 1], np.array(r["charges"]),
                                            runner.cell_matrix(case["cell"]), case["periodic"], P, r["diag"]["control"], False)
    ef = r["diag"]["energy_fields"]
    return max(abs(ef["e_vdW"] - ref["e_vdW"]), abs(ef["e_ele"] - ref["e_ele"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--workdir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label", default="exp")
    a = ap.parse_args()
    run = lambda c: runner.run_case(c, a.lmp, a.ffield_dir, Path(a.workdir) / c["id"], a.label, diag=True)
    rng = np.random.default_rng(11)
    out = {"E2_translation": {}, "E3_skin": {}, "E3_swb": {}}

    systems = {}
    dia, diac = g.diamond(3.567, 2)
    systems["diamond_2x2x2_pert"] = ([{"el": q["el"], "xyz": [float(v + rng.uniform(-0.07, 0.07)) for v in q["xyz"]]} for q in dia], diac, (True,) * 3)
    gr, grc = g.graphene_sheet(3, 2)
    systems["graphene_3x2_pert"] = ([{"el": q["el"], "xyz": [float(v + rng.uniform(-0.04, 0.04)) for v in q["xyz"]]} for q in gr], grc, (True, True, False))
    wat = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                w = g.rotate(g.water(), rng.normal(size=3), float(rng.uniform(0, 360)))
                wat += g.translate(w, [(i + 0.5) * 4.8, (j + 0.5) * 4.8, (k + 0.5) * 4.8])
    systems["water8_box9.6"] = (wat, {"lo": [0, 0, 0], "hi": [9.6] * 3}, (True,) * 3)

    # ---- E2: rigid translations by fractions of the cell (no pre-wrapping; LAMMPS remaps on read)
    shifts = [(0, 0, 0), (0.5, 0, 0), (0.37, 0.21, 0.83), (-0.3, 1.2, 2.5), (0.999, 0.001, 0.5)]
    for name, (at, cell, per) in systems.items():
        L = np.array(cell["hi"]) - np.array(cell["lo"])
        base = None; rows = []
        for sh in shifts:
            sv = np.array(sh) * L * np.array(per, float)
            at2 = [{"el": q["el"], "xyz": [float(v) for v in np.array(q["xyz"]) + sv]} for q in at]
            r = run(mk(f"{name}_sh{sh[0]}_{sh[1]}_{sh[2]}", at2, cell, per))
            row = {"shift_frac": sh, "valid": r["valid"], "reasons": r["invalid_reasons"]}
            if r["valid"]:
                s = np.array([r["energy"]["slots"][k] for k in runner.SLOT_NAMES])
                if base is None: base = (s, np.array(r["charges"]), np.array(r["forces"]))
                row["max_slot_dev"] = float(np.abs(s - base[0]).max()); row["total_dev"] = abs(float(s.sum() - base[0].sum()))
                row["max_q_dev"] = float(np.abs(np.array(r["charges"]) - base[1]).max()); row["max_f_dev"] = float(np.abs(np.array(r["forces"]) - base[2]).max())
            rows.append(row)
        out["E2_translation"][name] = rows

    # ---- E3a: neighbor skin sweep at the default taper radius (ghost range = max cutoff + skin)
    for name in ("diamond_2x2x2_pert", "water8_box9.6"):
        at, cell, per = systems[name]
        base = None; rows = []
        for skin in (2.0, 1.0, 0.3, 0.0):
            c = mk(f"{name}_skin{skin}", at, cell, per, skin=skin); r = run(c)
            row = {"skin": skin, "valid": r["valid"], "reasons": r["invalid_reasons"]}
            if "energy" in r:
                if base is None and r["valid"]: base = r["energy"]["total_pe"]
                row["total_pe"] = r["energy"]["total_pe"]; row["dev_vs_skin2"] = (r["energy"]["total_pe"] - base) if base is not None else None
                if "independent_eem" in r: row["eem_residual_eV"] = r["independent_eem"]["equalization_residual_eV"]
                if "diag" in r: row["nonbonded_dev_vs_independent_ref"] = nb_dev(c, r)
            rows.append(row)
        out["E3_skin"][name] = rows

    # ---- E3b: QEq taper radius (fix swb) vs the ghost range of the pair style (nonb_cut 10 + skin 2 = 12 A)
    for name in ("diamond_2x2x2_pert", "water8_box9.6"):
        at, cell, per = systems[name]
        rows = []
        for swb in (6.0, 8.0, 10.0, 11.5, 12.5, 14.0):
            c = mk(f"{name}_swb{swb}", at, cell, per, swb=swb); r = run(c)
            row = {"swb": swb, "valid": r["valid"], "reasons": r["invalid_reasons"]}
            if "independent_eem" in r: row["eem_residual_eV"] = r["independent_eem"]["equalization_residual_eV"]
            if "energy" in r: row["total_pe"] = r["energy"]["total_pe"]
            rows.append(row)
        out["E3_swb"][name] = rows

    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print("==", k)
        for name, rows in v.items():
            print(" ", name)
            for r_ in rows:
                keys = {kk: (f"{vv:.3e}" if isinstance(vv, float) else vv) for kk, vv in r_.items() if kk not in ("reasons",)}
                print("    ", keys, "|", "; ".join(r_["reasons"])[:150])


if __name__ == "__main__":
    main()
