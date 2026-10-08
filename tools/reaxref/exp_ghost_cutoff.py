#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""REF-GHOST: how large must the ghost shell be for pinned LAMMPS to give the same answer as an unlimited shell?
The shell is max(nonb_cut, hbond_cut, bond_cut) + neighbor skin (pair_reaxff.cpp:370). We shrink `nonb_cut` (ffield general
parameter 13) so that cutmax < 2*bond_cut (=10 A) and sweep the skin; the reference value is the same force field with a
large skin (4 A). Measurement only. usage: exp_ghost_cutoff.py --lmp BIN --ffield-dir DIR --workdir DIR --out FILE"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner, geometries as g  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ("lmp", "ffield-dir", "workdir", "out"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    work = Path(a.workdir); work.mkdir(parents=True, exist_ok=True)
    base = Path(a.ffield_dir, "ffield.reax.cho").read_text().splitlines()
    assert "Upper Taper-radius" in base[14], base[14]
    rng = np.random.default_rng(23)
    pert = lambda at, amp: [{"el": q["el"], "xyz": [float(v + rng.uniform(-amp, amp)) for v in q["xyz"]]} for q in at]
    dia, diac = g.diamond(3.567, 2)
    wat = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                wat += g.translate(g.rotate(g.water(), rng.normal(size=3), float(rng.uniform(0, 360))), [(i + .5) * 4.8, (j + .5) * 4.8, (k + .5) * 4.8])
    gr, grc = g.graphene_sheet(3, 2)
    systems = {"diamond_2x2x2": (pert(dia, 0.07), diac, (True,) * 3), "water8_box9.6": (wat, {"lo": [0] * 3, "hi": [9.6] * 3}, (True,) * 3),
               "graphene_3x2": (pert(gr, 0.04), grc, (True, True, False))}
    out = {}
    for swb in (10.0, 8.0, 6.0, 5.0):
        L = list(base); L[14] = f"{swb:10.4f} !Upper Taper-radius (swb)"
        name = f"ffield.reax.cho_swb{swb:g}"; (work / name).write_text("\n".join(L) + "\n")
        for sname, (at, cell, per) in systems.items():
            ref = None; rows = []
            for skin in (4.0, 2.0, 1.0, 0.5, 0.0):
                case = {"id": f"{sname}_swb{swb:g}_skin{skin:g}", "ffield": {"name": name}, "elements": ["C", "H", "O"], "cell": cell, "periodic": list(per), "atoms": at,
                        "charge": {"model": "qeq/reaxff", "swb": swb, "tolerance": 1e-12, "maxiter": 1000}, "neighbor": {"skin": skin},
                        "expected_warnings": [r"Total cutoff < 2\*bond cutoff"] if swb < 10.0 else []}
                r = runner.run_case(case, a.lmp, work, work / "runs" / case["id"], "inst", diag=True)
                if "energy" not in r: rows.append({"skin": skin, "valid": False, "reasons": r["invalid_reasons"]}); continue
                F = np.array(r["forces"]); E = np.array([r["energy"]["slots"][s] for s in runner.SLOT_NAMES]); q = np.array(r["charges"])
                if ref is None: ref = (E, F, q, r["energy"]["total_pe"])
                rows.append({"skin": skin, "shell_A": max(swb, 7.5, 5.0) + skin, "valid": r["valid"], "reasons": r["invalid_reasons"],
                             "dE_total_vs_skin4": r["energy"]["total_pe"] - ref[3], "max_dSlot_vs_skin4": float(np.abs(E - ref[0]).max()),
                             "max_dF_vs_skin4": float(np.abs(F - ref[1]).max()), "max_dq_vs_skin4": float(np.abs(q - ref[2]).max()),
                             "equalization_residual_eV": r.get("independent_eem", {}).get("equalization_residual_eV")})
            out[f"swb={swb:g} {sname}"] = rows
            print(f"== nonb_cut(swb)={swb:g}  {sname}")
            for r_ in rows:
                if "dE_total_vs_skin4" in r_:
                    print(f"   skin {r_['skin']:3.1f} shell {r_['shell_A']:5.1f} A: dE={r_['dE_total_vs_skin4']:+.2e} dSlot={r_['max_dSlot_vs_skin4']:.2e} dF={r_['max_dF_vs_skin4']:.2e} dq={r_['max_dq_vs_skin4']:.1e} valid={r_['valid']} {r_['reasons'][:1]}")
                else: print("  ", r_)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
