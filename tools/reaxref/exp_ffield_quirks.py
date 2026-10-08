#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""M1 experiments Q-09 (three-body table duplication / j==l zero slot / >5 sets) and Q-12 (absent bond-pair parameters).
Measurement only. usage: exp_ffield_quirks.py --lmp BIN --ffield-dir DIR --workdir DIR --out FILE"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner, geometries as g, ffield_edit as fe  # noqa: E402

CHO = ["C", "H", "O"]
S2 = (90.0, 8.0, 1.5, 0.0, 1.2, 0.0, 1.5)       # arbitrary second/third parameter sets (distinct, finite, p_val1 > 0.001)
S3 = (105.0, 12.0, 2.5, 0.0, 0.8, 0.0, 2.0)


def case_for(cid, ffname, at, periodic=False):
    return {"id": cid, "ffield": {"name": ffname}, "elements": CHO, "cell": g.box_around(at, 15.0), "periodic": [False] * 3,
            "atoms": at, "charge": {"model": "qeq/reaxff", "swb": 10.0, "tolerance": 1e-12, "maxiter": 500}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--workdir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = Path(a.workdir); work.mkdir(parents=True, exist_ok=True)
    base = Path(a.ffield_dir, "ffield.reax.cho").read_text()
    L, k0, angles = fe.angle_block(base)
    find = lambda trip: next(l for l in angles if tuple(int(x) for x in l.split()[:3]) == trip)
    out = {}

    def variant(name, angle_lines):
        (work / f"ffield.reax.{name}").write_text(fe.set_angle_lines(base, angle_lines))
        return f"ffield.reax.{name}"

    def run(cid, ff, at):
        r = runner.run_case(case_for(cid, ff, at), a.lmp, work, work / "runs" / cid, "inst", diag=True)
        return r

    def ang_slots(r):
        s = r["energy"]["slots"]; return np.array([s["ev"], s["epen"], s["ecoa"]]), np.array([s[x] for x in runner.SLOT_NAMES])

    s1_232 = find((2, 3, 2)); w1 = [float(x) for x in s1_232.split()[3:]]
    s1_132 = find((1, 3, 2))
    others = [l for l in angles if l not in (s1_232, s1_132)]
    water = g.rotate(g.water(), (0.3, 1, 0.2), 17.0)
    meoh = g.methanol()

    # ---------------- Q-09a: j == l triple (H,O,H = 2,3,2): file sets -> LAMMPS slots
    ff_none = variant("q09_232_none", others + [s1_132])
    ff_s1 = "ffield.reax.cho"
    ff_s2 = variant("q09_232_S2only", others + [s1_132, fe.fmt_angle(2, 3, 2, *S2)])
    ff_s1s2 = variant("q09_232_S1_S2", others + [s1_132, s1_232, fe.fmt_angle(2, 3, 2, *S2)])
    ff_s1s2s3 = variant("q09_232_S1_S2_S3", others + [s1_132, s1_232, fe.fmt_angle(2, 3, 2, *S2), fe.fmt_angle(2, 3, 2, *S3)])
    ff_s3only = variant("q09_232_S3only", others + [s1_132, fe.fmt_angle(2, 3, 2, *S3)])
    res = {}
    for nm, ff, d in (("none", ff_none, work), ("S1", ff_s1, Path(a.ffield_dir)), ("S2", ff_s2, work), ("S3", ff_s3only, work), ("S1+S2", ff_s1s2, work), ("S1+S2+S3", ff_s1s2s3, work)):
        r = runner.run_case(case_for("q09a_" + nm, ff, water), a.lmp, d, work / "runs" / ("q09a_" + nm), "inst", diag=True)
        pt = runner.read_params_dump(Path(r["diag"]["file"]).parent / "params.txt") if "diag" in r else None
        h = pt["H3"].get((1, 2, 1)) if pt else None     # zero-based (H=1,O=2,H=1)
        res[nm] = {"valid": r["valid"], "reasons": r["invalid_reasons"], "angle_slots_ev_epen_ecoa": ang_slots(r)[0].tolist() if "energy" in r else None,
                   "raw_cnt": h[0] if h else None, "tallies_angle": {k: v["count"] for k, v in r.get("diag", {}).get("tallies", {}).items() if k.startswith(("angle", "penalty", "coalition"))},
                   "all_slots": ang_slots(r)[1].tolist() if "energy" in r else None, "stored_prm_p_val1": [float(h[1 + 7 * c + 1]) for c in range(5)] if h else None}
    pred = lambda a_: (np.array(res[a_[0]]["angle_slots_ev_epen_ecoa"]) + np.array(res[a_[1]]["angle_slots_ev_epen_ecoa"]) - np.array(res["none"]["angle_slots_ev_epen_ecoa"]))
    res["additivity_S1+S2"] = {"predicted": pred(("S1", "S2")).tolist(), "observed": res["S1+S2"]["angle_slots_ev_epen_ecoa"],
                               "max_abs_diff": float(np.abs(pred(("S1", "S2")) - np.array(res["S1+S2"]["angle_slots_ev_epen_ecoa"])).max())}
    p3 = pred(("S1", "S2")) + np.array(res["S3"]["angle_slots_ev_epen_ecoa"]) - np.array(res["none"]["angle_slots_ev_epen_ecoa"])
    res["additivity_S1+S2+S3"] = {"predicted": p3.tolist(), "observed": res["S1+S2+S3"]["angle_slots_ev_epen_ecoa"],
                                  "max_abs_diff": float(np.abs(p3 - np.array(res["S1+S2+S3"]["angle_slots_ev_epen_ecoa"])).max())}
    out["Q09a_j_eq_l"] = res

    # ---------------- Q-09b: j != l (C,O,H = 1,3,2): duplicates and explicit mirror lines
    s2_132 = fe.fmt_angle(1, 3, 2, *S2)
    mir_s2 = fe.fmt_angle(2, 3, 1, *S2)
    cases = {"base": (None, "ffield.reax.cho"), "none": (None, None), "S2only": None}
    f_none = variant("q09_132_none", [l for l in angles if l != s1_132])
    f_s2 = variant("q09_132_S2only", [l for l in angles if l != s1_132] + [s2_132])
    f_dup = variant("q09_132_S1_S2", [l for l in angles if l != s1_132] + [s1_132, s2_132])
    f_mir = variant("q09_132_S1_mirrorS2", [l for l in angles if l != s1_132] + [s1_132, mir_s2])
    res_b = {}
    for nm, ff, d in (("none", f_none, work), ("S1", "ffield.reax.cho", Path(a.ffield_dir)), ("S2", f_s2, work), ("S1+S2 (same order)", f_dup, work), ("S1 + S2 as mirror line (2,3,1)", f_mir, work)):
        r = runner.run_case(case_for("q09b_" + nm.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("+", "p"), ff, meoh), a.lmp, d, work / "runs" / ("q09b_" + nm.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("+", "p")), "inst", diag=True)
        res_b[nm] = {"valid": r["valid"], "reasons": r["invalid_reasons"], "angle_slots": ang_slots(r)[0].tolist() if "energy" in r else None}
    pb = np.array(res_b["S1"]["angle_slots"]) + np.array(res_b["S2"]["angle_slots"]) - np.array(res_b["none"]["angle_slots"])
    for nm in ("S1+S2 (same order)", "S1 + S2 as mirror line (2,3,1)"):
        res_b[nm]["predicted_additive"] = pb.tolist()
        res_b[nm]["max_abs_diff_vs_additive"] = float(np.abs(pb - np.array(res_b[nm]["angle_slots"])).max())
    out["Q09b_j_ne_l"] = res_b

    # ---------------- Q-12: absent bond-pair block: phantom bond order at every distance <= bond_cut
    res12 = {}
    for label, pair, els in (("H-O", (2, 3), ("H", "O")), ("C-O", (1, 3), ("C", "O")), ("C-C", (1, 1), ("C", "C"))):
        ffm = work / f"ffield.reax.q12_no{label.replace('-', '')}"
        ffm.write_text(fe.drop_bond_pair(base, *pair))
        for nm, ff, d in ((label + " present", "ffield.reax.cho", Path(a.ffield_dir)), (label + " block removed", ffm.name, work)):
            rows = []
            for dist in (1.0, 2.0, 3.0, 4.0, 4.9):
                at = [{"el": els[0], "xyz": [0.0, 0.0, 0.0]}, {"el": els[1], "xyz": [0.0, 0.0, dist]}]
                tag = nm.replace(" ", "_")
                r = runner.run_case(case_for(f"q12_{tag}_{dist}", ff, at), a.lmp, d, work / "runs" / f"q12_{tag}_{dist}", "inst", diag=True)
                row = {"r": dist, "valid": r["valid"], "reasons": r["invalid_reasons"][:1]}
                if "diag" in r:
                    D = runner.read_call_dump(r["diag"]["file"])
                    bl = [x for x in D["bonds"] if x[0] == "0" and x[1] == "1"]
                    row["n_bond_entries"] = len(D["bonds"]); row["e_bond"] = r["diag"]["energy_fields"]["e_bond"]
                    if bl:
                        x = bl[0]
                        row.update({"BOp_s": float(x[5]), "BOp_pi": float(x[6]), "BOp_pipi": float(x[7]), "BO": float(x[8])})
                rows.append(row)
            res12[nm] = rows
    out["Q12_absent_bond_pair"] = res12
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
