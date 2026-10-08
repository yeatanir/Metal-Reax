#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Confirm two reference quirks by modifying the reference source in THROWAWAY experimental builds (never used for fixtures):
  Q-32  EXP-0001 replaces the H-bond donor/acceptor tag comparison by an atom-identity comparison (`i != k`):
        supercell invariance of one-molecule periodic cells must then be restored.
  Q-34  EXP-0002 replaces `dDelta_lp[j]` by `dDelta_lp_temp[j]` in the over/under-coordination force loop:
        the finite-difference error of SO2 (FC force field) must then drop from O(10) to the FD floor.
usage: exp_q32_q34.py --stock BIN --exp1 BIN --exp2 BIN --ffield-dir DIR --cases DIR --workdir DIR --out32 FILE --out34 FILE"""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner, geometries as g, exp_periodic as ep  # noqa: E402


def fd_err(case, lmp, ffdir, wd, h=1e-5):
    b = runner.run_case(case, lmp, ffdir, Path(wd) / "base", "x", diag=False)
    q = np.array(b["charges"]); F = np.array(b["forces"])
    def E(c, t):
        c = copy.deepcopy(c)
        for a_, qq in zip(c["atoms"], q): a_["q"] = float(qq)
        c["charge"] = {"model": "fixed"}; c["pair"] = {"checkqeq": "no"}
        return runner.run_case(c, lmp, ffdir, Path(wd) / t, "x", diag=False)["energy"]["total_pe"]
    Ffd = np.zeros_like(F)
    for i in range(len(q)):
        for d in range(3):
            e = []
            for s in (1, -1):
                c = copy.deepcopy(case); c["atoms"][i]["xyz"][d] += s * h; e.append(E(c, f"{i}{d}{s}"))
            Ffd[i, d] = -(e[0] - e[1]) / (2 * h)
    return float(np.abs(F - Ffd).max()), float(np.sqrt((F ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    for k in ("stock", "exp1", "exp2", "ffield-dir", "cases", "workdir", "out32", "out34"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    # ---- Q-32
    rng = np.random.default_rng(7)
    def pert(at, amp): return [{"el": q["el"], "xyz": [float(v + rng.uniform(-amp, amp)) for v in q["xyz"]]} for q in at]
    systems = {"water_1mol_cell6.2": (pert(g.translate(g.water(), [3, 3, 3]), 0.05), {"lo": [0] * 3, "hi": [6.2] * 3}),
               "water_dimer_cell7.0": (pert(g.translate(g.water_dimer(), [3, 3, 3]), 0.05), {"lo": [0] * 3, "hi": [7.0] * 3}),
               "water_dimer_cell9.0": (pert(g.translate(g.water_dimer(), [4.5, 4.5, 4.5]), 0.05), {"lo": [0] * 3, "hi": [9.0] * 3})}
    out = {"systems": {}}
    class A: pass
    for label, lmp in (("stock_with_patch0001", a.stock), ("EXP-0001_identity_comparison", a.exp1)):
        out["systems"][label] = {}
        for name, (at, cell) in systems.items():
            x = A(); x.lmp = lmp; x.ffield_dir = a.ffield_dir; x.workdir = str(Path(a.workdir) / "q32" / label); x.label = label
            rows = ep.supercell_experiment(x, name, "cho", ["C", "H", "O"], at, cell, (True,) * 3, [(1, 1, 1), (2, 2, 2)], tol=1e-11)
            out["systems"][label][name] = {"ehb_per_cell": [r["slots_per_cell"]["ehb"] for r in rows], "total_dev_2x2x2_vs_1x1x1": rows[1]["total_dev_vs_first_per_cell"]}
    st = out["systems"]["stock_with_patch0001"]; ex = out["systems"]["EXP-0001_identity_comparison"]
    out["confirmed"] = bool(st["water_1mol_cell6.2"]["total_dev_2x2x2_vs_1x1x1"] > 0.1 and st["water_dimer_cell7.0"]["total_dev_2x2x2_vs_1x1x1"] > 0.1
                            and st["water_dimer_cell9.0"]["total_dev_2x2x2_vs_1x1x1"] < 1e-8
                            and all(v["total_dev_2x2x2_vs_1x1x1"] < 1e-8 for v in ex.values()))
    Path(a.out32).write_text(json.dumps(out, indent=1))
    print("Q-32 confirmed:", out["confirmed"], json.dumps({k: {n: round(v["total_dev_2x2x2_vs_1x1x1"], 12) for n, v in d.items()} for k, d in out["systems"].items()}))
    # ---- Q-34 : evaluated on every SO2 fixture (the defect needs a heavy atom with pi bonds AND a non-zero lone-pair derivative on it)
    r34 = {}
    for cname in ("mattsson_so2", "fc_so2"):
        case = json.loads((Path(a.cases) / (cname + ".json")).read_text())
        r34[cname] = {}
        for label, lmp in (("stock_with_patch0001", a.stock), ("EXP-0002_dDelta_lp_temp", a.exp2)):
            err, frms = fd_err(case, lmp, a.ffield_dir, Path(a.workdir) / "q34" / cname / label)
            r34[cname][label] = {"max_abs_fd_error": err, "F_rms": frms}
    m = r34["mattsson_so2"]
    r34["confirmed"] = bool(m["stock_with_patch0001"]["max_abs_fd_error"] > 1.0 and m["EXP-0002_dDelta_lp_temp"]["max_abs_fd_error"] < 1e-4)
    Path(a.out34).write_text(json.dumps(r34, indent=1))
    print("Q-34 confirmed:", r34["confirmed"], json.dumps(r34))


if __name__ == "__main__":
    main()
