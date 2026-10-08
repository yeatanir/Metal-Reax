#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Finite-difference validation of the pinned-LAMMPS reference forces (kept separate from any GPU tolerance, NUMERICAL_POLICY 5.2).

Two comparisons, each by central differences of the LAMMPS energy:
  fixed-q   : charges frozen at their converged values (checkqeq no, charge model fixed). The reported force must equal
              -dE/dx at fixed q, independent of the charge model.
  relaxed-q : charges re-equilibrated at each displaced geometry (full QEq). Differs from the reported force by the
              Hellmann-Feynman defect of the pinned code (the QEq functional uses 14.4 eV*A, the Coulomb energy
              332.06371/23.02 = 14.4254 eV*A; ENGINE_SPEC Q-27), which this run QUANTIFIES.
usage: fd_check.py --lmp BIN --ffield-dir DIR --cases DIR --workdir DIR --out FILE [--ids a,b] [--h 1e-4] [--natoms-sample 8]"""
import argparse, copy, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner  # noqa: E402


def energy(case, lmp, ffdir, wd, fixed_q=None):
    c = copy.deepcopy(case)
    if fixed_q is not None:
        for a, q in zip(c["atoms"], fixed_q): a["q"] = float(q)
        c["charge"] = {"model": "fixed"}; c["pair"] = dict(c.get("pair", {}), checkqeq="no")
    r = runner.run_case(c, lmp, ffdir, wd, "fd", diag=False)
    return r


def work(args):
    case_path, lmp, ffdir, wd, h, nsample = args
    case = json.loads(Path(case_path).read_text())
    cid = case["id"]; wd = Path(wd) / cid
    base = energy(case, lmp, ffdir, wd / "base")
    if "energy" not in base: return cid, {"error": base["invalid_reasons"]}
    q0 = np.array(base["charges"]); F0 = np.array(base["forces"]); n = len(q0)
    fq = energy(case, lmp, ffdir, wd / "fixedq0", fixed_q=q0)
    rng = np.random.default_rng(5)
    idx = list(range(n)) if n <= nsample else sorted(rng.choice(n, nsample, replace=False).tolist())
    out = {"natoms": n, "h": h, "atoms_checked": idx, "force_fixedq_vs_relaxed_run_maxdiff": float(np.abs(np.array(fq["forces"]) - F0).max()),
           "E_fixedq_vs_relaxed": float(fq["energy"]["total_pe"] - base["energy"]["total_pe"])}
    # fixed-q central differences at two step sizes (consistency test exposes energy discontinuities within +-h),
    # relaxed-q at the smaller step only
    H = (h, 2 * h)
    Ffd = {hh: np.zeros((len(idx), 3)) for hh in H}; Ffd_r = np.zeros((len(idx), 3))
    for a_, i in enumerate(idx):
        for d in range(3):
            for hh in H:
                e = []
                for s_ in (+1, -1):
                    c = copy.deepcopy(case); c["atoms"][i]["xyz"][d] += s_ * hh
                    e.append(energy(c, lmp, ffdir, wd / f"f_{i}_{d}_{s_}_{hh:g}", fixed_q=q0)["energy"]["total_pe"])
                Ffd[hh][a_, d] = -(e[0] - e[1]) / (2 * hh)
            e = []
            for s_ in (+1, -1):
                c = copy.deepcopy(case); c["atoms"][i]["xyz"][d] += s_ * h
                rr = energy(c, lmp, ffdir, wd / f"r_{i}_{d}_{s_}", None)
                e.append(rr["energy"]["total_pe"] if "energy" in rr else float("nan"))
            Ffd_r[a_, d] = -(e[0] - e[1]) / (2 * h)
    Fl = F0[idx]
    cons = np.abs(Ffd[H[0]] - Ffd[H[1]])
    suspect = cons > (1e-5 + 1e-6 * np.abs(Fl))        # step-size inconsistency => not a smooth point at this scale
    ok = ~suspect
    out["h"] = list(H)
    out["fixedq"] = {"max_abs_err_h": float(np.abs(Ffd[H[0]] - Fl).max()),
                     "max_abs_err_consistent_components": float(np.abs(Ffd[H[0]] - Fl)[ok].max()) if ok.any() else None,
                     "n_components": int(ok.size), "n_discontinuity_suspect": int(suspect.sum()),
                     "max_step_inconsistency": float(cons.max()),
                     "rms_err": float(np.sqrt(((Ffd[H[0]] - Fl) ** 2).mean())),
                     "F_rms_checked": float(np.sqrt((Fl ** 2).mean())), "F_max_checked": float(np.abs(Fl).max())}
    out["relaxed"] = {"max_abs_err": float(np.nanmax(np.abs(Ffd_r - Fl))), "rms_err": float(np.sqrt(np.nanmean((Ffd_r - Fl) ** 2)))}
    return cid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--ffield-dir", required=True); ap.add_argument("--cases", required=True)
    ap.add_argument("--workdir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ids", default="")
    ap.add_argument("--h", type=float, default=1e-5); ap.add_argument("--natoms-sample", type=int, default=6); ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    files = sorted(Path(a.cases).glob("*.json"))
    if a.ids: files = [f for f in files if f.stem in set(a.ids.split(","))]
    res = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for cid, o in ex.map(work, [(str(f), a.lmp, a.ffield_dir, a.workdir, a.h, a.natoms_sample) for f in files]):
            res[cid] = o
            if "error" in o: print(f"{cid:28s} ERROR {o['error']}"); continue
            fx = o["fixedq"]
            print(f"{cid:28s} N={o['natoms']:3d} fixed-q max|dF|={fx['max_abs_err_h']:.2e} (smooth comps {fx['max_abs_err_consistent_components']}) suspect={fx['n_discontinuity_suspect']}/{fx['n_components']} (F rms {fx['F_rms_checked']:.1f})  relaxed-q max|dF|={o['relaxed']['max_abs_err']:.2e}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
