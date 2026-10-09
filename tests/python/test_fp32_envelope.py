#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""FP32-1: the CPU-32 twin (the CPU engine compiled with float) against CPU-64 on the 58 fixtures = the single-precision error envelope eta_32
(NUMERICAL_POLICY 5.3), and Metal-32 against the same envelope.  Per fixture: total energy per atom, force max component and RMS of the (frozen
charge) total gradient. Reported over all fixtures and over the well-conditioned ones.
Gates:  (1) the twin itself satisfies the owner-set C3 criteria on the well-conditioned fixtures (otherwise FP32 cannot meet C3 and that is a
feasibility finding, not something to relax);  (2) with --metal: Metal is not an anomaly, i.e. its error stays within 3x the twin's envelope
(NUMERICAL_POLICY 5.3, diagnostic threshold) on the well-conditioned fixtures, per quantity, with a floor of 1e-9 so that exactly-zero twin errors do not
divide by zero.   usage: test_fp32_envelope.py --tool BIN --ffield-dir DIR [--metal]"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "reaxref")); sys.path.insert(0, str(HERE))
from test_full_fixtures import case_text, TERMS  # noqa: E402
from test_gpu_fixtures import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--ffield-dir", required=True); ap.add_argument("--metal", action="store_true")
    ap.add_argument("--fixtures", default=str(ROOT / "tests" / "fixtures"))
    a = ap.parse_args()
    fx = Path(a.fixtures)
    c3 = json.loads((ROOT / "tolerances" / "tolerances.json").read_text())["C3"]
    e_tol, f_max, f_rms = c3["energy_per_atom_abs_kcal_mol"], c3["force_max_component_kcal_mol_A"], c3["force_rms_kcal_mol_A"]
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for cf in sorted((fx / "cases").glob("*.json")):
            case = json.loads(cf.read_text()); ref = json.loads((fx / "reference" / cf.name).read_text())
            if not ref.get("valid", True): continue
            txt = Path(td) / "case.txt"; txt.write_text(case_text(case, ref))
            extra = ["--elements", ",".join(case["elements"])] + (["--lgvdw"] if case.get("pair", {}).get("lgvdw") else [])
            ff = Path(a.ffield_dir) / case["ffield"]["name"]
            nat = len(ref["positions"])
            res = {}
            for b in ("cpu64", "cpu32") + (("metal",) if a.metal else ()):
                try: res[b] = run(a.tool, ff, extra, txt, b)
                except RuntimeError as e: print("FAIL", cf.stem, b, e); return 1
            def errs(b):
                e0, g0 = res["cpu64"]; e1, g1 = res[b]
                de = abs(sum(e1[k] for k in TERMS) - sum(e0[k] for k in TERMS)) / nat
                d = np.array([np.array(g1[t]) - np.array(g0[t]) for t in sorted(g0)])
                return de, np.abs(d).max(), np.sqrt((d ** 2).mean())
            rows.append((cf.stem, ref["conditioning"].get("well_conditioned_extended", True), errs("cpu32"), errs("metal") if a.metal else None))
    def env(sel, idx, which):
        v = [r[which][idx] for r in rows if sel(r)]
        return max(v) if v else 0.0
    names = ("E/atom", "F max", "F rms")
    fails = []
    for label, sel in (("all fixtures", lambda r: True), ("well conditioned", lambda r: r[1])):
        line = f"{label:17s} ({sum(1 for r in rows if sel(r)):2d}) CPU-32 envelope: " + "  ".join(f"{n} {env(sel, i, 2):.2e}" for i, n in enumerate(names))
        if a.metal: line += "   | Metal: " + "  ".join(f"{n} {env(sel, i, 3):.2e}" for i, n in enumerate(names))
        print(line)
    wc = lambda r: r[1]
    for (i, n), lim in zip(enumerate(names), (e_tol, f_max, f_rms)):
        if env(wc, i, 2) > lim: fails.append(f"CPU-32 twin {n} {env(wc, i, 2):.2e} exceeds C3 {lim} on well-conditioned fixtures (FP32 cannot meet C3)")
        if a.metal and env(wc, i, 3) > 3 * max(env(wc, i, 2), 1e-9): fails.append(f"Metal {n} {env(wc, i, 3):.2e} > 3x the twin's envelope {env(wc, i, 2):.2e} (anomaly to be explained)")
    for f in fails: print("FAIL", f)
    print(f"FP32-1: {len(rows)} fixtures, {len(fails)} failures")
    print("RESULT: PASS" if not fails else "RESULT: FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
