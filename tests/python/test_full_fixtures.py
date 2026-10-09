#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""FULL-1 (CPU-64, frozen charges): all 13 energy terms, the total energy and the total forces of reaxmetal_energy_tool vs the pinned-LAMMPS
values of M1, under the frozen C1 thresholds (tolerances/tolerances.json: energy_slot_rel_to_max1 per term, force component and RMS on the
fixtures classified well conditioned (primary = with the proposed X1/X2 classes, ADR-020). The other fixtures are only reported.
usage: test_full_fixtures.py --tool BIN --ffield-dir DIR"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "reaxref")); sys.path.insert(0, str(HERE))
from test_neighbor_fixtures import case_text as _case_text  # noqa: E402

TERMS = ["e_bond", "e_lp", "e_ov", "e_un", "e_ang", "e_pen", "e_coa", "e_tor", "e_con", "e_hb", "e_vdW", "e_ele", "e_pol"]


def case_text(case, ref):
    return _case_text(case, ref) + "charges\n" + " ".join(repr(float(v)) for v in ref["charges"]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--fixtures", default=str(ROOT / "tests" / "fixtures"))
    a = ap.parse_args()
    fx = Path(a.fixtures)
    tol = json.loads((ROOT / "tolerances" / "tolerances.json").read_text())["C1"]["primary"]["threshold"]
    rtol, fcomp, frms, eatom = tol["energy_slot_rel_to_max1"], tol["force_component_abs"], tol["force_rms"], tol["energy_total_per_atom_abs"]
    fails, n, nwc, worst = [], 0, 0, {"e": 0.0, "etot_atom": 0.0, "fcomp": 0.0, "frms": 0.0}
    skipped = {}
    with tempfile.TemporaryDirectory() as td:
        for cf in sorted((fx / "cases").glob("*.json")):
            case = json.loads(cf.read_text()); ref = json.loads((fx / "reference" / cf.name).read_text())
            if not ref.get("valid", True):
                continue
            txt = Path(td) / "case.txt"; txt.write_text(case_text(case, ref))
            extra = ["--elements", ",".join(case["elements"])] + (["--lgvdw"] if case.get("pair", {}).get("lgvdw") else [])
            r = subprocess.run([a.tool, "--ffield", str(Path(a.ffield_dir) / case["ffield"]["name"]), *extra, "--grad", str(txt)], capture_output=True, text=True)
            if r.returncode != 0:
                fails.append(f"{cf.stem}: tool failed: {r.stderr.strip()}"); continue
            e, g = {}, {}
            for ln in r.stdout.splitlines():
                k, *v = ln.split()
                if k == "grad": g[int(v[0])] = [float(t) for t in v[1:]]
                else: e[k] = float(v[0])
            n += 1
            nat = len(ref["positions"])
            for k in TERMS:
                want = ref["energy_data_fields"][k]; d = abs(e[k] - want) / max(1.0, abs(want))
                worst["e"] = max(worst["e"], d)
                if d > rtol: fails.append(f"{cf.stem}: {k} engine {e[k]!r} reference {want!r}")
            tot = sum(e[k] for k in TERMS); dtot = abs(tot - ref["energy"]["slot_sum"]) / nat
            worst["etot_atom"] = max(worst["etot_atom"], dtot)
            if dtot > eatom: fails.append(f"{cf.stem}: total energy per atom differs by {dtot:.3e}")
            cond = ref["conditioning"]
            wc = cond.get("well_conditioned_extended", cond.get("well_conditioned_pre_registered", True))
            dif = [-g[i + 1][c] - ref["forces"][i][c] for i in range(nat) for c in range(3)]
            fm = max(abs(x) for x in dif); fr = (sum(x * x for x in dif) / len(dif)) ** 0.5
            if wc:
                nwc += 1
                worst["fcomp"] = max(worst["fcomp"], fm); worst["frms"] = max(worst["frms"], fr)
                if fm > fcomp or fr > frms:
                    fails.append(f"{cf.stem}: force max {fm:.3e} (limit {fcomp:g}) rms {fr:.3e} (limit {frms:g})")
            else:
                skipped[cf.stem] = (fm, fr)
    for f in fails[:60]: print("FAIL", f)
    print("not well conditioned (reported only): " + ", ".join(f"{k} max {v[0]:.1e}" for k, v in skipped.items()))
    print(f"worst: slot rel {worst['e']:.2e} (limit {rtol:g}); total E/atom {worst['etot_atom']:.2e} (limit {eatom:g}); force comp {worst['fcomp']:.2e} (limit {fcomp:g}), rms {worst['frms']:.2e} (limit {frms:g})")
    print(f"FULL-1: {n} fixtures, {nwc} well conditioned, {len(fails)} failures")
    ok = not fails and n >= 50
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
