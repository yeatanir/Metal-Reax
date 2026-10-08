#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""M4 phase A (CPU-64): bond / lone-pair / over / under energies of reaxmetal_energy_tool vs the pinned-LAMMPS values recorded in M1
(tests/fixtures/reference/*.json, energy_data_fields), threshold = frozen C1 energy_slot_rel_to_max1 (|d| <= rtol*max(1,|ref|)).
With --fd the tool's gradient of those four terms is also checked against a central finite difference of their energy sum
(h = 1e-5 A, fixtures of <= --fd-max-atoms atoms; tolerance 5e-5 kcal/mol/A + 1e-6 relative; the M1 FD of LAMMPS itself reaches 1.9e-5).
usage: test_bonded_fixtures.py --tool BIN --ffield-dir DIR [--fd] [--only NAME]"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "reaxref"))
sys.path.insert(0, str(HERE))
import runner  # noqa: E402
from test_neighbor_fixtures import case_text  # noqa: E402

# Reference defect reproduced by default (ADR-021, ENGINE_SPEC Q-34): analytic force != gradient of the reported energy for heavy atoms with pi
# bonds. M1 measured 16.5 kcal/mol/A on this fixture against FD of LAMMPS itself. The test requires the deviation to be PRESENT (a silent "fix" would
# break parity) and of that size.
KNOWN_FD_QUIRK = {"mattsson_so2": (10.0, 25.0)}
TERMS = {"e_bond": "e_bond", "e_lp": "e_lp", "e_ov": "e_ov", "e_un": "e_un"}


def run_tool(tool, ffield, extra, txt, grad=False):
    r = subprocess.run([tool, "--ffield", str(ffield), *extra, *(["--grad"] if grad else []), str(txt)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    out, g = {}, {}
    for ln in r.stdout.splitlines():
        k, *v = ln.split()
        if k == "grad":
            g[int(v[0])] = [float(t) for t in v[1:]]
        else:
            out[k] = float(v[0])
    return out, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--fixtures", default=str(ROOT / "tests" / "fixtures"))
    ap.add_argument("--fd", action="store_true"); ap.add_argument("--fd-max-atoms", type=int, default=24)
    ap.add_argument("--only")
    a = ap.parse_args()
    fx = Path(a.fixtures)
    rtol = json.loads((ROOT / "tolerances" / "tolerances.json").read_text())["C1"]["primary"]["threshold"]["energy_slot_rel_to_max1"]
    fails, n, nfd, worst = [], 0, 0, {}
    with tempfile.TemporaryDirectory() as td:
        for cf in sorted((fx / "cases").glob("*.json")):
            if a.only and a.only != cf.stem:
                continue
            case = json.loads(cf.read_text()); ref = json.loads((fx / "reference" / cf.name).read_text())
            if not ref.get("valid", True):
                continue
            txt = Path(td) / "case.txt"; txt.write_text(case_text(case, ref))
            extra = ["--elements", ",".join(case["elements"])] + (["--lgvdw"] if case.get("pair", {}).get("lgvdw") else [])
            ff = Path(a.ffield_dir) / case["ffield"]["name"]
            try:
                got, g = run_tool(a.tool, ff, extra, txt, grad=a.fd)
            except RuntimeError as e:
                fails.append(f"{cf.stem}: tool failed: {e}"); continue
            n += 1
            for k, rk in TERMS.items():
                want = ref["energy_data_fields"][rk]
                d = abs(got[k] - want); tol = rtol * max(1.0, abs(want))
                worst[k] = max(worst.get(k, 0.0), d / max(1.0, abs(want)))
                if not d <= tol:
                    fails.append(f"{cf.stem}: {k} engine {got[k]!r} reference {want!r} |d|={d:.3e} > {tol:.3e}")
            if a.fd and len(ref["positions"]) <= a.fd_max_atoms:
                nfd += 1
                tot = sum(got[k] for k in TERMS)
                h = 1e-5
                pos = [list(p) for p in ref["positions"]]
                maxd = 0.0
                for i in range(len(pos)):
                    for c in range(3):
                        e = []
                        for sgn in (+1, -1):
                            q = [list(p) for p in pos]; q[i][c] += sgn * h
                            r2 = dict(ref); r2["positions"] = q
                            txt.write_text(case_text(case, r2))
                            o, _ = run_tool(a.tool, ff, extra, txt)
                            e.append(sum(o[k] for k in TERMS))
                        fd = (e[0] - e[1]) / (2 * h)
                        an = g[i + 1][c]
                        maxd = max(maxd, abs(fd - an))
                        if cf.stem not in KNOWN_FD_QUIRK and abs(fd - an) > 5e-5 + 1e-6 * abs(an):
                            fails.append(f"{cf.stem}: FD atom {i + 1} comp {c}: analytic {an!r} FD {fd!r}")
                if cf.stem in KNOWN_FD_QUIRK:
                    lo, hi = KNOWN_FD_QUIRK[cf.stem]
                    if not lo <= maxd <= hi:
                        fails.append(f"{cf.stem}: expected the reproduced reference defect (max |analytic-FD| in [{lo},{hi}]), got {maxd:.3e}")
                    print(f"note: {cf.stem}: max |analytic-FD| = {maxd:.3f} (reproduced reference defect, Q-34)")
                else:
                    worst["fd_abs"] = max(worst.get("fd_abs", 0.0), maxd)
    for f in fails[:60]:
        print("FAIL", f)
    print("worst relative energy differences:", {k: f"{v:.2e}" for k, v in worst.items()})
    print(f"BOND-1: {n} fixtures compared, {nfd} with finite-difference check, {len(fails)} failures")
    ok = not fails and n >= (1 if a.only else 50)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
