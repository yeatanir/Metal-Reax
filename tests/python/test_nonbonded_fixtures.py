#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""M5 (CPU-64): vdW / Coulomb / polarization energies (reference charges as input) of reaxmetal_energy_tool vs the pinned-LAMMPS values recorded in M1
(tests/fixtures/reference/*.json, energy_data_fields), threshold = frozen C1 energy_slot_rel_to_max1 (|d| <= rtol*max(1,|ref|)).
With --fd the tool's gradient of those terms is also checked against a central finite difference of their energy sum
(h = 1e-5 A, fixed charges, fixtures of <= --fd-max-atoms atoms; tolerance 5e-5 kcal/mol/A + 1e-6 relative).
usage: test_bonded_fixtures.py --tool BIN --ffield-dir DIR [--fd] [--only NAME]"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "reaxref"))
sys.path.insert(0, str(HERE))
import runner  # noqa: E402
from test_neighbor_fixtures import case_text as _case_text  # noqa: E402


def case_text(case, ref):
    return _case_text(case, ref) + "charges\n" + " ".join(repr(float(v)) for v in ref["charges"]) + "\n"

KNOWN_FD_QUIRK = {}
TERMS = {"e_vdW": "e_vdW", "e_ele": "e_ele", "e_pol": "e_pol"}


def run_tool(tool, ffield, extra, txt, grad=False, backend="cpu64"):
    r = subprocess.run([tool, "--ffield", str(ffield), *extra, "--backend", backend, *(["--grad", "--grad-terms", "nonbonded"] if grad else []), str(txt)], capture_output=True, text=True)
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


def main_metal(a, fx):
    """GPU-1 (nonbonded subset): Metal-32 vs CPU-64 and vs the pinned-LAMMPS reference values, against the owner-set C3 criteria of
    tolerances/tolerances.json (energy per atom, force max component and RMS). Also FORCE-2: two launches give identical output."""
    c3 = json.loads((ROOT / "tolerances" / "tolerances.json").read_text())["C3"]
    e_tol, f_max, f_rms = c3["energy_per_atom_abs_kcal_mol"], c3["force_max_component_kcal_mol_A"], c3["force_rms_kcal_mol_A"]
    fails, n, w = [], 0, {"e_atom": 0.0, "e_rel": 0.0, "f_max": 0.0, "f_rms": 0.0}
    gpu_ms = []
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
                g64_e, g64 = run_tool(a.tool, ff, extra, txt, grad=True)
                gm_e, gm = run_tool(a.tool, ff, extra, txt, grad=True, backend="metal")
                gm2_e, gm2 = run_tool(a.tool, ff, extra, txt, grad=True, backend="metal")
            except RuntimeError as e:
                fails.append(f"{cf.stem}: tool failed: {e}"); continue
            n += 1
            gpu_ms.append(gm_e.get("gpu_ms", 0.0))
            natoms = len(ref["positions"])
            if gm != gm2 or any(gm_e[k] != gm2_e[k] for k in TERMS):
                fails.append(f"{cf.stem}: FORCE-2 two Metal launches differ")
            for k, rk in TERMS.items():
                want = ref["energy_data_fields"][rk]
                d = abs(gm_e[k] - want)
                w["e_atom"] = max(w["e_atom"], d / natoms); w["e_rel"] = max(w["e_rel"], d / max(1.0, abs(want)))
                if d / natoms > e_tol:
                    fails.append(f"{cf.stem}: {k} Metal {gm_e[k]!r} reference {want!r} |d|/N={d / natoms:.3e} > {e_tol}")
            dif = [gm[t][c] - g64[t][c] for t in g64 for c in range(3)]
            fm = max(abs(x) for x in dif); fr = (sum(x * x for x in dif) / len(dif)) ** 0.5
            w["f_max"] = max(w["f_max"], fm); w["f_rms"] = max(w["f_rms"], fr)
            if fm > f_max or fr > f_rms:
                fails.append(f"{cf.stem}: gradient Metal vs CPU-64 max {fm:.3e} (limit {f_max}) rms {fr:.3e} (limit {f_rms})")
    for f in fails[:60]:
        print("FAIL", f)
    print(f"worst: energy/atom vs LAMMPS {w['e_atom']:.2e} (C3 {e_tol}), relative {w['e_rel']:.2e}; gradient Metal-32 vs CPU-64 max {w['f_max']:.2e} (C3 {f_max}) rms {w['f_rms']:.2e} (C3 {f_rms})")
    if gpu_ms:
        print(f"GPU time per evaluation: median {sorted(gpu_ms)[len(gpu_ms) // 2]:.3f} ms, max {max(gpu_ms):.3f} ms")
    print(f"GPU-1 nonbonded: {n} fixtures, {len(fails)} failures")
    ok = not fails and n >= (1 if a.only else 50)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--fixtures", default=str(ROOT / "tests" / "fixtures"))
    ap.add_argument("--fd", action="store_true"); ap.add_argument("--fd-max-atoms", type=int, default=24)
    ap.add_argument("--only"); ap.add_argument("--backend", default="cpu64", choices=["cpu64", "metal"])
    a = ap.parse_args()
    fx = Path(a.fixtures)
    rtol = json.loads((ROOT / "tolerances" / "tolerances.json").read_text())["C1"]["primary"]["threshold"]["energy_slot_rel_to_max1"]
    if a.backend == "metal":
        return main_metal(a, fx)
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
                    print(f"note: {cf.stem}: max |analytic-FD| = {maxd:.3f}")
                else:
                    worst["fd_abs"] = max(worst.get("fd_abs", 0.0), maxd)
    for f in fails[:60]:
        print("FAIL", f)
    print("worst relative energy differences:", {k: f"{v:.2e}" for k, v in worst.items()})
    print(f"NB-1: {n} fixtures compared, {nfd} with finite-difference check, {len(fails)} failures")
    ok = not fails and n >= (1 if a.only else 50)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
