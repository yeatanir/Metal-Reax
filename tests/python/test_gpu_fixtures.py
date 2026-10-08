#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""GPU-1 (Metal-32, all terms, frozen charges): the bonded and nonbonded kernels on the real Metal device vs the pinned-LAMMPS values of M1 and vs the
CPU-64 engine, under the owner-set C3 criteria of tolerances/tolerances.json (energy per atom, force max component and RMS; the force criteria on the
fixtures classified well conditioned, all others reported). FORCE-2: two launches must give bitwise identical results.
usage: test_gpu_fixtures.py --tool BIN --ffield-dir DIR [--only NAME]"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "reaxref")); sys.path.insert(0, str(HERE))
from test_full_fixtures import case_text, TERMS  # noqa: E402


def run(tool, ff, extra, txt, backend):
    r = subprocess.run([tool, "--ffield", str(ff), *extra, "--backend", backend, "--grad", str(txt)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:600])
    e, g = {}, {}
    for ln in r.stdout.splitlines():
        k, *v = ln.split()
        if k == "grad": g[int(v[0])] = [float(t) for t in v[1:]]
        else: e[k] = float(v[0])
    return e, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--fixtures", default=str(ROOT / "tests" / "fixtures")); ap.add_argument("--only")
    a = ap.parse_args()
    fx = Path(a.fixtures)
    c3 = json.loads((ROOT / "tolerances" / "tolerances.json").read_text())["C3"]
    e_tol, f_max, f_rms = c3["energy_per_atom_abs_kcal_mol"], c3["force_max_component_kcal_mol_A"], c3["force_rms_kcal_mol_A"]
    fails, n, nwc = [], 0, 0
    w = {"e_atom": 0.0, "term_rel": 0.0, "f_ref": 0.0, "f_ref_rms": 0.0, "f_cpu": 0.0, "f_cpu_all": 0.0}
    worst_f = ("", 0.0)
    gpu = []
    with tempfile.TemporaryDirectory() as td:
        for cf in sorted((fx / "cases").glob("*.json")):
            if a.only and cf.stem != a.only: continue
            case = json.loads(cf.read_text()); ref = json.loads((fx / "reference" / cf.name).read_text())
            if not ref.get("valid", True): continue
            txt = Path(td) / "case.txt"; txt.write_text(case_text(case, ref))
            extra = ["--elements", ",".join(case["elements"])] + (["--lgvdw"] if case.get("pair", {}).get("lgvdw") else [])
            ff = Path(a.ffield_dir) / case["ffield"]["name"]
            try:
                e64, g64 = run(a.tool, ff, extra, txt, "cpu64")
                em, gm = run(a.tool, ff, extra, txt, "metal")
                em2, gm2 = run(a.tool, ff, extra, txt, "metal")
            except RuntimeError as ex:
                fails.append(f"{cf.stem}: {ex}"); continue
            n += 1
            nat = len(ref["positions"])
            gpu.append(em.get("bonded_gpu_ms", 0) + em.get("gpu_ms", 0))
            if gm != gm2 or any(em[k] != em2[k] for k in TERMS): fails.append(f"{cf.stem}: FORCE-2 two Metal launches differ")
            tot_err = sum(abs(em[k] - ref["energy_data_fields"][k]) for k in TERMS)   # conservative: no cancellation between terms
            net_err = abs(sum(em[k] for k in TERMS) - ref["energy"]["slot_sum"])
            w["e_atom"] = max(w["e_atom"], net_err / nat)
            if net_err / nat > e_tol: fails.append(f"{cf.stem}: total energy per atom differs by {net_err / nat:.3e} (> {e_tol})")
            for k in TERMS:
                want = ref["energy_data_fields"][k]
                w["term_rel"] = max(w["term_rel"], abs(em[k] - want) / max(1.0, abs(want)))
                if abs(em[k] - want) / nat > e_tol: fails.append(f"{cf.stem}: {k} Metal {em[k]!r} ref {want!r}")
            dref = [-gm[i + 1][c] - ref["forces"][i][c] for i in range(nat) for c in range(3)]
            dcpu = [gm[i + 1][c] - g64[i + 1][c] for i in range(nat) for c in range(3)]
            fr, frr = max(abs(x) for x in dref), (sum(x * x for x in dref) / len(dref)) ** 0.5
            fc = max(abs(x) for x in dcpu)
            w["f_cpu_all"] = max(w["f_cpu_all"], fc)
            wc = ref["conditioning"].get("well_conditioned_extended", True)
            if wc:
                nwc += 1
                w["f_ref"] = max(w["f_ref"], fr); w["f_ref_rms"] = max(w["f_ref_rms"], frr); w["f_cpu"] = max(w["f_cpu"], fc)
                if fr > worst_f[1]: worst_f = (cf.stem, fr)
                if fr > f_max or frr > f_rms: fails.append(f"{cf.stem}: forces vs LAMMPS max {fr:.3e} (limit {f_max}) rms {frr:.3e} (limit {f_rms})")
    for f in fails[:60]: print("FAIL", f)
    print(f"worst: total E/atom {w['e_atom']:.2e} (C3 {e_tol}); per-term relative {w['term_rel']:.2e}; forces vs LAMMPS (well-conditioned) max {w['f_ref']:.2e} (C3 {f_max}) "
          f"rms {w['f_ref_rms']:.2e} (C3 {f_rms}) [worst: {worst_f[0]}]; vs CPU-64 {w['f_cpu']:.2e} (all fixtures {w['f_cpu_all']:.2e})")
    if gpu: print(f"GPU time per evaluation (bonded+nonbonded kernels): median {sorted(gpu)[len(gpu) // 2]:.3f} ms, max {max(gpu):.3f} ms")
    print(f"GPU-1 all terms: {n} fixtures, {nwc} well conditioned, {len(fails)} failures")
    ok = not fails and n >= (1 if a.only else 50)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
