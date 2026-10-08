#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Compare LAMMPS vdW / Coulomb / polarisation energies and pair counts with the independent explicit-image reference.
usage: check_nonbonded.py <run-dir> <cases-dir> [--tol-abs 1e-8] [--json out.json]"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner, ref_nonbonded  # noqa: E402


def main():
    run, cases = Path(sys.argv[1]), Path(sys.argv[2])
    tol_rel = 1e-11
    out = {}; worst = 0.0; bad = []
    for d in sorted(run.iterdir()):
        f = d / "result.json"
        if not f.exists(): continue
        r = json.loads(f.read_text())
        if not r.get("valid"): continue
        case = json.loads((cases / (d.name + ".json")).read_text())
        P = runner.read_params_dump(d / "diag" / "params.txt")
        x = np.array(r["positions"]); types = np.array(runner.ffield_indices(P, case["elements"]))[np.array(r["types"]) - 1]; q = np.array(r["charges"])
        cm = runner.cell_matrix(case["cell"])
        ctrl = r["diag"]["control"]
        lg = bool(case.get("pair", {}).get("lgvdw", False))
        ref = ref_nonbonded.nonbonded_reference(x, types, q, cm, case.get("periodic", [True] * 3), P, ctrl, lg)
        ef = r["diag"]["energy_fields"]
        cnt = sum(v["count"] for k, v in r["diag"]["tallies"].items() if k.startswith("vdw."))
        cnt_c = sum(v["count"] for k, v in r["diag"]["tallies"].items() if k.startswith("coulomb."))
        row = {}
        for k in ("e_vdW", "e_ele", "e_pol"):
            diff = abs(ef[k] - ref[k]); scale = max(abs(ef[k]), 1.0)
            row[k] = {"lammps": ef[k], "ref": ref[k], "abs": diff, "rel": diff / scale}
            worst = max(worst, diff / scale)
            if diff / scale > tol_rel: bad.append((d.name, k, diff))
        row["pairs"] = {"lammps_vdw": cnt, "lammps_coulomb": cnt_c, "ref": ref["n_pairs"]}
        if cnt != ref["n_pairs"] or cnt_c != ref["n_pairs"]: bad.append((d.name, "pair-count", (cnt, cnt_c, ref["n_pairs"])))
        out[d.name] = row
        print(f"{d.name:30s} dE_vdW={row['e_vdW']['abs']:.2e} dE_ele={row['e_ele']['abs']:.2e} dE_pol={row['e_pol']['abs']:.2e}  pairs lammps={cnt} ref={ref['n_pairs']}")
    print(f"worst relative difference {worst:.3e}; mismatches: {bad if bad else 'none'}")
    if "--json" in sys.argv: Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1))
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
