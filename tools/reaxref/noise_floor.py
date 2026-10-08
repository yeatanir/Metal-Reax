#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Measure the reference noise floor: how much do independent builds of the SAME pinned LAMMPS source differ?
Builds compared pairwise (all stock, uninstrumented): gcc, clang (strict IEEE, no FMA) and the same compilers with
-march=native -ffp-contract=fast (FMA contraction). Fixtures are split by conditioning.py into well-conditioned and
ill-conditioned classes; the classes are reported separately and only the well-conditioned class feeds the C1 thresholds.
usage: noise_floor.py <runs-dir> <conditioning.json> --out noise_floor.json [builds...]"""
import itertools, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_builds import load, vec  # noqa: E402


def stats(runs, cond, ids):
    pairs = list(itertools.combinations(sorted(runs), 2))
    out = {"pairs": {}, "worst": {}}
    quantities = ["dE_total_per_atom", "dE_slot_max", "dE_slot_max_per_atom", "dE_slot_rel", "dq_max", "dF_max", "dF_rms"]
    worst = {q: (0.0, None, None) for q in quantities}
    for pa, pb in pairs:
        row = {q: 0.0 for q in quantities}
        for cid in ids:
            ra, rb = runs[pa][cid], runs[pb][cid]
            ea, sa, qa, fa = vec(ra); eb, sb, qb, fb = vec(rb)
            n = len(qa)
            v = {"dE_total_per_atom": abs(ea - eb)[0] / n, "dE_slot_max": np.abs(sa - sb).max(), "dE_slot_max_per_atom": np.abs(sa - sb).max() / n,
                 "dE_slot_rel": float((np.abs(sa - sb) / np.maximum(np.abs(sa), np.abs(sb)).clip(min=1.0)).max()),
                 "dq_max": np.abs(qa - qb).max(), "dF_max": np.abs(fa - fb).max(), "dF_rms": float(np.sqrt(((fa - fb) ** 2).mean()))}
            for q in quantities:
                row[q] = max(row[q], float(v[q]))
                if float(v[q]) > worst[q][0]: worst[q] = (float(v[q]), cid, f"{pa} vs {pb}")
        out["pairs"][f"{pa} vs {pb}"] = row
    out["worst"] = {q: {"value": w[0], "case": w[1], "pair": w[2]} for q, w in worst.items()}
    return out


def main():
    rundir = Path(sys.argv[1]); cond = json.loads(Path(sys.argv[2]).read_text())
    out_path = sys.argv[sys.argv.index("--out") + 1]
    builds = [a for a in sys.argv[3:] if not a.startswith("--") and a != out_path]
    runs = {b: load(rundir / b) for b in builds}
    ids_all = sorted(set.intersection(*[set(r) for r in runs.values()]) & set(cond))
    sets = {
        "A_well_conditioned_pre_registered": [c for c in ids_all if cond[c]["well_conditioned_pre_registered"]],
        "B_well_conditioned_pre_registered_plus_proposed_X1_X2": [c for c in ids_all if cond[c]["well_conditioned_extended"]],
        "C_not_in_B": [c for c in ids_all if not cond[c]["well_conditioned_extended"]],
        "D_proposed_X1_X2_only": [c for c in ids_all if cond[c]["well_conditioned_pre_registered"] and not cond[c]["well_conditioned_extended"]],
    }
    rep = {"builds": builds, "n_cases": len(ids_all), "sets": {k: v for k, v in sets.items()},
           "stats": {k: stats(runs, cond, v) for k, v in sets.items() if v}}
    Path(out_path).write_text(json.dumps(rep, indent=1))
    for name, st in rep["stats"].items():
        print(f"== {name}  ({len(sets[name])} fixtures)")
        for q, w in st["worst"].items():
            print(f"  {q:22s} worst={w['value']:.3e}  ({w['case']}, {w['pair']})")


if __name__ == "__main__":
    main()
