#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Print the headline numbers of an M1 run (used to write VALIDATION.md / DEVELOPMENT_LOG.md from data, not memory).
usage: summarize_m1.py <work root>"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_builds  # noqa: E402

R = Path(sys.argv[1]); runs = R / "runs"; rep = R / "reports"; exp = R / "exp"
REPO = Path(__file__).resolve().parents[2]
J = lambda p: json.loads(Path(p).read_text())

print("## builds / runs"); 
for d in sorted(runs.iterdir()):
    if d.is_dir() and (d / "summary.json").exists():
        s = J(d / "summary.json"); n = len(s["cases"]); ok = sum(c["valid"] for c in s["cases"])
        sha = next(iter(J(d / next(iter(c["case"] for c in s["cases"])) / "result.json").get("provenance", {}).values()), "")
        r0 = J(d / s["cases"][0]["case"] / "result.json")
        print(f"  {d.name:24s} valid {ok}/{n}  lammps_sha256={r0['provenance']['lammps_binary_sha256'][:16]}.. launcher={r0['provenance'].get('launcher')} args={r0['provenance'].get('lammps_args')}")

print("## qeq statistics (inst-gcc)")
it, res, sq = [], [], []
for c in sorted((runs / "inst-gcc").iterdir()):
    f = c / "result.json"
    if f.exists():
        r = J(f)
        if r.get("qeq"): it.append(max(r["qeq"]["iters_s"], r["qeq"]["iters_t"]))
        if r.get("independent_eem"): res.append(r["independent_eem"]["equalization_residual_eV"]); sq.append(abs(r["sum_q"]))
print(f"  CG iterations min/median/max = {min(it)}/{int(np.median(it))}/{max(it)} (imax 500); independent equalization residual max = {max(res):.2e} eV; max |sum q| = {max(sq):.2e}")

print("## instrumentation equivalence")
for f in sorted(rep.glob("equiv_*.json")):
    d = J(f); s = d["summary"]; print(f"  {f.stem}: {s['bitwise_identical']}/{s['cases_compared']} bit-identical, max dE_slot={s['max_dE_slot']:.2e} dq={s['max_dq']:.2e} dF={s['max_dF']:.2e}")

nf = J(rep / "noise_floor.json")
print("## noise floor: builds", nf["builds"])
for name, st in nf["stats"].items():
    print(f"  set {name} ({len(nf['sets'][name])} fixtures)")
    for q, w in st["worst"].items(): print(f"    {q:22s} {w['value']:.3e}  ({w['case']}; {w['pair']})")
print("  strict-IEEE pair gcc vs clang in set B:", J(rep / "noise_floor.json")["stats"][[k for k in nf["stats"] if k.startswith("B_")][0]]["pairs"].get("stock-clang vs stock-gcc"))

t = J(REPO / "tolerances/tolerances.json")
print("## frozen tolerances: C1 primary threshold", t["C1"]["primary"]["threshold"]); print("   observed floor", t["C1"]["primary"]["observed_floor"])
print("   alternative (pre-registered set only):", t["C1"]["alternative_if_amendment_rejected"]["threshold"])
print("   C3 neutrality:", t["C3"]["neutrality_and_equalization_residual"]); print("   conditioning counts:", t["conditioning"]["counts"])
cond = t["conditioning"]["classification"]
print("   not pre-registered-well-conditioned:", {k: v["fails_pre_registered"] for k, v in cond.items() if not v["pre_registered_well_conditioned"]})
print("   proposed X1/X2 only:", {k: v["extended_flags"] for k, v in cond.items() if v["pre_registered_well_conditioned"] and not v["extended_well_conditioned"]})

nb = J(rep / "nonbonded_vs_independent.json")
worst = max(max(r[k]["rel"] for k in ("e_vdW", "e_ele", "e_pol")) for r in nb.values())
pc = [(k, r["pairs"]["lammps_vdw"], r["pairs"]["ref"]) for k, r in nb.items()]
print(f"## nonbonded vs independent reference: {len(nb)} fixtures, worst relative diff {worst:.2e}, pair counts equal in {sum(1 for _, a, b in pc if a == b)}/{len(pc)}; total pairs {sum(a for _, a, _ in pc)}")

p = J(exp / "periodic.json")
print("## E1 supercell invariance")
for k, rows in p.items():
    ok = [r for r in rows if r["valid"]]
    print(f"  {k:34s} reps={[tuple(r['reps']) for r in ok][-1]} max cell-energy dev={max(r['total_dev_vs_first_per_cell'] for r in ok):.2e} max q dev={max(r['max_charge_dev_vs_first'] for r in ok):.1e} max F dev={max(r['max_force_dev_vs_first'] for r in ok):.1e} nonzero-charge={max(abs(v) for v in ok[0]['_q0']):.2e}")
g = J(exp / "ghost.json")
print("## E2 translation: max slot dev", max(r.get("max_slot_dev", 0) for rows in g["E2_translation"].values() for r in rows if r["valid"]), "max q dev", max(r.get("max_q_dev", 0) for rows in g["E2_translation"].values() for r in rows if r["valid"]), "max F dev", max(r.get("max_f_dev", 0) for rows in g["E2_translation"].values() for r in rows if r["valid"]))
print("## E3 skin: ", {k: max(abs(r.get("dev_vs_skin2") or 0) for r in rows) for k, rows in g["E3_skin"].items()})
print("## E3 swb:", {k: [(r["swb"], r.get("eem_residual_eV"), r["valid"]) for r in rows] for k, rows in g["E3_swb"].items()})
gc = exp / "ghost_cutoff.json"
if gc.exists():
    d = J(gc); print("## REF-GHOST (shell vs same force field, skin 4):", {k: max((r.get("max_dF_vs_skin4", 0) for r in rows), default=0) for k, rows in d.items()})
q = J(exp / "quirks.json")
a = q["Q09a_j_eq_l"]; print("## Q-09:", {k: (a[k]["raw_cnt"], a[k]["angle_slots_ev_epen_ecoa"][0]) for k in ("none", "S1", "S2", "S3", "S1+S2", "S1+S2+S3")}, "additivity", a["additivity_S1+S2"]["max_abs_diff"], a["additivity_S1+S2+S3"]["max_abs_diff"])
print("   Q-09b:", {k: v.get("max_abs_diff_vs_additive") for k, v in q["Q09b_j_ne_l"].items()})
print("## Q-12:", {k: [(r["r"], r.get("BOp_s"), r.get("BOp_pi"), r.get("BOp_pipi"), round(r.get("e_bond", 0), 4)) for r in rows] for k, rows in q["Q12_absent_bond_pair"].items() if "removed" in k})
print("## Q-32:", J(exp / "q32_hbond_identity.json")["systems"]); print("## Q-34:", J(exp / "q34_ovun.json"))
fd = J(rep / "fd_all.json"); cases = {f.stem: J(f) for f in (REPO / "tests/fixtures/cases").glob("*.json")}
good = [(fd[c]["fixedq"]["max_abs_err_h"], c) for c in fd if "error" not in fd[c] and cond[c]["extended_well_conditioned"] and "known-reference-quirk" not in cases[c].get("tags", [])]
print(f"## FD fixed-q: {len(good)} well-conditioned non-quirk fixtures, max |dF| = {max(good)[0]:.2e} ({max(good)[1]}); quirk/ill-conditioned:", {c: f"{fd[c]['fixedq']['max_abs_err_h']:.2e}" for c in fd if "error" not in fd[c] and (not cond[c]["extended_well_conditioned"] or 'known-reference-quirk' in cases[c].get('tags', [])) and fd[c]['fixedq']['max_abs_err_h'] > 1e-4})
print("   suspect components:", {c: fd[c]["fixedq"]["n_discontinuity_suspect"] for c in fd if "error" not in fd[c] and fd[c]["fixedq"]["n_discontinuity_suspect"]})
rel = [fd[c]["relaxed"]["max_abs_err"] / max(fd[c]["fixedq"]["F_rms_checked"], 1.0) for c in fd if "error" not in fd[c] and fd[c]["natoms"] > 1 and fd[c]["relaxed"]["max_abs_err"] > 1e-4 and 'known-reference-quirk' not in cases[c].get('tags', [])]
print(f"   relaxed-q deviation (Q-03/Q-27) over {len(rel)} charged fixtures: max/F_rms median={np.median(rel):.2e} max={max(rel):.2e}; absolute max={max(fd[c]['relaxed']['max_abs_err'] for c in fd if 'error' not in fd[c] and 'known-reference-quirk' not in cases[c].get('tags', [])):.3f}")
