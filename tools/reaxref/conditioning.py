#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Mechanical conditioning classifier for reference fixtures (NUMERICAL_POLICY 5.2 / 5.6).

PRE-REGISTERED definition (NUMERICAL_POLICY 5.2, fixed at the M1 review before any data existed). A fixture is
"well-conditioned" iff ALL of:
  W1 QEq converged and independently verified            (runner validity)
  W2 no interaction within 1e-4 (relative) of a hard threshold of ENGINE_SPEC 8:
       - total BO' of any candidate pair vs bo_cut (bond creation), evaluated for EVERY pair within bond_cut incl. images
       - BO vs thb_cut and vs HB_THRESHOLD, product BO_ij*BO_jk vs thb_cutsq (valence/torsion/H-bond gates)
       - pair distances vs bond_cut (only where BO'(bond_cut) >= bo_cut, i.e. where a bond would be cut) and vs hbond_cut
         (only for donor-H / acceptor pairs): a distance edge is a threshold only if it gates a non-negligible interaction
       - nlp truncation kink: Delta_e/2 within 1e-4 of a NON-ZERO integer (trunc() is constant on (-2,2))
  W3 no atom with total BO' within 1e-3 (relative) of bo_cut
  W4 minimum interatomic distance (incl. periodic images) >= 0.8 A
  W5 |Delta_lp| < 1.0 for every owned atom

EXTENDED classes found by M1 execution (NOT part of the pre-registered definition; PROPOSED amendments awaiting owner
confirmation -- the freeze file records results under both):
  X1 NEAR_LINEAR : a valence-angle triple (centre owned, both BO > thb_cut) with sin(theta) < 1e-3. The kernel divides by
                   sin(theta) clamped at 1e-5; rounding of cos(theta) is amplified (VALIDATION E-NOISE-2, ENGINE_SPEC Q-33).
  X2 PRODUCT     : BO_ij*BO_jk within 1e-3 (relative) of thb_cutsq; the energy is discontinuous there (E-FD-2); wider than W2's
                   1e-4 because 1e-4 A displacements move products by ~1e-3 relative.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner, ref_nonbonded  # noqa: E402

HB_THRESHOLD = 1e-2        # reaxff_defs.h
W2_REL = 1e-4
W3_REL = 1e-3
W4_MIN_DIST = 0.8
W5_MAX_DLP = 1.0
X1_SIN = 1e-3
X2_REL = 1e-3


def _pairs(x, types, cell_mat, periodic, P, rcut):
    """Yield (i, j, r) for every unordered pair (incl. self-images) within rcut, with explicit images."""
    n = len(x)
    ext = float(np.ptp(x, axis=0).max()) if n else 0.0
    shifts = np.array(ref_nonbonded.image_shifts(cell_mat, periodic, rcut, ext), float)
    sh = shifts @ cell_mat
    for i in range(n):
        for j in range(i, n):
            d = x[j] + sh - x[i]
            r = np.sqrt((d * d).sum(axis=1))
            m = (r <= rcut) & (r > 0.0)
            if i == j:
                keep = np.array([tuple(s) > (0.0, 0.0, 0.0) for s in shifts]); m &= keep
            for rr in r[m]:
                yield i, j, float(rr)


def classify(run_dir, case):
    """run_dir = <suite>/<case>/ containing result.json and diag/."""
    res = json.loads((run_dir / "result.json").read_text())
    D = runner.read_call_dump(res["diag"]["file"])
    P = runner.read_params_dump(Path(res["diag"]["file"]).parent / "params.txt")
    n = int(D["meta"]["n"])
    ctrl = D["control"]
    bo_cut, thb_cut, thb_cutsq = ctrl["bo_cut"], ctrl["thb_cut"], ctrl["thb_cutsq"]
    bond_cut, hbond_cut = ctrl["bond_cut"], ctrl["hbond_cut"]
    x = np.array(res["positions"]); types = np.array(runner.ffield_indices(P, case["elements"]))[np.array(res["types"]) - 1]
    cm = runner.cell_matrix(case["cell"]); per = case.get("periodic", [True] * 3)
    X = np.array([[float(v) for v in a[4:7]] for a in D["atoms"]])    # owned + ghost positions
    W = {}

    # ---- W2 / W3: bond-order thresholds ------------------------------------------------------------------
    rel2 = np.inf
    G = P["G"]
    for i, j, r in _pairs(x, types, cm, per, P, max(bond_cut, hbond_cut)):
        Si, Sj = P["S"][int(types[i])], P["S"][int(types[j])]
        # hbond_cut edge: only a threshold for a donor-hydrogen / acceptor pair (p_hbond 1 and 2); E_hb is non-zero at the edge
        if hbond_cut > 0 and abs(r - hbond_cut) / hbond_cut < W2_REL and {int(Si[14]), int(Sj[14])} == {1, 2}:
            rel2 = min(rel2, abs(r - hbond_cut) / hbond_cut)
        if r > bond_cut: continue
        T = P["T"][(int(types[i]), int(types[j]))]
        rs_i, rs_j = float(Si[1]), float(Sj[1]); rp_i, rp_j = float(Si[7]), float(Sj[7]); rpp_i, rpp_j = float(Si[17]), float(Sj[17])
        BO = 0.0
        if rs_i > 0 and rs_j > 0: BO += (1.0 + bo_cut) * np.exp(T[0] * (r / T[6]) ** T[1])
        if rp_i > 0 and rp_j > 0: BO += np.exp(T[2] * (r / T[7]) ** T[3])
        if rpp_i > 0 and rpp_j > 0: BO += np.exp(T[4] * (r / T[8]) ** T[5])
        if bo_cut > 0: rel2 = min(rel2, abs(BO - bo_cut) / bo_cut)
        # bond_cut edge: only a threshold where a bond would exist there (BO'(r) >= bo_cut); otherwise nothing is cut
        if abs(r - bond_cut) / bond_cut < W2_REL and BO >= bo_cut: rel2 = min(rel2, abs(r - bond_cut) / bond_cut)
    nbr = {}
    atom_bo = np.zeros(n)
    for b in D["bonds"]:
        i, j = int(b[0]), int(b[1]); BO = float(b[8]); prime = float(b[5]) + float(b[6]) + float(b[7])
        nbr.setdefault(i, []).append((j, BO))
        if i < n: atom_bo[i] += prime
        for thr in (thb_cut, HB_THRESHOLD):
            if thr > 0: rel2 = min(rel2, abs(BO - thr) / thr)
    # W3 : total BO' of an atom vs bo_cut (an atom with NO bonds has total 0 and is far from the cut)
    rel3 = min((abs(atom_bo[i] - bo_cut) / bo_cut for i in range(n) if atom_bo[i] > 0), default=np.inf)

    # ---- valence-angle triples: products vs thb_cutsq, near-linearity -------------------------------------
    min_sin = 1.0; prod_rel = np.inf
    for j in range(n):
        lst = [(k, bo) for k, bo in nbr.get(j, []) if bo > thb_cut]
        for a in range(len(lst)):
            for c in range(a + 1, len(lst)):
                v1 = X[lst[a][0]] - X[j]; v2 = X[lst[c][0]] - X[j]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                s = np.sqrt(max(0.0, 1.0 - min(1.0, abs(cos)) ** 2))
                min_sin = min(min_sin, s)
                if thb_cutsq > 0: prod_rel = min(prod_rel, abs(lst[a][1] * lst[c][1] - thb_cutsq) / thb_cutsq)
    rel2 = min(rel2, prod_rel) if prod_rel < W2_REL else rel2

    # ---- nlp truncation kink and Delta_lp ----------------------------------------------------------------
    V = np.array(D["vars"])
    de = V[:n, 6]          # Delta_e
    # nlp = exp(-p_lp1 (2+vlpex)^2) - trunc(De/2) is discontinuous where trunc() steps, i.e. at NON-ZERO integer De/2
    # (trunc is constant on (-2, 2), so De ~ 0 -- a saturated sp3 atom -- is NOT a kink)
    half = de / 2.0
    dist_to_nonzero_int = np.where(np.round(half) != 0, np.abs(half - np.round(half)), np.inf)
    kink = float(np.min(dist_to_nonzero_int)) if n else np.inf
    if kink < W2_REL: rel2 = min(rel2, kink)
    dlp_max = float(np.max(np.abs(V[:n, 5]))) if n else 0.0

    # ---- W4 minimum distance ------------------------------------------------------------------------------
    dmin = min((r for _, _, r in _pairs(x, types, cm, per, P, 3.0)), default=np.inf)

    fixed_fail = []
    if rel2 < W2_REL: fixed_fail.append("W2_threshold")
    if rel3 < W3_REL: fixed_fail.append("W3_atom_BOprime_near_bo_cut")
    if dmin < W4_MIN_DIST: fixed_fail.append("W4_min_distance")
    if dlp_max >= W5_MAX_DLP: fixed_fail.append("W5_Delta_lp")
    if not res["valid"]: fixed_fail.append("W1_invalid")
    ext = []
    if min_sin < X1_SIN: ext.append("X1_NEAR_LINEAR")
    if prod_rel < X2_REL: ext.append("X2_PRODUCT")
    return {"W2_min_rel_to_threshold": None if not np.isfinite(rel2) else float(rel2), "W3_min_rel_atom_BOprime": None if not np.isfinite(rel3) else float(rel3),
            "W4_min_distance": None if not np.isfinite(dmin) else float(dmin), "W5_max_abs_Delta_lp": dlp_max,
            "nlp_kink_min_dist": None if not np.isfinite(kink) else kink,
            "X1_min_sin_theta": float(min_sin), "X2_min_rel_to_thb_cutsq": None if not np.isfinite(prod_rel) else float(prod_rel),
            "fails_pre_registered": fixed_fail, "extended_flags": ext,
            "well_conditioned_pre_registered": not fixed_fail, "well_conditioned_extended": not fixed_fail and not ext}


def main():
    run = Path(sys.argv[1]); cases = Path(sys.argv[2]); out = {}
    for d in sorted(run.iterdir()):
        if (d / "result.json").exists() and (d / "diag" / "call_000000.txt").exists():
            case = json.loads((cases / (d.name + ".json")).read_text())
            out[d.name] = classify(d, case)
    if "--json" in sys.argv: Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if not v["well_conditioned_extended"]:
            print(f"{k:28s} pre-registered fails={v['fails_pre_registered']} extended={v['extended_flags']} "
                  f"(W2 {v['W2_min_rel_to_threshold']}, W3 {v['W3_min_rel_atom_BOprime']}, dmin {v['W4_min_distance']}, dlp {v['W5_max_abs_Delta_lp']:.2f}, min_sin {v['X1_min_sin_theta']:.1e})")
    n1 = sum(1 for v in out.values() if v["well_conditioned_pre_registered"]); n2 = sum(1 for v in out.values() if v["well_conditioned_extended"])
    print(f"{len(out)} fixtures: well-conditioned (pre-registered) {n1}; well-conditioned (pre-registered + proposed X1/X2) {n2}")


if __name__ == "__main__":
    main()
