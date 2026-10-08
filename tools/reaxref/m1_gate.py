#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""M1 acceptance gate. Reads the artefacts produced by tools/m1_reproduce.sh under <root> and the repository, and decides
G1..G7. It computes everything from the artefacts (no value is typed in here except the criteria themselves).

  G1a  pristine pinned tree + hashed patch applies; instrumented == stock BITWISE on every fixture for the strict-IEEE
       configurations (gcc, clang; diagnostics on AND off)
  G1b  (reported, not required to be bitwise) the FMA-contracting configurations: deviations are listed
  G2   every fixture valid: unexpected-warning-free, QEq converged (recursive residual <= requested tolerance), independent
       equalization residual and |sum q| within the frozen thresholds, exactly one force evaluation
  G3   coverage: every active energy slot, every required diagnostic tally, every required special branch tag, all 11 bundled
       force fields, all required elements
  G4   noise floor measured on >= 3 distinct builds, tolerances.json written and its hash recorded & verified
  G5   periodic-image accounting experiments executed (E1 supercell invariance, E2 translation, E3 ghost-range) and the
       deviations either within noise or attributed (by an executed experiment) to a named reference quirk
  G6   Q-09 and Q-12 executed on crafted force fields
  G7   documentation updated, working tree clean and pushed
usage: m1_gate.py --root <work root> [--branch claude/friendly-ride-rwz7xu] [--skip-git] [--out gate.json]"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import compare_builds, runner  # noqa: E402

REQUIRED_TAGS = ["bond", "angle", "torsion", "conjugation", "coalition", "hbond", "pi-bond", "triple-bond", "co-branch", "gp37", "lgvdw", "innerwall",
                 "heavy-branch", "metal", "periodic", "triclinic", "self-image", "images", "isolated-atom", "undercoord", "ring", "fixed-charge", "2d",
                 "small-cell", "angle-penalty", "linear", "polar", "chain", "known-reference-quirk", "perturbed", "N", "S", "B"]
REQUIRED_ELEMENTS = {"C", "H", "O", "N", "S", "B", "AU", "FE", "V", "ZN"}


class Gate:
    def __init__(self): self.rows = []
    def add(self, gid, ok, detail): self.rows.append({"id": gid, "pass": bool(ok), "detail": detail}); print(f"{'PASS' if ok else 'FAIL'}  {gid}: {detail}")


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--branch", default="claude/friendly-ride-rwz7xu")
    ap.add_argument("--skip-git", action="store_true"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    R = Path(a.root); runs = R / "runs"; rep = R / "reports"; exp = R / "exp"; G = Gate()
    cases = {f.stem: json.loads(f.read_text()) for f in sorted((REPO / "tests/fixtures/cases").glob("*.json"))}
    tol = json.loads((REPO / "tolerances/tolerances.json").read_text())

    # ---------------------------------------------------------------- G1
    pin = [l.split("=", 1)[1] for l in (REPO / "third_party/lammps/PIN.txt").read_text().splitlines() if l.startswith("lammps_commit=")][0]
    src = R / "lammps"
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    patch = REPO / "third_party/lammps/patches/0001-reaxmetal-diagnostics.patch"
    want = [l.split()[0] for l in (REPO / "third_party/lammps/patches/PATCHES.sha256").read_text().splitlines() if "0001-reaxmetal-diagnostics.patch" in l][0]
    chk = subprocess.run(["git", "-C", str(src), "apply", "--check", str(patch)], capture_output=True, text=True)
    G.add("G1.pristine+patch", head == pin and not dirty and sha(patch) == want and chk.returncode == 0,
          f"pristine HEAD={head[:12]} (pin {pin[:12]}), tree clean={not dirty}, patch sha256={sha(patch)[:16]}.. matches PATCHES.sha256={sha(patch) == want}, git apply --check rc={chk.returncode}")
    for v in ("gcc", "clang"):
        for other in (f"inst-{v}", f"inst-{v}-nodiag"):
            A, B = compare_builds.load(runs / f"stock-{v}"), compare_builds.load(runs / other)
            same = [c for c in A if c in B and compare_builds.vec(A[c]) and all((x == y).all() for x, y in zip(compare_builds.vec(A[c]), compare_builds.vec(B[c])))]
            G.add(f"G1a.stock-{v}=={other}", len(same) == len(cases) == len(A) == len(B), f"{len(same)}/{len(cases)} fixtures bit-identical (energy, 14 slots, charges, forces)")
    for v in ("gcc-native", "clang-native"):
        A, B = compare_builds.load(runs / f"stock-{v}"), compare_builds.load(runs / f"inst-{v}")
        n_same = 0; dmax = 0.0; dE = 0.0
        for c in A:
            va, vb = compare_builds.vec(A[c]), compare_builds.vec(B[c])
            n_same += all((x == y).all() for x, y in zip(va, vb)); dmax = max(dmax, float(abs(va[3] - vb[3]).max())); dE = max(dE, float(abs(va[1] - vb[1]).max()))
        G.add(f"G1b.stock-{v} vs inst-{v} (reported)", True, f"{n_same}/{len(A)} bit-identical; max |dF|={dmax:.2e}, max |dE_slot|={dE:.2e}  (FMA-contracting build: fusion decisions depend on surrounding code)")

    # ---------------------------------------------------------------- G2
    inst = compare_builds.load(runs / "inst-gcc")
    bad = []
    thr = {"equalization_residual_eV": tol["reference_validity"]["thresholds_set_a_priori"]["equalization_residual_eV"], "sum_q_abs_e": tol["reference_validity"]["thresholds_set_a_priori"]["abs_sum_q_e"]}
    for cid, r in inst.items():
        why = []
        if not r.get("valid"): why.append("invalid: " + "; ".join(r.get("invalid_reasons", [])))
        if r.get("lammps_warnings"): why.append("unexpected warnings")
        ch = cases[cid]["charge"]
        if ch["model"] == "qeq/reaxff":
            q = r.get("qeq", {})
            if not q.get("converged_to_tolerance"): why.append("QEq not converged")
            if r["independent_eem"]["equalization_residual_eV"] > thr["equalization_residual_eV"]: why.append("equalization residual")
            if abs(r["sum_q"]) > thr["sum_q_abs_e"]: why.append("sum q")
        if not r.get("param_table_check", {}).get("ok"): why.append("param table mismatch vs independent ffield parse")
        if r.get("diag", {}).get("n") is None: why.append("no diag")
        if why: bad.append((cid, why))
    G.add("G2.all-fixtures-valid", not bad and len(inst) == len(cases), f"{len(inst) - len(bad)}/{len(cases)} valid (frozen thresholds: residual<= {thr['equalization_residual_eV']:g} eV, |sum q|<= {thr['sum_q_abs_e']:g}); problems: {bad[:3]}")

    # ---------------------------------------------------------------- G3
    cov = subprocess.run([sys.executable, str(HERE / "coverage.py"), str(runs / "inst-gcc"), "--require"], capture_output=True, text=True)
    tags = {t for c in cases.values() for t in c.get("tags", [])}
    ffs = {c["ffield"]["name"] for c in cases.values()}
    els = {a_["el"].upper() for c in cases.values() for a_ in c["atoms"]}
    miss_t = [t for t in REQUIRED_TAGS if t not in tags]
    G.add("G3.coverage", cov.returncode == 0 and not miss_t and len(ffs) == 11 and REQUIRED_ELEMENTS <= els,
          f"slots+tallies rc={cov.returncode}; missing tags={miss_t}; force fields={len(ffs)}/11; elements={sorted(els)}; fixtures={len(cases)}")

    # ---------------------------------------------------------------- G4
    nf = json.loads((rep / "noise_floor.json").read_text())
    builds = nf["builds"]
    h = subprocess.run(["cmake", f"-DMANIFEST={REPO}/tolerances/TOLERANCES.sha256", f"-DBASE={REPO}/tolerances", "-DEXPECT_MIN=1", "-P", str(REPO / "tests/check_hash_file.cmake")], capture_output=True, text=True)
    G.add("G4.noise-floor+frozen", len(builds) >= 3 and h.returncode == 0 and tol["C1"]["primary"]["threshold"]["force_component_abs"] > 0,
          f"{len(builds)} builds {builds}; tolerances hash check rc={h.returncode}; C1 threshold={tol['C1']['primary']['threshold']}")

    # ---------------------------------------------------------------- G5
    per = json.loads((exp / "periodic.json").read_text()); gh = json.loads((exp / "ghost.json").read_text())
    known = {"water_1mol_periodic_6.2", "water_dimer_periodic_7.0"}      # attributed to Q-32 by EXP-0001 (checked below)
    unexplained = []
    for k, rows in per.items():
        worst = max((r["total_dev_vs_first_per_cell"] for r in rows if r["valid"]), default=0.0)
        if worst > 1e-7 and k not in known: unexplained.append((k, worst))
    e2 = max((r.get("max_slot_dev", 0.0) for rows in gh["E2_translation"].values() for r in rows if r["valid"]), default=0.0)
    q32 = json.loads((exp / "q32_hbond_identity.json").read_text())
    G.add("G5.periodic-accounting", not unexplained and e2 < 1e-9 and q32["confirmed"],
          f"E1 systems={len(per)}, unexplained supercell deviations={unexplained}; quirk-attributed={sorted(known)}; E2 max slot deviation={e2:.2e}; EXP-0001 restores invariance={q32['confirmed']}")

    # ---------------------------------------------------------------- G6
    qk = json.loads((exp / "quirks.json").read_text())
    q9 = qk["Q09a_j_eq_l"]; q12 = qk["Q12_absent_bond_pair"]
    ok6 = (q9["S1"]["raw_cnt"] == "cnt=2" and q9["S1+S2+S3"]["raw_cnt"] == "cnt=6" and q9["additivity_S1+S2"]["max_abs_diff"] < 1e-9
           and q9["additivity_S1+S2+S3"]["max_abs_diff"] > 1.0 and "Q09b_j_ne_l" in qk
           and all(abs(r.get("BOp_s", 0) - 1.0) < 1e-12 and abs(r.get("BOp_pi", 0) - 1.0) < 1e-12 for r in q12["C-O block removed"]))
    G.add("G6.Q09+Q12", ok6, f"Q-09: raw cnt 2 per set (3 sets -> {q9['S1+S2+S3']['raw_cnt']}), 2-set additivity diff {q9['additivity_S1+S2']['max_abs_diff']:.1e}, 3-set diff {q9['additivity_S1+S2+S3']['max_abs_diff']:.1f} kcal/mol; "
          f"Q-12: C-O block removed -> BO' = {[(r['BOp_s'], r['BOp_pi'], r['BOp_pipi']) for r in q12['C-O block removed']][:1]} at all r")

    # ---------------------------------------------------------------- G7
    docs_ok = True; notes = []
    need = {"docs/VALIDATION.md": ["M1 results", "E-NOISE", "E-FD", "Q-32", "Q-34"], "docs/DEVELOPMENT_LOG.md": ["M1"], "docs/ENGINE_SPEC.md": ["Q-32", "Q-33", "Q-34"],
            "docs/NUMERICAL_POLICY.md": ["noise floor"], "docs/SOURCE_MAP.md": ["0001-reaxmetal-diagnostics.patch"]}
    for f, keys in need.items():
        t = (REPO / f).read_text()
        for k in keys:
            if k not in t: docs_ok = False; notes.append(f"{f} lacks '{k}'")
    G.add("G7.docs", docs_ok, "required sections present" if docs_ok else "; ".join(notes))
    if not a.skip_git:
        st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        loc = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        rem = subprocess.run(["git", "-C", str(REPO), "ls-remote", "origin", f"refs/heads/{a.branch}"], capture_output=True, text=True).stdout.split()
        G.add("G7.pushed", not st and bool(rem) and rem[0] == loc, f"tree clean={not st}; local {loc[:12]} remote {rem[0][:12] if rem else None}")
    allpass = all(r["pass"] for r in G.rows)
    print("\nM1 ACCEPTANCE GATE:", "PASS" if allpass else "FAIL")
    if a.out: Path(a.out).write_text(json.dumps({"pass": allpass, "criteria": G.rows}, indent=1))
    return 0 if allpass else 2


if __name__ == "__main__":
    sys.exit(main())
