#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Produce the FROZEN tolerance file (tolerances/tolerances.json) and its hash from measured data (NUMERICAL_POLICY 3.2, 5.4).

Inputs are the artefacts of the M1 measurement run; nothing here is tuned by hand:
  C1 thresholds  = CEIL1( 10 x observed noise floor )  (CEIL1 = round up to one significant digit; both numbers recorded)
  C3 thresholds  = the owner-set values of NUMERICAL_POLICY 5.2 (copied verbatim)
  stretch        = NUMERICAL_POLICY 5.5 (copied verbatim)
usage: make_tolerances.py --noise noise_floor.json --conditioning conditioning.json --fd fd_all.json --runs <suite run dir (diag on)>
                          --cases tests/fixtures/cases --out tolerances --patch-sha256 HEX
"""
import argparse, hashlib, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner  # noqa: E402


def ceil1(x):
    if x <= 0: return 0.0
    e = math.floor(math.log10(x)); m = x / 10 ** e
    return float(math.ceil(round(m, 9)) * 10 ** e)


def main():
    ap = argparse.ArgumentParser()
    for k in ("noise", "conditioning", "fd", "runs", "cases", "out", "patch-sha256"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    noise = json.loads(Path(a.noise).read_text()); cond = json.loads(Path(a.conditioning).read_text()); fd = json.loads(Path(a.fd).read_text())
    setB = "B_well_conditioned_pre_registered_plus_proposed_X1_X2"; setA = "A_well_conditioned_pre_registered"
    wB = noise["stats"][setB]["worst"]; wA = noise["stats"][setA]["worst"]

    def c1(w):
        floor = {"energy_slot_rel_to_max1": w["dE_slot_rel"]["value"], "energy_total_per_atom_abs": w["dE_total_per_atom"]["value"],
                 "charge_abs": w["dq_max"]["value"], "force_component_abs": w["dF_max"]["value"], "force_rms": w["dF_rms"]["value"]}
        return {"observed_floor": floor, "ten_x": {k: 10 * v for k, v in floor.items()}, "threshold": {k: ceil1(10 * v) for k, v in floor.items()}}

    # QEq / neutrality / residual statistics over every valid fixture of the reference run
    res_eV, sumq, rel_s = [], [], []
    for d in sorted(Path(a.runs).iterdir()):
        f = d / "result.json"
        if not f.exists(): continue
        r = json.loads(f.read_text())
        if r.get("independent_eem"): res_eV.append(r["independent_eem"]["equalization_residual_eV"]); sumq.append(abs(r["sum_q"]))
        if r.get("qeq"): rel_s.append(max(r["qeq"]["final_rel_residual_s"], r["qeq"]["final_rel_residual_t"]))
    # finite-difference statistics (fixed-q), well-conditioned (extended) fixtures only
    tags = {c: json.loads((Path(a.cases) / (c + ".json")).read_text()).get("tags", []) for c in fd}
    quirk = sorted(c for c in fd if "known-reference-quirk" in tags[c])
    fd_good = [(fd[c]["fixedq"]["max_abs_err_h"], c) for c in fd if "error" not in fd[c] and cond.get(c, {}).get("well_conditioned_extended") and c not in quirk]
    fd_max = max(fd_good)
    fd_quirk = {c: fd[c]["fixedq"]["max_abs_err_h"] for c in quirk}
    t = {
        "schema": "reaxmetal.tolerances/1",
        "frozen_at": "M1 (before any engine or Metal result exists)",
        "rule": "Changing this file requires an owner-approved DEVELOPMENT_LOG entry stating a physical/numerical reason (NUMERICAL_POLICY 5.4). "
                "A mismatch is never closed by editing a tolerance or a reference.",
        "pins": {"lammps_commit": "8de817dd79bfe4525d5d39246a212d833e6dee07", "instrumentation_patch_sha256": a.patch_sha256},
        "fixtures": {"n_total": len(cond), "cases_dir_hash_file": "tests/fixtures/FIXTURES.sha256"},
        "conditioning": {
            "pre_registered_definition": "NUMERICAL_POLICY 5.2 (W1-W5 as implemented in tools/reaxref/conditioning.py)",
            "proposed_amendment_pending_owner_confirmation": {
                "X1_NEAR_LINEAR": "valence-angle triple with sin(theta) < 1e-3 (pinned kernel divides by sin(theta) clamped at 1e-5; ENGINE_SPEC Q-33)",
                "X2_PRODUCT": "BO_ij*BO_jk within 1e-3 (relative) of thb_cutsq (energy discontinuity; VALIDATION E-FD-2)"},
            "classification": {c: {"pre_registered_well_conditioned": v["well_conditioned_pre_registered"], "extended_well_conditioned": v["well_conditioned_extended"],
                                   "fails_pre_registered": v["fails_pre_registered"], "extended_flags": v["extended_flags"]} for c, v in sorted(cond.items())},
            "counts": {"pre_registered_well_conditioned": sum(1 for v in cond.values() if v["well_conditioned_pre_registered"]),
                       "extended_well_conditioned": sum(1 for v in cond.values() if v["well_conditioned_extended"])}},
        "noise_floor": {"builds": noise["builds"], "pairwise_sets": {k: len(v) for k, v in noise["sets"].items()}},
        "C1": {
            "role": "CPU-64 vs pinned LAMMPS: strict parity at 10x the measured reference noise floor",
            "basis_primary": f"set {setB} (pre-registered well-conditioned + proposed X1/X2)",
            "primary": c1(wB),
            "alternative_if_amendment_rejected": {"basis": f"set {setA} (pre-registered definition only)", **c1(wA),
                                                  "note": "dominated by the exactly collinear H-bond fixtures (|dF| ~ 1e-2 between FMA and non-FMA builds): a threshold derived this way would not detect real errors"},
            "metric_form": "energy: |d| <= rtol*max(1,|ref|) per slot; |dE|/N <= energy_total_per_atom_abs; charge: |dq| <= charge_abs; force: per-component |dF| <= force_component_abs and RMS over atoms <= force_rms",
            "validity_range": "measured on fixtures of <= 216 atoms; re-measure before applying to larger systems (floor grows with system size)"},
        "C2": {"role": "CPU-32 vs CPU-64: characterisation only (error distributions), not a gate"},
        "C3": {"role": "Metal-32 vs CPU-64 acceptance on well-conditioned fixtures (owner-set, NUMERICAL_POLICY 5.2)",
               "energy_per_atom_abs_kcal_mol": 1e-3, "energy_significant_category_rel": 1e-5,
               "force_rms_kcal_mol_A": 5e-3, "force_max_component_kcal_mol_A": 5e-2, "charge_max_abs_e": 1e-4, "charge_rms_e": 2e-5,
               "neutrality_and_equalization_residual": {
                   "status": "PROPOSED -- awaiting owner confirmation (the owner left these to the M1 reference check)",
                   "sum_q_abs_per_atom_e": 2e-5, "equalization_residual_eV": 2e-3,
                   "basis": "|sum q|/N <= the owner's charge-RMS bound (2e-5 e); residual <= owner's max |dq| bound (1e-4 e) x ~20 eV/e (largest diagonal EEM hardness in the bundled force fields). "
                            "NOT derived from the FP64 reference noise, which is ~1e-15 / ~1e-10 eV and would be meaningless for an FP32 backend."},
               "md_nve_drift": "provisional (NUMERICAL_POLICY 5.2); not frozen numerically here"},
        "stretch_targets": {"source": "NUMERICAL_POLICY 5.5 (original M0.5 proposal)", "reported_not_gated": True},
        "reference_validity": {
            "thresholds_set_a_priori": {"equalization_residual_eV": 1e-8, "abs_sum_q_e": 1e-9, "note": "runner defaults chosen before any data; observed maxima below are 10-1000x smaller"},
            "qeq_reference_tolerance": 1e-12, "qeq_reference_maxiter": 500,
            "max_final_recursive_relative_residual_observed": max(rel_s),
            "independent_equalization_residual_eV_max_observed": max(res_eV), "abs_sum_q_max_observed": max(sumq),
            "rule": "a reference calculation is VALID only if LAMMPS emitted no unexpected warning, the fix's recursive relative residual is <= the requested tolerance, "
                    "the independent equalization residual and |sum q| are within the a-priori thresholds above, and exactly one force evaluation occurred. "
                    "Invalid calculations never enter the golden dataset."},
        "finite_difference": {"scope": "CPU-64 analytic forces vs central FD of the energy (kept separate from GPU tolerances, NUMERICAL_POLICY 5.2/6)",
                              "protocol": "fixed-q central difference at h=1e-5 and 2e-5 A; components whose two estimates differ by > 1e-5 + 1e-6|F| are 'discontinuity suspects' and excluded from the smooth-point statistic",
                              "observed_max_abs_err_pinned_lammps_well_conditioned_fixtures": fd_max[0], "worst_case": fd_max[1],
                              "known_reference_quirk_fixtures_excluded_from_that_statistic": fd_quirk,
                              "note": "the excluded fixtures reproduce ENGINE_SPEC Q-34 (analytic force != gradient of energy for heavy atoms with pi bonds) and Q-32 (H-bond self-image)"},
    }
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(t, indent=1, sort_keys=True) + "\n"
    (out / "tolerances.json").write_text(txt)
    h = hashlib.sha256(txt.encode()).hexdigest()
    (out / "TOLERANCES.sha256").write_text(f"{h}  tolerances.json\n")
    print("tolerances.json sha256", h)
    print(json.dumps({"C1 primary threshold": t["C1"]["primary"]["threshold"], "C3 neutrality": t["C3"]["neutrality_and_equalization_residual"]}, indent=1))


if __name__ == "__main__":
    main()
