#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Write the committed golden reference results (tests/fixtures/reference/<case>.json) from a validated suite run.
Only VALID results are accepted (QEq converged to tolerance, independent equalization residual within threshold, no
unexpected warnings): an invalid calculation never enters the golden dataset (owner decision, M1).
usage: make_golden.py <run-dir> <cases-dir> <conditioning.json> <out-dir> --build-desc "text" --patch-sha256 HEX"""
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("cases"); ap.add_argument("conditioning"); ap.add_argument("out")
    ap.add_argument("--build-desc", required=True); ap.add_argument("--patch-sha256", required=True)
    a = ap.parse_args()
    cond = json.loads(Path(a.conditioning).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted(Path(a.run).iterdir()):
        f = d / "result.json"
        if not f.exists(): continue
        r = json.loads(f.read_text())
        if not r.get("valid"): raise SystemExit(f"refusing to write golden for invalid result {d.name}: {r.get('invalid_reasons')}")
        case = json.loads((Path(a.cases) / (d.name + ".json")).read_text())
        g = {"schema": "reaxmetal.golden/1", "case_id": d.name, "case_sha256": hashlib.sha256(runner.canonical_json(case)).hexdigest(), "valid": True,
             "provenance": {"lammps_pinned_commit": r["provenance"]["lammps_pinned_commit"], "ffield": r["provenance"]["ffield"],
                            "instrumentation_patch": "third_party/lammps/patches/0001-reaxmetal-diagnostics.patch", "instrumentation_patch_sha256": a.patch_sha256,
                            "build": a.build_desc, "lammps_binary_sha256": r["provenance"]["lammps_binary_sha256"],
                            "generator": "tools/reaxref/runner.py via run_suite.py"},
             "charge_settings": r["charge_settings"], "qeq": r.get("qeq"), "independent_eem": r.get("independent_eem"),
             "energy": r["energy"], "energy_data_fields": r["diag"]["energy_fields"], "sum_q": r["sum_q"], "force_sum": r["force_sum"],
             "types": r["types"], "positions": r["positions"], "charges": r["charges"], "forces": r["forces"],
             "tallies": {k: v["count"] for k, v in r["diag"]["tallies"].items()},
             "lammps_warnings_expected": r.get("lammps_warnings_expected", []),
             "conditioning": {k: cond[d.name][k] for k in ("well_conditioned_pre_registered", "well_conditioned_extended", "fails_pre_registered", "extended_flags")}}
        (out / (d.name + ".json")).write_text(json.dumps(g, indent=1, sort_keys=True) + "\n")
        n += 1
    print(f"wrote {n} golden references to {out}")


if __name__ == "__main__":
    main()
