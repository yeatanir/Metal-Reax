#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Compare two suite runs (run_suite.py output dirs) case by case.
  --bitwise : require every energy slot, total, charge and force to be bit-identical (instrumentation-equivalence test)
  default   : report max abs/rel deviations (noise-floor measurement)
Both modes write a JSON report with --json. Exit 2 if --bitwise and anything differs, or a case is missing/invalid in either."""
import json, sys
from pathlib import Path
import numpy as np

SLOTS = ["eb", "ea", "elp", "emol", "ev", "epen", "ecoa", "ehb", "et", "eco", "ew", "ep", "efi", "eqeq"]


def load(d):
    out = {}
    for c in sorted(Path(d).iterdir()):
        f = c / "result.json"
        if f.exists(): out[c.name] = json.loads(f.read_text())
    return out


def vec(r):
    return (np.array([r["energy"]["total_pe"]]), np.array([r["energy"]["slots"][s] for s in SLOTS]),
            np.array(r["charges"]), np.array(r["forces"]))


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    bitwise = "--bitwise" in sys.argv
    A, B = load(a), load(b)
    rep = {"a": str(a), "b": str(b), "cases": {}, "missing": sorted(set(A) ^ set(B))}
    bad = list(rep["missing"])
    for cid in sorted(set(A) & set(B)):
        ra, rb = A[cid], B[cid]
        if "energy" not in ra or "energy" not in rb:
            bad.append(cid + " (no result)"); continue
        ea, sa, qa, fa = vec(ra); eb_, sb, qb, fb = vec(rb)
        row = {"dE_total": float(abs(ea - eb_)[0]), "dE_slot_max": float(np.abs(sa - sb).max()), "dq_max": float(np.abs(qa - qb).max()),
               "dF_max": float(np.abs(fa - fb).max()), "dF_rms": float(np.sqrt(((fa - fb) ** 2).mean())),
               "natoms": len(qa), "E_total": float(ea[0]), "F_rms": float(np.sqrt((fa ** 2).mean())), "F_max": float(np.abs(fa).max())}
        row["bitwise_identical"] = bool(np.array_equal(ea, eb_) and np.array_equal(sa, sb) and np.array_equal(qa, qb) and np.array_equal(fa, fb))
        rep["cases"][cid] = row
        if bitwise and not row["bitwise_identical"]: bad.append(cid)
    n = len(rep["cases"]); nb = sum(1 for r in rep["cases"].values() if r["bitwise_identical"])
    rep["summary"] = {"cases_compared": n, "bitwise_identical": nb,
                      "max_dE_total": max((r["dE_total"] for r in rep["cases"].values()), default=0.0),
                      "max_dE_slot": max((r["dE_slot_max"] for r in rep["cases"].values()), default=0.0),
                      "max_dq": max((r["dq_max"] for r in rep["cases"].values()), default=0.0),
                      "max_dF": max((r["dF_max"] for r in rep["cases"].values()), default=0.0)}
    if "--json" in sys.argv: Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(rep, indent=1))
    print(f"{a.name} vs {b.name}: {nb}/{n} bit-identical; max dE_slot={rep['summary']['max_dE_slot']:.3e} dq={rep['summary']['max_dq']:.3e} dF={rep['summary']['max_dF']:.3e}"
          + (f"  MISMATCH: {bad[:8]}" if bad else ""))
    return 2 if (bitwise and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
