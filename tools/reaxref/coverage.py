#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Coverage of the fixture set: which pvector slots / diagnostics tallies / special branches are exercised by VALID results.
usage: coverage.py <run-dir> [--require]   (exit 2 if --require and any required item is uncovered)"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import SLOT_NAMES  # noqa: E402

ACTIVE_SLOTS = [s for s in SLOT_NAMES if s not in ("emol", "efi")]    # emol, efi are constant 0 in pinned LAMMPS
THRESH = 1e-6

# tally keys that must have count>0 in at least one valid result
REQUIRED_TALLIES = ["bond.oo", "bond.og", "bond.self", "bond_triple_stab.oo", "angle.g0", "angle.g1", "angle.g2",
                    "penalty.g0", "coalition.g0", "torsion.g0", "torsion.g1", "torsion.g2", "conjugation.g0", "hbond.g0",
                    "vdw.oo", "vdw.og", "vdw.self", "coulomb.oo", "coulomb.og", "coulomb.self", "lonepair_extra_C2.oo"]


def main():
    run = Path(sys.argv[1]); require = "--require" in sys.argv
    res = {}
    for d in sorted(run.iterdir()):
        f = d / "result.json"
        if f.exists():
            r = json.loads(f.read_text())
            if r.get("valid"): res[d.name] = r
    slot_cases = {s: [] for s in ACTIVE_SLOTS}
    tally_cases = {}
    for cid, r in res.items():
        for s in ACTIVE_SLOTS:
            if abs(r["energy"]["slots"][s]) > THRESH: slot_cases[s].append(cid)
        for k, v in r.get("diag", {}).get("tallies", {}).items():
            if v["count"] > 0: tally_cases.setdefault(k, []).append(cid)
    miss = []
    print(f"valid cases: {len(res)}")
    print("== pvector slots with |value|>1e-6 ==")
    for s in ACTIVE_SLOTS:
        print(f"  {s:5s} {len(slot_cases[s]):3d} cases  e.g. {slot_cases[s][:3]}")
        if not slot_cases[s]: miss.append("slot " + s)
    print("== diagnostic tallies (term.class) ==")
    for k in sorted(tally_cases): print(f"  {k:22s} {len(tally_cases[k]):3d} cases")
    for k in REQUIRED_TALLIES:
        if k not in tally_cases: miss.append("tally " + k)
    print("== uncovered required items ==")
    print("  " + (", ".join(miss) if miss else "none"))
    return 2 if (miss and require) else 0


if __name__ == "__main__":
    sys.exit(main())
