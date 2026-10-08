#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""NBR-3: the engine's pair enumeration reproduces the interaction tallies recorded from pinned LAMMPS in M1.

For every fixture the geometry is expanded with the standalone image expander, the CPU-64 far list is built with
nonb_cut taken from the fixture's force field, and the owner-computes counting rules are applied; the resulting
(oo, og, self) counts must equal the reference tallies vdw.oo / vdw.og / vdw.self (and coulomb.*, which the reference
counts in the same loop) of LAMMPS' own run. Missing tally keys mean zero.
usage: test_neighbor_fixtures.py --tool BIN --ffield-dir DIR [--fixtures tests/fixtures] [--tmp DIR]"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "reaxref"))
import runner  # noqa: E402


def case_text(case, ref):
    lo, hi, tilt = (tuple(float(v) for v in t) for t in runner.cell_to_lammps(case["cell"]))
    a = [hi[0] - lo[0], 0.0, 0.0]; b = [tilt[0], hi[1] - lo[1], 0.0]; c = [tilt[1], tilt[2], hi[2] - lo[2]]
    per = case["periodic"]
    out = [f"origin {lo[0]!r} {lo[1]!r} {lo[2]!r}", f"a {a[0]!r} {a[1]!r} {a[2]!r}", f"b {b[0]!r} {b[1]!r} {b[2]!r}",
           f"c {c[0]!r} {c[1]!r} {c[2]!r}", "periodic " + " ".join("1" if p else "0" for p in per), f"atoms {len(ref['positions'])}"]
    for i, (p, t) in enumerate(zip(ref["positions"], ref["types"])):
        out.append(f"{i + 1} {t} {p[0]!r} {p[1]!r} {p[2]!r}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--fixtures", default=str(HERE.parents[1] / "tests" / "fixtures"))
    a = ap.parse_args()
    fx = Path(a.fixtures)
    failures, n = [], 0
    with tempfile.TemporaryDirectory() as td:
        for cf in sorted((fx / "cases").glob("*.json")):
            case = json.loads(cf.read_text()); ref = json.loads((fx / "reference" / cf.name).read_text())
            if not ref.get("valid", True):
                continue
            txt = Path(td) / "case.txt"; txt.write_text(case_text(case, ref))
            extra = ["--lgvdw"] if case.get("pair", {}).get("lgvdw") else []
            r = subprocess.run([a.tool, "--ffield", str(Path(a.ffield_dir) / case["ffield"]["name"]), *extra, str(txt)], capture_output=True, text=True)
            if r.returncode != 0:
                failures.append(f"{cf.stem}: tool failed: {r.stderr.strip()}"); continue
            got = {}
            for ln in r.stdout.splitlines():
                k, *v = ln.split()
                got[k] = [float(t) for t in v]
            oo, og, sf = (int(t) for t in got["vdw"])
            t = ref["tallies"]
            want = (t.get("vdw.oo", 0), t.get("vdw.og", 0), t.get("vdw.self", 0))
            cwant = (t.get("coulomb.oo", 0), t.get("coulomb.og", 0), t.get("coulomb.self", 0))
            n += 1
            if (oo, og, sf) != want or want != cwant:
                failures.append(f"{cf.stem}: engine (oo,og,self)={(oo, og, sf)} reference vdw={want} coulomb={cwant}")
    for f in failures:
        print("FAIL", f)
    print(f"NBR-3: {n} fixtures compared, {len(failures)} mismatches")
    print("RESULT: PASS" if not failures and n >= 50 else f"RESULT: FAIL ({len(failures)} mismatches, {n} compared)")
    return 0 if not failures and n >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
