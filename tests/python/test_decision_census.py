#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""CENSUS-1: decision-mismatch census (FP32-1, NUMERICAL_POLICY 5.3). The discrete decisions of the force field (bond-order cut-off, three-body and
four-body gates, hydrogen-bond gate, non-bonded cut-off) are counted per owned atom by CPU-64, by the CPU-32 twin and by Metal-32; every atom whose
count differs from CPU-64 is a decision flip caused by single precision. Measured on the 58 fixtures as given (sigma 0) and on random displacements of
them (sigma 0.05 and 0.15 A, several seeds), which sample the neighbourhood of the thresholds.
Gate: on the fixtures as given there is no flip (Metal and CPU-32); on the displaced ensembles the rates are REPORTED (they are measurements, not a
criterion), with one sanity limit: a flip rate above 1e-3 of the decisions of a category would mean a logic difference rather than a rounding event.
usage: test_decision_census.py --tool BIN --ffield-dir DIR [--samples 4] [--metal]"""
import argparse, json, random, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "reaxref")); sys.path.insert(0, str(HERE))
from test_full_fixtures import case_text  # noqa: E402

CATS = ["bonds", "angle_sets", "torsions", "hbonds", "nonbonded"]


def census(tool, ff, extra, txt, backend):
    r = subprocess.run([tool, "--ffield", str(ff), *extra, "--backend", backend, "--census", str(txt)], capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(r.stderr.strip()[:400])
    out = {}
    for ln in r.stdout.splitlines():
        p = ln.split()
        if p and p[0] == "census": out[p[1]] = [int(v) for v in p[2:]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--samples", type=int, default=4); ap.add_argument("--metal", action="store_true")
    ap.add_argument("--fixtures", default=str(ROOT / "tests" / "fixtures"))
    a = ap.parse_args()
    fx = Path(a.fixtures)
    backends = ["cpu32"] + (["metal"] if a.metal else [])
    rng = random.Random(20261009)
    # stats[(bucket, backend, cat)] = [decisions, flipped atoms, sum |diff|]
    stats = {}
    runs = 0
    with tempfile.TemporaryDirectory() as td:
        for cf in sorted((fx / "cases").glob("*.json")):
            case = json.loads(cf.read_text()); ref = json.loads((fx / "reference" / cf.name).read_text())
            if not ref.get("valid", True): continue
            extra = ["--elements", ",".join(case["elements"])] + (["--lgvdw"] if case.get("pair", {}).get("lgvdw") else [])
            ff = Path(a.ffield_dir) / case["ffield"]["name"]
            for sigma, nsamp in ((0.0, 1), (0.05, a.samples), (0.15, a.samples)):
                for s in range(nsamp):
                    r2 = json.loads(json.dumps(ref))
                    if sigma > 0:
                        r2["positions"] = [[x + rng.gauss(0, sigma) for x in p] for p in ref["positions"]]
                    txt = Path(td) / "case.txt"; txt.write_text(case_text(case, r2))
                    try:
                        base = census(a.tool, ff, extra, txt, "cpu64")
                        others = {b: census(a.tool, ff, extra, txt, b) for b in backends}
                    except RuntimeError as e:
                        print("FAIL", cf.stem, e); return 1
                    runs += 1
                    bucket = "given" if sigma == 0 else f"sigma{sigma}"
                    for b, o in others.items():
                        for c in CATS:
                            st = stats.setdefault((bucket, b, c), [0, 0, 0])
                            st[0] += sum(base[c])
                            for x, y in zip(base[c], o[c]):
                                if x != y: st[1] += 1; st[2] += abs(x - y)
    fails = []
    print(f"{runs} configurations (x {len(backends) + 1} backends)")
    print(f"{'bucket':9s} {'backend':6s} " + "  ".join(f"{c:>26s}" for c in CATS))
    for bucket in ("given", "sigma0.05", "sigma0.15"):
        for b in backends:
            cells = []
            for c in CATS:
                n, fl, ad = stats[(bucket, b, c)]
                cells.append(f"{n:9d} dec {fl:4d} flips {ad:4d}")
                if bucket == "given" and fl > 0: fails.append(f"{b} {c}: {fl} atoms differ from CPU-64 on the fixtures as given")
                if n and fl / n > 1e-3: fails.append(f"{bucket} {b} {c}: flip rate {fl / n:.2e} > 1e-3 (logic difference?)")
            print(f"{bucket:9s} {b:6s} " + "  ".join(f"{x:>26s}" for x in cells))
    for f in fails: print("FAIL", f)
    print("RESULT: PASS" if not fails else "RESULT: FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
