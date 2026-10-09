#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""CENSUS-2: threshold scans (FP32-1, THR-n of the test matrix). For small systems with one geometric parameter d (a C-C pair; a C-C-C and a C-C-C-C chain
whose last bond is stretched; a water dimer whose O...H distance is stretched) the CPU-64 decision census is evaluated on a grid; wherever it changes,
the crossing d* is found by bisection to double precision. Around each crossing the decisions of the CPU-32 twin and of Metal-32 are compared with CPU-64
at d* +- delta for delta = 1e-3 ... 1e-9 A: the flip WINDOW of single precision (the largest delta that still gives a different decision) is reported,
together with the energy jump of the discontinuity itself (CPU-64, E(d*+1e-9) - E(d*-1e-9)).
Gate: no flip at |delta| >= 1e-4 A (a flip further from the threshold than FP32 resolution of a ~5 A distance, 5e-7 A, would be a logic difference).
usage: test_threshold_scan.py --tool BIN --ffield-dir DIR [--metal]"""
import argparse, math, subprocess, sys, tempfile
from pathlib import Path

CATS = ["bonds", "angle_sets", "torsions", "hbonds", "nonbonded"]
TERMS = ["e_bond", "e_lp", "e_ov", "e_un", "e_ang", "e_pen", "e_coa", "e_tor", "e_con", "e_hb", "e_vdW", "e_ele", "e_pol"]


def text(types, pos, q):
    out = ["origin -30 -30 -30", "a 80 0 0", "b 0 80 0", "c 0 0 80", "periodic 0 0 0", f"atoms {len(types)}"]
    out += [f"{i + 1} {t} {p[0]!r} {p[1]!r} {p[2]!r}" for i, (t, p) in enumerate(zip(types, pos))]
    return "\n".join(out) + "\ncharges\n" + " ".join(repr(v) for v in q) + "\n"


def evaluate(tool, ff, txt, backend):
    r = subprocess.run([tool, "--ffield", str(ff), "--elements", "C,H,O", "--backend", backend, "--census", str(txt)], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[:300])
    cen, en = {}, 0.0
    for ln in r.stdout.splitlines():
        p = ln.split()
        if p[0] == "census": cen[p[1]] = tuple(int(v) for v in p[2:])
        elif p[0] in TERMS: en += float(p[1])
    return cen, en


def families():
    th = math.radians(109.5)
    def pair(d): return [1, 1], [[0, 0, 0], [d, 0, 0]], [0.1, -0.1]
    def chain3(d): return [1, 1, 1], [[0, 0, 0], [1.5, 0, 0], [1.5 + d * math.cos(math.pi - th), d * math.sin(math.pi - th), 0]], [0.0, 0.0, 0.0]
    def chain4(d): return [1, 1, 1, 1], [[0, 0, 0], [1.5, 0, 0], [1.5 + 1.5 * math.cos(math.pi - th), 1.5 * math.sin(math.pi - th), 0],
                                           [1.5 + 1.5 * math.cos(math.pi - th) + d, 1.5 * math.sin(math.pi - th), 0.4]], [0.0] * 4
    def dimer(d):   # O-H bonded water (O at origin, H1 0.96 A, H2) and a second molecule's O at O...H distance d along the O-H axis
        return [3, 2, 2, 3, 2, 2], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [0.96 + d, 0, 0], [0.96 + d + 0.96, 0, 0], [0.96 + d - 0.24, 0.93, 0]], [-0.8, 0.4, 0.4, -0.8, 0.4, 0.4]
    return [("C-C pair", pair, 0.8, 10.6), ("C-C-C, last bond", chain3, 1.0, 4.5), ("C-C-C-C, last bond", chain4, 1.0, 4.5), ("water dimer O...H", dimer, 1.2, 8.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--ffield-dir", required=True); ap.add_argument("--metal", action="store_true")
    a = ap.parse_args()
    ff = Path(a.ffield_dir) / "ffield.reax.cho"
    backends = ["cpu32"] + (["metal"] if a.metal else [])
    fails, rows = [], []
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "c.txt"
        def run(fam, d, b):
            f.write_text(text(*fam(d)))
            return evaluate(a.tool, ff, f, b)
        for name, fam, d0, d1 in families():
            n = int(round((d1 - d0) / 0.05))
            grid = [d0 + k * (d1 - d0) / n for k in range(n + 1)]
            prev = None
            for d in grid:
                cen, _ = run(fam, d, "cpu64")
                if prev is not None and cen != prev[1]:
                    lo, hi = prev[0], d
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        if run(fam, mid, "cpu64")[0] == prev[1]: lo = mid
                        else: hi = mid
                    ds = 0.5 * (lo + hi)
                    changed = [c for c in CATS if cen[c] != prev[1][c]]
                    jump = run(fam, ds + 1e-9, "cpu64")[1] - run(fam, ds - 1e-9, "cpu64")[1]
                    window = {b: 0.0 for b in backends}
                    for k in range(3, 10):
                        for sgn in (-1, 1):
                            dd = ds + sgn * 10.0 ** -k
                            base = run(fam, dd, "cpu64")[0]
                            for b in backends:
                                if run(fam, dd, b)[0] != base: window[b] = max(window[b], 10.0 ** -k)
                    rows.append((name, ds, "+".join(changed), jump, window))
                    for b in backends:
                        if window[b] >= 1e-4: fails.append(f"{name} d*={ds:.9f}: {b} takes a different decision {window[b]:.0e} A from the threshold")
                prev = (d, cen)
    print(f"{'system':20s} {'d* (A)':>13s} {'decision':>22s} {'energy jump kcal/mol':>22s}   flip window (A) " + "  ".join(backends))
    for name, ds, ch, jump, window in rows:
        print(f"{name:20s} {ds:13.9f} {ch:>22s} {jump:22.3e}   " + "  ".join(f"{window[b]:.0e}" if window[b] else "none" for b in backends))
    for x in fails: print("FAIL", x)
    print(f"CENSUS-2: {len(rows)} threshold crossings, {len(fails)} failures")
    print("RESULT: PASS" if rows and not fails else "RESULT: FAIL")
    return 0 if rows and not fails else 1


if __name__ == "__main__":
    sys.exit(main())
