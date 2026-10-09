#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Structure analysis of the carbon dumps of in.graphitization (id x y z per atom): coordination number distribution (bond = C-C distance < --rcut, default 1.85 A),
mean coordination, fraction of sp2-like (3-coordinated) atoms, mean bond length, and the smallest-ring statistics (rings of 5, 6, 7 members through each bond).
usage: analyze.py --box 84.26481 dump1 dump2 ..."""
import argparse
from pathlib import Path
from collections import Counter, deque
import numpy as np
from scipy.spatial import cKDTree


def read(path):
    L = Path(path).read_text().splitlines()
    k = [i for i, l in enumerate(L) if l.startswith("ITEM: ATOMS")][0] + 1
    step = int(L[1])
    a = np.array([[float(x) for x in l.split()] for l in L[k:] if l.strip()])
    a = a[np.argsort(a[:, 0])]
    return step, a[:, 1:4]


def smallest_ring(adj, i, j, maxlen=8):
    """length of the smallest ring through bond i-j (BFS from i to j without the bond), or 0 if none up to maxlen"""
    seen = {i: 0}; q = deque([i])
    while q:
        u = q.popleft()
        if seen[u] >= maxlen - 1: continue
        for w in adj[u]:
            if (u == i and w == j): continue
            if w == j: return seen[u] + 2
            if w not in seen: seen[w] = seen[u] + 1; q.append(w)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+"); ap.add_argument("--box", type=float, required=True); ap.add_argument("--rcut", type=float, default=1.85)
    a = ap.parse_args()
    print(f"{'step':>8s} {'<CN>':>6s} {'CN=1':>6s} {'CN=2':>6s} {'CN=3':>6s} {'CN=4':>6s} {'CN>=5':>6s} {'sp2 frac':>8s} {'<bond A>':>9s} {'5-rings':>8s} {'6-rings':>8s} {'7-rings':>8s}")
    for d in sorted(a.dumps, key=lambda p: read(p)[0]):
        step, x = read(d)
        x = x % a.box
        tree = cKDTree(x, boxsize=a.box)
        pairs = tree.query_pairs(a.rcut, output_type="ndarray")
        n = len(x)
        cn = np.bincount(pairs.ravel(), minlength=n)
        dv = x[pairs[:, 0]] - x[pairs[:, 1]]; dv -= a.box * np.round(dv / a.box)
        bl = np.linalg.norm(dv, axis=1)
        adj = [[] for _ in range(n)]
        for i, j in pairs: adj[i].append(j); adj[j].append(i)
        rings = Counter(smallest_ring(adj, int(i), int(j)) for i, j in pairs[: min(len(pairs), 20000)])
        scale = len(pairs) / max(1, min(len(pairs), 20000))
        frac = lambda k: np.mean(cn == k)
        print(f"{step:8d} {cn.mean():6.3f} {frac(1):6.3f} {frac(2):6.3f} {frac(3):6.3f} {frac(4):6.3f} {np.mean(cn >= 5):6.3f} {frac(3):8.3f} {bl.mean():9.4f} "
              f"{rings[5] * scale / 5:8.0f} {rings[6] * scale / 6:8.0f} {rings[7] * scale / 7:8.0f}")


if __name__ == "__main__":
    from pathlib import Path
    main()
