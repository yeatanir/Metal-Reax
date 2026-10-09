#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Random packing of N carbon atoms in a periodic cubic box of a given mass density (default 30000 atoms, 1.0 g/cm^3), as a LAMMPS data file (atom_style charge,
all charges 0). Atoms are placed uniformly at random; any pair closer than --dmin (Angstrom) is resolved by re-placing one of the two atoms (random sequential
addition), so the structure is random but free of overlaps that would explode a reactive force field.
usage: make_box.py out.data [--n 30000] [--density 1.0] [--dmin 1.2] [--seed 1]"""
import argparse
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--n", type=int, default=30000); ap.add_argument("--density", type=float, default=1.0)
ap.add_argument("--dmin", type=float, default=1.2); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
AMU_G = 1.66053906660e-24
mass = 12.0107
L = (a.n * mass * AMU_G / a.density * 1e24) ** (1.0 / 3.0)   # Angstrom^3 = 1e-24 cm^3
rng = np.random.default_rng(a.seed)
x = rng.random((a.n, 3)) * L
for it in range(1000):
    pairs = cKDTree(x, boxsize=L).query_pairs(a.dmin, output_type="ndarray")
    if len(pairs) == 0: break
    bad = np.unique(pairs[:, 1])
    x[bad] = rng.random((len(bad), 3)) * L
else:
    raise SystemExit("could not remove all overlaps")
d = cKDTree(x, boxsize=L).query(x, k=2)[0][:, 1]
print(f"N {a.n}  box {L:.5f} A  density {a.n * mass * AMU_G / (L ** 3 * 1e-24):.5f} g/cm3  rounds {it}  nearest neighbour min {d.min():.3f} mean {d.mean():.3f} A")
with open(a.out, "w") as f:
    f.write(f"random carbon, {a.n} atoms, {a.density} g/cm3\n\n{a.n} atoms\n1 atom types\n\n0 {L:.8f} xlo xhi\n0 {L:.8f} ylo yhi\n0 {L:.8f} zlo zhi\n\nMasses\n\n1 {mass}\n\nAtoms # charge\n\n")
    for i, p in enumerate(x):
        f.write(f"{i + 1} 1 0.0 {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
