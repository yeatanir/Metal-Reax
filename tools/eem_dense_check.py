#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""M0.5 exploratory check (NOT a validation test): is LAMMPS `fix qeq/reaxff` equal to an independent dense EEM solve
with *explicit periodic images* in a cell smaller than the taper cutoff?

LAMMPS documents (doc/src/fix_qeq_reaxff.rst, Restrictions): the fix "does not correctly handle interactions involving
multiple periodic images of the same atom ... should not be used for periodic cell dimensions smaller than the
non-bonded cutoff radius". This script measures what that means numerically for the LAMMPS regression geometry
(7.54 A cell, taper swb = 8.0 A).

Model, from fix_qeq_reaxff.cpp:744-760 and 851-880 (ENGINE_SPEC section 7) -- written independently here:
  A_ii = eta_i + sum_{n != 0} H(|nL|)            (self-image pairs)
  A_ij = sum_n H(|x_j + nL - x_i|)               (i != j; all shifts, r <= swb)
  H(r) = 14.4 * Tap(r) / (r^3 + (g_i g_j)^-1.5)^(1/3),  Tap = 1 - 35x^4 + 84x^5 - 70x^6 + 20x^7, x = r/swb
  minimise  sum chi q + 1/2 q A q   subject to sum q = 0.
usage: eem_dense_check.py <dump with id type q x y z> <ffield> <swb> <lattice const> <nrep>
"""
import sys
import numpy as np


def read_ffield(path, elems):
    L = open(path).read().splitlines()
    i = 1
    ng = int(L[i].split()[0]); i += 1 + ng
    nat = int(L[i].split()[0]); i += 4
    out = {}
    for _ in range(nat):
        b = L[i:i + 4]; i += 4
        w0, w1 = b[0].split(), b[1].split()
        out[w0[0].upper()] = dict(gamma=float(w0[6]), chi=float(w1[5]), eta=2.0 * float(w1[6]))
    return [out[e] for e in elems]


def main():
    dump, ff, swb, a0, nrep = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    lines = open(dump).read().splitlines()
    k = [n for n, l in enumerate(lines) if l.startswith("ITEM: ATOMS")][0]
    rows = np.array([[float(v) for v in l.split()] for l in lines[k + 1:] if l.strip()])
    rows = rows[np.argsort(rows[:, 0])]
    typ = rows[:, 1].astype(int); q_lmp = rows[:, 2]; x = rows[:, 3:6]
    box = 4.0 * 0 + a0 * nrep  # diamond cubic conventional cell edge = a0
    par = read_ffield(ff, ["H", "C", "O"])                     # types 1,2,3 as in tests/lammps/in.common_setup
    chi = np.array([par[t - 1]["chi"] for t in typ]); eta = np.array([par[t - 1]["eta"] for t in typ])
    gam = np.array([par[t - 1]["gamma"] for t in typ])
    n = len(typ)
    rng = int(np.ceil(swb / box)) + 1
    shifts = [(a, b, c) for a in range(-rng, rng + 1) for b in range(-rng, rng + 1) for c in range(-rng, rng + 1)]

    def tap(r):
        t = r / swb
        return 1 - 35 * t**4 + 84 * t**5 - 70 * t**6 + 20 * t**7

    A = np.diag(eta).astype(float)
    n_img_pairs = 0
    for i in range(n):
        for j in range(n):
            shield = (gam[i] * gam[j]) ** -1.5
            for s in shifts:
                if i == j and s == (0, 0, 0): continue
                d = x[j] + box * np.array(s) - x[i]
                r = np.sqrt(d @ d)
                if r <= swb:
                    A[i, j] += 14.4 * tap(r) / (r**3 + shield) ** (1.0 / 3.0)
                    n_img_pairs += (s != (0, 0, 0))
    ones = np.ones(n)
    sol_s = np.linalg.solve(A, -chi); sol_t = np.linalg.solve(A, -ones)
    q = sol_s - (sol_s.sum() / sol_t.sum()) * sol_t
    print("atoms=%d  shifts per pair=%d  directed image-pair terms (r<=swb, shift!=0)=%d" % (n, len(shifts), n_img_pairs))
    print("max|q_dense - q_lammps| = %.3e   (max|q| = %.3e, sum q_dense = %.2e)" % (np.abs(q - q_lmp).max(), np.abs(q).max(), q.sum()))
    # same model but with MINIMUM IMAGE only, to show what an engine that ignores multiple images would get
    A2 = np.diag(eta).astype(float)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            shield = (gam[i] * gam[j]) ** -1.5
            d = x[j] - x[i]; d -= box * np.round(d / box)
            r = np.sqrt(d @ d)
            if r <= swb: A2[i, j] += 14.4 * tap(r) / (r**3 + shield) ** (1.0 / 3.0)
    s2 = np.linalg.solve(A2, -chi); t2 = np.linalg.solve(A2, -ones); q2 = s2 - (s2.sum() / t2.sum()) * t2
    print("minimum-image-only model: max|q_minimg - q_lammps| = %.3e" % np.abs(q2 - q_lmp).max())


if __name__ == "__main__":
    main()
