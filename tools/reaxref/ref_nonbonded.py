#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Independent numpy reference for the NON-BONDED ReaxFF terms (vdW incl. inner wall / lg, shielded Coulomb, EEM
self-energy) with EXPLICIT periodic images and a unique-pair counting rule, written from the formulas in ENGINE_SPEC
section 6 (not by importing LAMMPS code). It deliberately knows nothing about ghost atoms: it enumerates every distinct
unordered pair (i, j, lattice shift n) once, so agreement with the LAMMPS energy and with LAMMPS' own pair-count
tallies verifies the owned/ghost counting rule (ENGINE_SPEC section 3).

Reproduced reference quirks (so that agreement is meaningful; each is a pinned-LAMMPS behaviour, catalogue Q-xx):
  * C_ele = 332.06371 and the Coulomb cube-root exponent 0.33333333333333 (truncated)
  * e_pol uses KCALpMOL_to_EV = 23.02 (not 23.060549)
  * the taper polynomial coefficients are re-derived here by solving the defining linear system (independent of the
    closed forms in reaxff_init_md.cpp).
"""
import numpy as np

C_ELE = 332.06371
KCAL_PER_EV_LAMMPS = 23.02
CBRT_EXP = 0.33333333333333


def taper_poly(swa, swb):
    """Degree-7 polynomial T with T(swa)=1, T(swb)=0 and T', T'', T''' = 0 at both ends. Returns coefficients c[0..7] (c[k] r^k)."""
    A = []; b = []
    for r, val in ((swa, 1.0), (swb, 0.0)):
        for d in range(4):
            row = []
            for k in range(8):
                if k < d: row.append(0.0)
                else:
                    f = 1.0
                    for m in range(d): f *= (k - m)
                    row.append(f * r ** (k - d))
            A.append(row); b.append(val if d == 0 else 0.0)
    return np.linalg.solve(np.array(A), np.array(b))


def image_shifts(cell_mat, periodic, rcut, extent):
    """Integer shifts that can bring any atom within rcut of any other: |n| range from perpendicular cell heights and
    the spatial extent of the atom cloud along each lattice direction."""
    vol = abs(np.linalg.det(cell_mat))
    a, b, c = cell_mat
    h = [vol / np.linalg.norm(np.cross(b, c)), vol / np.linalg.norm(np.cross(c, a)), vol / np.linalg.norm(np.cross(a, b))]
    rng = [int(np.ceil((rcut + extent) / h[k])) + 1 if periodic[k] else 0 for k in range(3)]
    return [(i, j, k) for i in range(-rng[0], rng[0] + 1) for j in range(-rng[1], rng[1] + 1) for k in range(-rng[2], rng[2] + 1)]


def nonbonded_reference(x, types, q, cell_mat, periodic, P, ctrl, lgflag):
    """x (n,3), types (n,) zero-based type indices, q (n,), P parsed params dump (runner.read_params_dump), ctrl: dict with
    nonb_cut / nonb_low. Returns dict(e_vdW, e_ele, e_pol, n_pairs, n_pairs_self)."""
    G = P["G"]; hdr = P["meta"].split()
    vdw_type = int(hdr[hdr.index("vdw_type") + 1])
    p_vdW1 = G[28]
    swa, swb = ctrl["nonb_low"], ctrl["nonb_cut"]
    c = taper_poly(swa, swb)
    n = len(x)
    ext = float(np.ptp(x, axis=0).max()) if n else 0.0
    shifts = np.array(image_shifts(cell_mat, periodic, swb, ext), float)
    sh_vec = shifts @ cell_mat
    zero_idx = [k for k, s in enumerate(shifts) if not s.any()][0]
    e_vdw = e_ele = 0.0
    npairs = nself = 0
    for i in range(n):
        for j in range(i, n):
            T = P["T"][(int(types[i]), int(types[j]))]
            T2 = P["T2"][(int(types[i]), int(types[j]))]
            D, alpha, r_vdW, gamma_w, rcore, ecore, acore, lgcij, lgre = T[18], T[19], T[20], T[21], T[22], T[23], T[24], T[25], T[26]
            gamma_c = T2[0]
            d = x[j] + sh_vec - x[i]
            r = np.sqrt((d * d).sum(axis=1))
            m = (r <= swb) & (r > 0.0)
            if i == j:
                # self-image pairs: n and -n are the same physical pair; keep the half-space n > 0 (lexicographic)
                keep = np.zeros(len(shifts), bool)
                for k, s in enumerate(shifts):
                    t = tuple(s)
                    if t > (0.0, 0.0, 0.0): keep[k] = True
                m &= keep
            rr = r[m]
            if not rr.size: continue
            tap = sum(c[k] * rr ** k for k in range(8))
            if vdw_type in (1, 3):
                fn13 = (rr ** p_vdW1 + (1.0 / gamma_w) ** p_vdW1) ** (1.0 / p_vdW1)
                e1 = np.exp(alpha * (1.0 - fn13 / r_vdW)); e2 = np.exp(0.5 * alpha * (1.0 - fn13 / r_vdW))
            else:
                e1 = np.exp(alpha * (1.0 - rr / r_vdW)); e2 = np.exp(0.5 * alpha * (1.0 - rr / r_vdW))
            ev = D * (e1 - 2.0 * e2)
            if vdw_type in (2, 3):
                ev = ev + ecore * np.exp(acore * (1.0 - rr / rcore))
                if lgflag:
                    ev = ev - lgcij / (rr ** 6 + lgre ** 6)
            e_vdw += float((tap * ev).sum())
            e_ele += float((C_ELE * q[i] * q[j] * tap / (rr ** 3 + gamma_c) ** CBRT_EXP).sum())
            npairs += int(rr.size)
            if i == j: nself += int(rr.size)
    e_pol = 0.0
    for i in range(n):
        S = P["S"][int(types[i])]
        chi, eta = float(S[15]), float(S[16])
        e_pol += KCAL_PER_EV_LAMMPS * (chi * q[i] + 0.5 * eta * q[i] ** 2)
    return {"e_vdW": e_vdw, "e_ele": e_ele, "e_pol": e_pol, "n_pairs": npairs, "n_pairs_self": nself}
