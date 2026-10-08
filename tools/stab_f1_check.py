#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""M0.5 value-equivalence check of the proposed stable form of the bond-order overcoordination factor f1
(docs/NUMERICAL_POLICY.md 4.4). VALUES ONLY -- derivative coefficients are NOT checked here (M4 deliverable).
reference (reaxff_bond_orders.cpp:321-329):  f2=a_i+a_j, f3=-(1/p2)ln(.5(b_i+b_j)), f1=.5[(v_i+f2)/(v_i+f2+f3)+(v_j+f2)/(v_j+f2+f3)]
stable:  m=min(D_i,D_j);  m>=0 -> reference;  m<0 -> factor e^{-p1 m} out of f2 and divide through.
"""
import numpy as np

p1, p2 = 50.0, 9.5469   # ffield.reax.cho gp[0], gp[1]

def f1_ref(Di, Dj, vi, vj, dtype):
    d = dtype
    with np.errstate(all="ignore"):
        ai = np.exp(d(-p1) * d(Di)); aj = np.exp(d(-p1) * d(Dj))
        bi = np.exp(d(-p2) * d(Di)); bj = np.exp(d(-p2) * d(Dj))
        f2 = ai + aj; f3 = d(-1.0 / p2) * np.log(d(0.5) * (bi + bj))
        return d(0.5) * ((d(vi) + f2) / (d(vi) + f2 + f3) + (d(vj) + f2) / (d(vj) + f2 + f3))

def f1_stable(Di, Dj, vi, vj, dtype):
    d = dtype
    m = min(Di, Dj)
    # f3 by log-sum-exp (shift by the larger exponent argument of b)
    xi, xj = d(-p2) * d(Di), d(-p2) * d(Dj); mx = max(xi, xj)
    f3 = d(-1.0 / p2) * (mx + np.log(d(0.5) * (np.exp(xi - mx) + np.exp(xj - mx))))
    if m >= 0:
        return f1_ref(Di, Dj, vi, vj, dtype)
    em = np.exp(d(p1) * d(m))                      # <= 1, may underflow to 0 harmlessly
    ai = np.exp(d(-p1) * (d(Di) - d(m))); aj = np.exp(d(-p1) * (d(Dj) - d(m)))   # alpha_x <= 1
    s = ai + aj
    r = lambda v: (d(v) * em + s) / ((d(v) + f3) * em + s)
    return d(0.5) * (r(vi) + r(vj))

rng = np.random.default_rng(7)
worst64 = worst32 = 0.0; nan_ref32 = 0; nan_st32 = 0; n = 0
for vi, vj in ((4.0, 1.0), (1.0, 1.0), (4.0, 4.0), (2.0, 6.0), (3.0, 3.0)):
    for _ in range(4000):
        Di = rng.uniform(-vi, 4.0); Dj = rng.uniform(-vj, 4.0)     # Delta' >= -valence (no bonds) up to over-coordination
        n += 1
        ref64 = f1_ref(Di, Dj, vi, vj, np.float64)
        if not np.isfinite(ref64): continue
        st64 = f1_stable(Di, Dj, vi, vj, np.float64)
        worst64 = max(worst64, abs(st64 - ref64) / max(1e-300, abs(ref64)))
        r32 = f1_ref(Di, Dj, vi, vj, np.float32); s32 = f1_stable(Di, Dj, vi, vj, np.float32)
        nan_ref32 += not np.isfinite(r32); nan_st32 += not np.isfinite(s32)
        if np.isfinite(s32): worst32 = max(worst32, abs(float(s32) - ref64) / abs(ref64))
print("samples=%d  max rel |stable(FP64)-reference(FP64)| = %.2e" % (n, worst64))
print("FP32 reference form non-finite: %d / %d ; FP32 stable form non-finite: %d / %d" % (nan_ref32, n, nan_st32, n))
print("max rel |stable(FP32)-reference(FP64)| over finite samples = %.2e" % worst32)
