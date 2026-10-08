#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""M0 arithmetic checks behind docs/NUMERICAL_POLICY.md section 4 (reproducible: `python3 -I tools/fp32_hazards.py`).

Pure arithmetic on parameter values taken from third_party LAMMPS `ffield.reax.cho` (p_boc1=gp[0]=50.0,
p_boc2=gp[1]=9.5469, p_lp1=gp[15]=6.0891). It does not run any ReaxFF code; it only evaluates where
FP32 exp() overflows for expressions that appear verbatim in the reference (ENGINE_SPEC sections 4, 5.2, 5.9).
"""
import math

import numpy as np

f32max = float(np.finfo(np.float32).max)
ln32 = math.log(f32max)
print("FP32 exp overflow argument: %.3f   (FP64: %.1f)" % (ln32, math.log(1.7976931348623157e308)))

p_boc1, p_boc2 = 50.0, 9.5469
for val in (1.0, 2.0, 3.0, 4.0, 6.0):
    # weakly bonded atom: total BO' ~ 0 -> Delta' = -val ; f2 = exp(-p_boc1*Delta') = exp(p_boc1*val)
    a, b = p_boc1 * val, p_boc2 * val
    print("valence %.0f: exp(p_boc1*val)=exp(%.0f)=%.2e FP32-overflow=%s | exp(p_boc2*val)=exp(%.1f)=%.2e FP32-overflow=%s"
          % (val, a, math.exp(a), a > ln32, b, math.exp(b), b > ln32))
print("Overflow threshold on Delta' for p_boc1=50: Delta' < %.3f (FP32) vs %.2f (FP64)" % (-ln32 / p_boc1, -709.78 / p_boc1))
print("lone pair exp(-75*dlp): FP32 overflow when dlp < %.3f ; isolated O (dlp=-1.0): exp(75)=%.2e (%.1e of FP32 max)"
      % (-ln32 / 75, math.exp(75), math.exp(75) / f32max))


def tap_coeffs(swa, swb):  # verbatim from reaxff_init_md.cpp:90-105
    d = swb - swa
    d7 = d ** 7
    a2, a3, b2, b3 = swa ** 2, swa ** 3, swb ** 2, swb ** 3
    return [(-35 * a3 * b2 * b2 + 21 * a2 * b3 * b2 - 7 * swa * b3 * b3 + b3 * b3 * swb) / d7,
            140 * a3 * b3 / d7, -210 * (a3 * b2 + a2 * b3) / d7,
            140 * (a3 * swb + 3 * a2 * b2 + swa * b3) / d7, -35 * (a3 + 9 * a2 * swb + 9 * swa * b2 + b3) / d7,
            84 * (a2 + 3 * swa * swb + b2) / d7, -70 * (swa + swb) / d7, 20 / d7]


for swa, swb in ((0.0, 10.0), (0.5, 10.0), (0.0, 8.0)):
    c = tap_coeffs(swa, swb)
    worst = worst32 = 0.0
    for r in np.linspace(swa, swb, 2001):
        h = 0.0
        for k in range(7, -1, -1):
            h = h * r + c[k]
        x = (r - swa) / (swb - swa)
        s = 1 - 35 * x ** 4 + 84 * x ** 5 - 70 * x ** 6 + 20 * x ** 7
        worst = max(worst, abs(h - s))
        h32 = np.float32(0)
        for k in range(7, -1, -1):
            h32 = np.float32(np.float32(h32) * np.float32(r) + np.float32(c[k]))
        worst32 = max(worst32, abs(float(h32) - s))
    print("swa=%.1f swb=%.1f: max|Horner(FP64)-scaled|=%.2e   max|Horner(FP32)-scaled|=%.2e"
          % (swa, swb, worst, worst32))
