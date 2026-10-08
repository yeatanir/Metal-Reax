#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Convert the parameter-table dump of the instrumented pinned LAMMPS (params.txt, written when REAXMETAL_DIAG_DIR is set)
into the canonical text of reaxmetal::canonical_table_dump (src/io/table_dump.cpp). Equal text <=> equal tables (PARSE-1).
The reference stores three-body sets with doubled slots for j==l (inert zero slot) and counts mirrored lines in both orientations;
the canonical form keeps the EFFECTIVE sets in summation order (ENGINE_SPEC Q-09). A header whose slot count exceeds the array
(cnt > 5) cannot be represented and is reported as OVERRUN.
usage: canonical_tables.py params.txt [--portable]"""
import sys


def convert(path, portable=False):
    G = None; hdr = None; S = {}; S2 = {}; T = {}; T2 = {}; H3 = {}; H4 = {}; HB = {}
    for line in open(path):
        if line.startswith("#ntypes"):
            hdr = line.split(); continue
        w = line.split()
        if not w: continue
        k = w[0]
        if k == "G": G = w[1:]
        elif k == "S": S[int(w[1])] = w[2:]
        elif k == "S2": S2[int(w[1])] = w[2:]
        elif k == "T": T[(int(w[1]), int(w[2]))] = w[3:]
        elif k == "T2": T2[(int(w[1]), int(w[2]))] = w[3:]
        elif k == "H3": H3[tuple(int(x) for x in w[1:4])] = w[4:]
        elif k == "H4": H4[tuple(int(x) for x in w[1:5])] = w[5:]
        elif k == "HB": HB[tuple(int(x) for x in w[1:4])] = w[4:]
    n = int(hdr[hdr.index("#ntypes") + 1]); vdw = int(hdr[hdr.index("vdw_type") + 1])
    out = ["FFIELD-TABLES 1", f"GP {len(G)} {vdw}"]
    out += [f"G {i} {v}" for i, v in enumerate(G)]
    for i in range(n):
        s = S[i]; s2 = S2[i]
        # dump S: name r_s valency mass r_vdw epsilon gamma r_pi valency_e nlp_opt alpha gamma_w valency_boc p_ovun5 p_hbond chi eta r_pi_pi p_lp2
        #         b_o_131 b_o_132 b_o_133 bcut_acks2 p_ovun2 p_val3 valency_val p_val5 rcore2 ecore2 ; S2: acore2 lgcij lgre
        out.append(f"S {i} " + " ".join(s) + " " + " ".join(s2))
    for i in range(n):
        for j in range(i, n):
            t = T[(i, j)]; t2 = T2[(i, j)]    # t2 = gamma v13cor ovc
            vals = t + ([] if portable else [t2[0]]) + [t2[1], t2[2]]
            out.append(f"T {i} {j} " + " ".join(vals))
    overrun = []
    for key in sorted(H3):
        j, k, l = key; h = H3[key]
        cnt = int(h[0].split("=")[1]); slots = h[1:1 + 35]
        if cnt > 5: overrun.append(key); continue
        sel = list(range(0, cnt, 2)) if j == l else list(range(cnt))
        if j == l and cnt % 2: overrun.append(key); continue
        if not sel: continue
        out.append(f"H3 {j} {k} {l} {len(sel)} " + " ".join(" ".join(slots[7 * c:7 * c + 7]) for c in sel))
    for key in sorted(H4):
        h = H4[key]; cnt = int(h[0].split("=")[1])
        if cnt >= 1: out.append("H4 " + " ".join(map(str, key)) + " " + " ".join(h[1:6]))
    for key in sorted(HB):
        h = HB[key]
        if [float(x) for x in h] == [-1.0, 0.0, 0.0, 0.0]: continue
        out.append("HB " + " ".join(map(str, key)) + " " + " ".join(h))
    if overrun:
        out.append("OVERRUN " + " ".join("-".join(map(str, k)) for k in overrun))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.write(convert(sys.argv[1], "--portable" in sys.argv))
