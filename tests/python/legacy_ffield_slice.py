#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Independent re-implementation of how the LEGACY eReaxFF code reads a force field: fixed-column Fortran formats, no token splitting.
(reac.f subroutine ffinpt, read-only source of the Margaret group; formats 1100 / 1200 / 1250 / 1300 / 1400 / 1450.)

  general parameters : (f10.4)            columns 1-10 of each line
  atom line 1        : (1x,a2,10f9.4)     name = columns 2-3, then 9-character fields
  atom lines 2-4     : (3x,10f9.4)        three skipped columns, then 9-character fields
  bond line 1        : (2i3,8f9.4)        two 3-character integers, then 9-character fields
  bond line 2        : (6x,8f9.4)         six skipped columns, then 9-character fields

Fortran numeric input: a blank field is 0; a field without a decimal point has an implied decimal point before the last 4 digits (f9.4).
A value wider than its field is silently clipped by this reader -- exactly the case a token parser would not notice.
Post-read rule of the legacy code (reac.f:4622-4629, "Equate vval3 to valf for first-row elements"): for every atom type with mass < 21,
vval3 (atom line 4, column 4) is replaced by valf (atom line 2, column 3), with a warning. The effective values are printed.

Prints  GEN <n> v0..vn-1 / ATOM <name> <32 values> / BOND <i> <j> <16 values>  (i, j: 1-based atom-type indices in file order).
usage: legacy_ffield_slice.py <ffield>"""
import sys


def fnum(field, d=4):
    f = field.strip()
    if not f:
        return 0.0
    if "." in f or "e" in f.lower() or "d" in f.lower():
        return float(f.lower().replace("d", "e"))
    return int(f) * 10.0 ** (-d)


def fields(line, start, n, width=9):
    line = line.ljust(start + n * width)
    return [fnum(line[start + k * width: start + (k + 1) * width]) for k in range(n)]


def main(path):
    L = open(path).read().split("\n")
    pos = 1                                  # line 0 is the identifier
    npar = int(L[pos][:3]); pos += 1
    print("GEN", npar, *("%.4f" % fnum(L[pos + i][:10]) for i in range(npar)))
    pos += npar
    nso = int(L[pos][:3]); pos += 1
    pos += 3                                 # three header lines
    names = []
    for _ in range(nso):
        l1 = L[pos]; names.append(l1[1:3].strip()); v = fields(l1, 3, 10)[:8]
        v += fields(L[pos + 1], 3, 10)[:8] + fields(L[pos + 2], 3, 10)[:8] + fields(L[pos + 3], 3, 10)[:8]
        if v[2] < 21.0 and v[27] != v[10]:   # amas < 21: vval3 := valf
            v[27] = v[10]
        print("ATOM", names[-1], *("%.4f" % x for x in v))
        pos += 4
    nbo = int(L[pos][:3]); pos += 1
    pos += 1                                 # header line
    for _ in range(nbo):
        a, b = int(L[pos][0:3]), int(L[pos][3:6])
        v = fields(L[pos], 6, 8) + fields(L[pos + 1], 6, 8)
        print("BOND", a, b, *("%.4f" % x for x in v))
        pos += 2


if __name__ == "__main__":
    main(sys.argv[1])
