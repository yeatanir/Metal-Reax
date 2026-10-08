#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Minimal force-field text editing for the Q-09 / Q-12 parser-quirk experiments (never used to build fixtures)."""
import re


def _find(lines, pat):
    for k, l in enumerate(lines):
        if re.search(pat, l): return k
    raise KeyError(pat)


def angle_block(text):
    """Returns (lines, start_idx_of_count_line, [angle lines])"""
    L = text.splitlines()
    k = _find(L, r"Nr of angles")
    n = int(L[k].split()[0])
    return L, k, L[k + 1:k + 1 + n]


def set_angle_lines(text, new_lines):
    L, k, old = angle_block(text)
    head = L[k].split("!", 1)
    L[k] = f"{len(new_lines):3d}    !" + head[1]
    return "\n".join(L[:k + 1] + list(new_lines) + L[k + 1 + len(old):]) + "\n"


def fmt_angle(j, k, l, theta, p1, p2, pcoa, p7, ppen, p4):
    return f"{j:3d}{k:3d}{l:3d}{theta:9.4f}{p1:9.4f}{p2:9.4f}{pcoa:9.4f}{p7:9.4f}{ppen:9.4f}{p4:9.4f}"


def drop_bond_pair(text, a, b):
    """Remove the two-line bond-parameter block for the pair (a,b) (1-based, order-insensitive) and decrement the count."""
    L = text.splitlines()
    k = _find(L, r"Nr of bonds")
    n = int(L[k].split()[0])
    i = k + 2     # two header lines? the count line is followed by one continuation line in the header comment
    # the header has a second comment line ("p(be2);p(bo3)...") directly after the count line
    start = k + 2
    out = L[:k + 2]
    removed = False
    j = start
    for _ in range(n):
        w = L[j].split()
        pair = (int(w[0]), int(w[1]))
        if set(pair) == {a, b} and len(set(pair)) == len({a, b}) and not removed:
            removed = True
        else:
            out += [L[j], L[j + 1]]
        j += 2
    assert removed, "pair not found"
    out += L[j:]
    out[k] = f"{n - 1:3d}" + out[k][3:]
    return "\n".join(out) + "\n"
