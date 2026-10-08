#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Independently generated test geometries (no LAMMPS involvement, no copying from LAMMPS example data).

Every geometry is built from textbook internal coordinates (bond lengths / angles) or crystal parameters stated in the
function that builds it, so provenance is the construction itself. Output is a list of {"el","xyz"} dicts; callers wrap
them into a case (see fixtures.py). Coordinates in Angstrom.
"""
import numpy as np

D2R = np.pi / 180.0


def _rot(axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = axis
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def atoms(*items):
    return [{"el": e, "xyz": [float(v) for v in p]} for e, p in items]


def translate(at, d):
    return [{"el": a["el"], "xyz": [float(a["xyz"][k] + d[k]) for k in range(3)]} for a in at]


def rotate(at, axis, ang_deg):
    R = _rot(axis, ang_deg * D2R)
    return [{"el": a["el"], "xyz": [float(v) for v in R @ np.array(a["xyz"])]} for a in at]


def merge(*groups):
    return [a for g in groups for a in g]


# ---------------------------------------------------------------------------------------------- C/H/O/N molecules
def methane(rch=1.09):
    t = rch / np.sqrt(3.0)   # tetrahedral: H at (+-1,+-1,+-1) pattern scaled
    return atoms(("C", (0, 0, 0)), ("H", (t, t, t)), ("H", (t, -t, -t)), ("H", (-t, t, -t)), ("H", (-t, -t, t)))


def ethane(rcc=1.54, rch=1.09, tors=60.0):
    """Staggered by default (H-C-C-H dihedral between the two methyls = `tors` degrees)."""
    th = (180.0 - 109.4712206) * D2R   # angle between C-C axis and the C-H bond projection
    out = [("C", (0, 0, 0)), ("C", (0, 0, rcc))]
    for k in range(3):
        a = k * 120.0 * D2R
        out.append(("H", (rch * np.sin(th) * np.cos(a), rch * np.sin(th) * np.sin(a), -rch * np.cos(th))))
    for k in range(3):
        a = k * 120.0 * D2R + tors * D2R
        out.append(("H", (rch * np.sin(th) * np.cos(a), rch * np.sin(th) * np.sin(a), rcc + rch * np.cos(th))))
    return atoms(*out)


def ethene(rcc=1.34, rch=1.085, hcc=121.3):
    """H-C=C angle `hcc`; planar, C=C along z."""
    a = hcc * D2R
    out = [("C", (0, 0, -rcc / 2)), ("C", (0, 0, rcc / 2))]
    for zc, toward in ((-rcc / 2, 1.0), (rcc / 2, -1.0)):   # `toward` = direction (along z) of the other carbon
        for sx in (-1.0, 1.0):
            out.append(("H", (sx * rch * np.sin(a), 0.0, zc + toward * rch * np.cos(a))))
    return atoms(*out)


def ethyne(rcc=1.20, rch=1.06):
    return atoms(("C", (0, 0, -rcc / 2)), ("C", (0, 0, rcc / 2)), ("H", (0, 0, -rcc / 2 - rch)), ("H", (0, 0, rcc / 2 + rch)))


def water(roh=0.9572, hoh=104.52):
    a = hoh / 2 * D2R
    return atoms(("O", (0, 0, 0)), ("H", (roh * np.sin(a), 0, roh * np.cos(a))), ("H", (-roh * np.sin(a), 0, roh * np.cos(a))))


def water_dimer(roo=2.91):
    """Linear-ish H-bond: donor water's H1 points at acceptor O (donor-H...O angle 180)."""
    w1 = water()
    h1 = np.array(w1[1]["xyz"])
    u = h1 / np.linalg.norm(h1)
    shift = u * roo
    w2 = rotate(water(), (0, 1, 0), 0.0)
    return merge(w1, translate(w2, shift))


def carbon_monoxide(r=1.128):
    return atoms(("C", (0, 0, 0)), ("O", (0, 0, r)))


def carbon_dioxide(r=1.16):
    return atoms(("C", (0, 0, 0)), ("O", (0, 0, r)), ("O", (0, 0, -r)))


def formaldehyde(rco=1.21, rch=1.10, hch=116.0):
    a = hch / 2 * D2R
    return atoms(("C", (0, 0, 0)), ("O", (0, 0, rco)), ("H", (rch * np.sin(a), 0, -rch * np.cos(a))), ("H", (-rch * np.sin(a), 0, -rch * np.cos(a))))


def ammonia(rnh=1.012, hnh=106.7):
    # pyramidal: H's on a cone around -z; cone half-angle from H-N-H angle
    cosang = np.cos(hnh * D2R)
    # chord geometry: for 3 equal vectors with mutual angle theta, cos(alpha)= sqrt((1+2cos theta)/3) (angle to the axis)
    ca = np.sqrt((1 + 2 * cosang) / 3.0)
    sa = np.sqrt(1 - ca * ca)
    out = [("N", (0, 0, 0))]
    for k in range(3):
        b = k * 120.0 * D2R
        out.append(("H", (rnh * sa * np.cos(b), rnh * sa * np.sin(b), -rnh * ca)))
    return atoms(*out)


def hydrogen_molecule(r=0.7414):
    return atoms(("H", (0, 0, 0)), ("H", (0, 0, r)))


def oxygen_molecule(r=1.208):
    return atoms(("O", (0, 0, 0)), ("O", (0, 0, r)))


def single_atom(el):
    return atoms((el, (0, 0, 0)))


def methyl_radical(rch=1.079):
    return atoms(("C", (0, 0, 0)), *[("H", (rch * np.cos(k * 120 * D2R), rch * np.sin(k * 120 * D2R), 0)) for k in range(3)])


def methanol(rco=1.43, roh=0.96, rch=1.09):
    """CH3-OH, staggered; angles from textbook tetrahedral / 108.5 deg C-O-H."""
    th = (180.0 - 109.4712206) * D2R
    out = [("C", (0, 0, 0)), ("O", (0, 0, rco))]
    for k in range(3):
        a = k * 120.0 * D2R
        out.append(("H", (rch * np.sin(th) * np.cos(a), rch * np.sin(th) * np.sin(a), -rch * np.cos(th))))
    coh = 108.5 * D2R
    out.append(("H", (roh * np.sin(np.pi - coh) * np.cos(60 * D2R), roh * np.sin(np.pi - coh) * np.sin(60 * D2R),
                      rco + roh * np.cos(np.pi - coh))))
    return atoms(*out)


def benzene(rcc=1.397, rch=1.084):
    out = []
    for k in range(6):
        a = k * 60.0 * D2R
        out.append(("C", (rcc * np.cos(a), rcc * np.sin(a), 0.0)))
    for k in range(6):
        a = k * 60.0 * D2R
        out.append(("H", ((rcc + rch) * np.cos(a), (rcc + rch) * np.sin(a), 0.0)))
    return atoms(*out)


def butadiene(r_single=1.467, r_double=1.338, rch=1.09, ccc=122.4):
    """trans-1,3-butadiene, planar zigzag in the xy plane; used for conjugation / 4-body terms."""
    t = (180.0 - ccc) * D2R
    headings = [0.0, t, 0.0]
    lengths = [r_double, r_single, r_double]
    pts = [np.zeros(3)]
    for h, r in zip(headings, lengths):
        pts.append(pts[-1] + r * np.array([np.cos(h), np.sin(h), 0.0]))
    out = [("C", p) for p in pts]

    def bisector_h(center, n1, n2):
        v1 = n1 - center; v2 = n2 - center
        bis = -(v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)); bis /= np.linalg.norm(bis)
        return center + rch * bis
    out.append(("H", bisector_h(pts[1], pts[0], pts[2])))
    out.append(("H", bisector_h(pts[2], pts[1], pts[3])))
    for end, nb in ((0, 1), (3, 2)):
        v = pts[end] - pts[nb]; v /= np.linalg.norm(v)
        p = np.cross(np.array([0.0, 0.0, 1.0]), v)
        for sgn in (1.0, -1.0):
            out.append(("H", pts[end] + rch * (np.cos(60 * D2R) * v + sgn * np.sin(60 * D2R) * p)))
    return atoms(*out)


def cyclopropane(rcc=1.51, rch=1.083):
    """Strained 3-ring (angle 60 deg) -> exercises valence-angle penalty / small-angle terms."""
    R = rcc / np.sqrt(3.0)
    out = []
    for k in range(3):
        a = k * 120.0 * D2R + 90 * D2R
        out.append(("C", (R * np.cos(a), R * np.sin(a), 0.0)))
    for k in range(3):
        a = k * 120.0 * D2R + 90 * D2R
        for s in (1.0, -1.0):
            out.append(("H", ((R + rch * np.cos(58 * D2R)) * np.cos(a), (R + rch * np.cos(58 * D2R)) * np.sin(a), s * rch * np.sin(58 * D2R))))
    return atoms(*out)


# ---------------------------------------------------------------------------------------------- periodic crystals
def diamond(a0=3.567, n=2, el="C"):
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                     [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]])
    pts = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]
    at = [{"el": el, "xyz": [float(v) for v in (b + np.array(p)) * a0]} for p in pts for b in base]
    return at, {"lo": [0, 0, 0], "hi": [n * a0] * 3}


def graphene_sheet(nx=3, ny=3, a=1.42, vac=12.0):
    """Orthogonal 4-atom graphene cell repeated nx x ny; periodic in-plane, vacuum `vac` along z."""
    ax = np.sqrt(3.0) * a
    pts = []
    for i in range(nx):
        for j in range(ny):
            ox, oy = i * ax, j * 3.0 * a
            pts += [(ox, oy), (ox, oy + a), (ox + ax / 2, oy + 1.5 * a), (ox + ax / 2, oy + 2.5 * a)]
    at = [{"el": "C", "xyz": [float(x), float(y), vac / 2]} for x, y in pts]
    return at, {"lo": [0, 0, 0], "hi": [nx * ax, ny * 3.0 * a, vac]}


def fcc(el, a0, n=2):
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pts = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]
    at = [{"el": el, "xyz": [float(v) for v in (b + np.array(p)) * a0]} for p in pts for b in base]
    return at, {"lo": [0, 0, 0], "hi": [n * a0] * 3}


def bcc(el, a0, n=2):
    base = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    pts = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]
    at = [{"el": el, "xyz": [float(v) for v in (b + np.array(p)) * a0]} for p in pts for b in base]
    return at, {"lo": [0, 0, 0], "hi": [n * a0] * 3}


def rocksalt_like(el1, el2, a0, n=1):
    """NaCl-type 8-atom conventional cell."""
    cat = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    ani = cat + np.array([0.5, 0, 0])
    pts = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]
    at = [{"el": el1, "xyz": [float(v) for v in (b + np.array(p)) * a0]} for p in pts for b in cat]
    at += [{"el": el2, "xyz": [float(v) for v in (b + np.array(p)) * a0]} for p in pts for b in ani]
    return at, {"lo": [0, 0, 0], "hi": [n * a0] * 3}


def box_around(at, pad=15.0):
    X = np.array([a["xyz"] for a in at])
    lo = X.min(axis=0) - pad; hi = X.max(axis=0) + pad
    return {"lo": [float(v) for v in lo], "hi": [float(v) for v in hi]}
