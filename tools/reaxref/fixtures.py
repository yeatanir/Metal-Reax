#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Generate the M1 reference fixtures (tests/fixtures/cases/*.json) and the force-field manifest.

Geometries come from geometries.py (independent construction). Force fields are the files bundled in the pinned LAMMPS
tree (potentials/ffield.reax.*); only their names + SHA-256 are committed (ADR-017), never the files.
Perturbations use numpy.random.default_rng(seed) with the seed stored in the case; the case JSON stores the final
coordinates, so reproduction never depends on the RNG.

usage: fixtures.py --ffield-dir <dir with ffield.reax.*> --out tests/fixtures
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import geometries as g

FF_USED = ["cho", "rdx", "budzien", "mattsson", "lg", "AB", "AuO", "Fe_O_C_H", "V_O_C_H", "ZnOH", "FC"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def perturb(at, amp, seed):
    rng = np.random.default_rng(seed)
    d = rng.uniform(-amp, amp, size=(len(at), 3))
    return [{"el": a["el"], "xyz": [float(a["xyz"][k] + d[i][k]) for k in range(3)]} for i, a in enumerate(at)], {"seed": seed, "amplitude": amp}


class Builder:
    def __init__(self, ffdir):
        self.ffdir = Path(ffdir); self.cases = []

    def add(self, cid, ff, els, at, cell=None, periodic=(False, False, False), desc="", tags=(), tol=1e-12, pad=15.0,
            pair=None, charge=None, extra=None, perturbation=None, eem_thr=None):
        name = "ffield.reax." + ff
        c = {"id": cid, "description": desc, "ffield": {"name": name, "sha256": sha(self.ffdir / name)},
             "elements": list(els), "cell": cell or g.box_around(at, pad), "periodic": list(periodic), "atoms": at,
             "charge": charge or {"model": "qeq/reaxff", "swb": 10.0, "tolerance": tol, "maxiter": 500},
             "tags": sorted(set(tags))}
        if pair: c["pair"] = pair
        if perturbation: c["perturbation"] = perturbation
        if eem_thr: c["eem_residual_threshold_eV"] = eem_thr
        if extra: c.update(extra)
        self.cases.append(c)

    def pert(self, cid, ff, els, at, amp, seed, **kw):
        at2, info = perturb(at, amp, seed)
        self.add(cid, ff, els, at2, perturbation=info, **kw)


def build(ffdir):
    b = Builder(ffdir)
    CHO = ["C", "H", "O"]
    # ---------------------------------------------------------------- isolated molecules, CHO force field
    b.add("cho_methane", "cho", CHO, g.methane(), desc="CH4, Td", tags=["bond", "angle", "molecule"])
    b.add("cho_ethane_staggered", "cho", CHO, g.ethane(tors=60.0), desc="C2H6 staggered", tags=["torsion", "angle", "molecule"])
    b.add("cho_ethane_eclipsed", "cho", CHO, g.ethane(tors=0.0), desc="C2H6 eclipsed (torsion barrier)", tags=["torsion", "molecule"])
    b.add("cho_ethene", "cho", CHO, g.ethene(), desc="C2H4 planar, C=C", tags=["pi-bond", "torsion", "molecule"])
    b.add("cho_ethyne", "cho", CHO, g.ethyne(), desc="C2H2, C#C (linear angles, sin(theta) clamp)", tags=["triple-bond", "linear", "molecule"])
    b.add("cho_butadiene", "cho", CHO, g.butadiene(), desc="trans-1,3-butadiene, conjugation", tags=["conjugation", "torsion", "coalition", "molecule"])
    b.add("cho_benzene", "cho", CHO, g.benzene(), desc="C6H6 planar ring, conjugation", tags=["conjugation", "ring", "molecule"])
    b.add("cho_cyclopropane", "cho", CHO, g.cyclopropane(), desc="C3H6, strained 60 deg angles", tags=["ring", "angle-penalty", "molecule"])
    b.add("cho_water", "cho", CHO, g.water(), desc="H2O", tags=["molecule", "polar"])
    b.add("cho_water_dimer", "cho", CHO, g.water_dimer(), desc="(H2O)2 hydrogen bonded", tags=["hbond", "molecule", "polar"])
    b.add("cho_methanol", "cho", CHO, g.methanol(), desc="CH3OH", tags=["molecule", "polar", "torsion"])
    b.add("cho_formaldehyde", "cho", CHO, g.formaldehyde(), desc="H2C=O (C-O pair, BO>=1: mass-based stabilisation branch)", tags=["pi-bond", "co-branch", "molecule"])
    b.add("cho_co", "cho", CHO, g.carbon_monoxide(), desc="C#O (terminal triple-bond stabilisation, mass-based branch)", tags=["triple-bond", "co-branch", "molecule", "diatomic"])
    b.add("cho_co2", "cho", CHO, g.carbon_dioxide(), desc="O=C=O linear", tags=["co-branch", "linear", "molecule"])
    b.add("cho_h2", "cho", CHO, g.hydrogen_molecule(), desc="H2", tags=["molecule", "diatomic"])
    b.add("cho_o2", "cho", CHO, g.oxygen_molecule(), desc="O2", tags=["molecule", "diatomic", "lonepair"])
    b.add("cho_ch3_radical", "cho", CHO, g.methyl_radical(), desc="CH3 planar radical (under-coordination)", tags=["undercoord", "molecule"])
    for el in ("C", "H", "O"):
        b.add(f"cho_atom_{el}", "cho", CHO, g.single_atom(el), desc=f"isolated {el} atom (enobonds default)", tags=["isolated-atom", "lonepair" if el == "O" else "atom"])
    # perturbed versions: non-zero forces everywhere, no symmetric cancellation
    b.pert("cho_methane_pert", "cho", CHO, g.methane(), 0.08, 101, desc="CH4 perturbed", tags=["bond", "angle", "molecule", "perturbed"])
    b.pert("cho_ethane_pert", "cho", CHO, g.ethane(), 0.08, 102, desc="C2H6 perturbed", tags=["torsion", "angle", "molecule", "perturbed"])
    b.pert("cho_butadiene_pert", "cho", CHO, g.butadiene(), 0.06, 103, desc="butadiene perturbed", tags=["conjugation", "torsion", "coalition", "molecule", "perturbed"])
    b.pert("cho_benzene_pert", "cho", CHO, g.benzene(), 0.05, 104, desc="benzene perturbed (out-of-plane too)", tags=["conjugation", "ring", "molecule", "perturbed"])
    b.pert("cho_water_dimer_pert", "cho", CHO, g.water_dimer(), 0.08, 105, desc="(H2O)2 perturbed", tags=["hbond", "molecule", "polar", "perturbed"])
    b.pert("cho_methanol_pert", "cho", CHO, g.methanol(), 0.08, 106, desc="CH3OH perturbed", tags=["molecule", "polar", "torsion", "perturbed"])
    b.pert("cho_co2_pert", "cho", CHO, g.carbon_dioxide(), 0.08, 107, desc="CO2 bent/perturbed", tags=["co-branch", "molecule", "perturbed", "angle"])
    b.pert("cho_cyclopropane_pert", "cho", CHO, g.cyclopropane(), 0.05, 108, desc="cyclopropane perturbed", tags=["ring", "angle-penalty", "molecule", "perturbed"])
    # fixed charge (no QEq): isolates the non-charge-model part of the oracle; charges chosen by hand, neutral
    wq = g.water()
    wq[0]["q"] = -0.8; wq[1]["q"] = 0.4; wq[2]["q"] = 0.4
    b.add("cho_water_fixedq", "cho", CHO, wq, desc="H2O with fixed hand-set charges, no charge fix (checkqeq no)",
          tags=["molecule", "fixed-charge"], charge={"model": "fixed"}, pair={"checkqeq": "no"})

    # ---------------------------------------------------------------- other force fields / branches
    b.add("rdx_ethyne", "rdx", ["C", "H", "O", "N"], g.ethyne(), desc="gp[37]==2 force field: stabilisation applies to every bond with BO>=1", tags=["gp37", "triple-bond", "molecule"])
    b.add("rdx_co", "rdx", ["C", "H", "O", "N"], g.carbon_monoxide(), desc="CO under gp[37]==2", tags=["gp37", "triple-bond", "co-branch", "molecule", "diatomic"])
    b.add("rdx_methane", "rdx", ["C", "H", "O", "N"], g.methane(), desc="CH4 under gp[37]==2 (BO<1: stabilisation inactive)", tags=["gp37", "molecule"])
    b.add("rdx_ammonia", "rdx", ["C", "H", "O", "N"], g.ammonia(), desc="NH3, N-containing", tags=["N", "molecule", "polar"])
    b.add("budzien_ammonia", "budzien", ["C", "H", "O", "N"], g.ammonia(), desc="NH3 with the budzien parameter set", tags=["N", "molecule", "polar"])
    b.add("mattsson_h2s", "mattsson", ["C", "H", "O", "N", "S"], [{"el": "S", "xyz": [0, 0, 0]}, {"el": "H", "xyz": [1.336 * np.sin(46.1 * np.pi / 180), 0, 1.336 * np.cos(46.1 * np.pi / 180)]}, {"el": "H", "xyz": [-1.336 * np.sin(46.1 * np.pi / 180), 0, 1.336 * np.cos(46.1 * np.pi / 180)]}],
          desc="H2S (S: mass>21 -> valency override branch)", tags=["S", "heavy-branch", "molecule", "polar"])
    so2 = [{"el": "S", "xyz": [0, 0, 0]}, {"el": "O", "xyz": [1.431 * np.sin(59.25 * np.pi / 180), 0, 1.431 * np.cos(59.25 * np.pi / 180)]}, {"el": "O", "xyz": [-1.431 * np.sin(59.25 * np.pi / 180), 0, 1.431 * np.cos(59.25 * np.pi / 180)]}]
    so2_fc = [{"el": "S", "xyz": [0, 0, 0]}, {"el": "O", "xyz": [1.431 * np.sin(59.25 * np.pi / 180), 0, 1.431 * np.cos(59.25 * np.pi / 180)]}, {"el": "O", "xyz": [-1.431 * np.sin(59.25 * np.pi / 180), 0, 1.431 * np.cos(59.25 * np.pi / 180)]}]
    b.add("mattsson_so2", "mattsson", ["C", "H", "O", "N", "S"], so2, desc="SO2 with the mattsson set: its S-O bond block has De=0 (bond order active, bond energy zero); exercises the mass>21 over/under branch (Q-34)", tags=["S", "heavy-branch", "molecule", "polar", "pi-bond", "known-reference-quirk"])
    b.add("mattsson_ethane", "mattsson", ["C", "H", "O", "N", "S"], g.ethane(), desc="C2H6 with the mattsson set", tags=["torsion", "molecule"])
    # FC set has real S-O bond energies: S=O pi bonds on a heavy (mass>21) atom with light neighbours -> Q-34 derivative defect
    b.add("fc_so2", "FC", ["S", "O"], so2_fc, desc="SO2 with the FC set (non-zero S-O De): heavy S with pi bonds to light O; control for Q-34 (here the analytic force IS consistent with the energy, FD 1e-7)",
          tags=["S", "heavy-branch", "molecule", "polar", "pi-bond"], extra={"expected_warnings": [r"Changed valency_val to valency_boc for X"]})
    b.pert("fc_so2_pert", "FC", ["S", "O"], so2_fc, 0.06, 109, desc="SO2 (FC) perturbed", tags=["S", "heavy-branch", "molecule", "polar", "pi-bond", "perturbed"], extra={"expected_warnings": [r"Changed valency_val to valency_boc for X"]})
    # lg dispersion: ffield.reax.lg + 'lgvdw yes'
    b.add("lg_ethane_lgvdw", "lg", ["C", "H", "O", "N", "S"], g.ethane(), desc="C2H6, lg dispersion (lgvdw yes), vdw_type 3", tags=["lgvdw", "molecule", "torsion"], pair={"lgvdw": True})
    b.add("lg_water_dimer_lgvdw", "lg", ["C", "H", "O", "N", "S"], g.water_dimer(), desc="(H2O)2, lg dispersion", tags=["lgvdw", "hbond", "molecule"], pair={"lgvdw": True})
    # inner-wall vdW (vdw_type 3) without lg: ffield AB
    nh3bh3 = [{"el": "N", "xyz": [0, 0, 0]}, {"el": "B", "xyz": [0, 0, 1.658]}]
    thN = 111.0 * np.pi / 180      # H-N-B angle
    thB = 104.5 * np.pi / 180      # H-B-N angle
    for k in range(3):
        a_ = k * 120.0 * np.pi / 180
        nh3bh3.append({"el": "H", "xyz": [1.014 * np.sin(thN) * np.cos(a_), 1.014 * np.sin(thN) * np.sin(a_), 1.014 * np.cos(thN)]})
        a2 = a_ + 60.0 * np.pi / 180   # staggered
        nh3bh3.append({"el": "H", "xyz": [1.211 * np.sin(thB) * np.cos(a2), 1.211 * np.sin(thB) * np.sin(a2), 1.658 - 1.211 * np.cos(thB)]})
    b.add("ab_ammonia_borane", "AB", ["H", "O", "N", "B", "X"], nh3bh3, desc="H3N-BH3 (inner wall vdw_type 3, B,N)", tags=["innerwall", "N", "B", "molecule", "polar"], extra={"expected_warnings": [r"Changed valency_val to valency_boc for X"]})
    # metals (periodic crystals and a hydroxide molecule)
    au, auc = g.fcc("Au", 4.078, 2)
    b.pert("auo_au_fcc", "AuO", ["H", "O", "Au"], au, 0.05, 201, desc="Au fcc 2x2x2 (32 atoms), periodic", cell=auc, periodic=(True, True, True), tags=["metal", "periodic", "perturbed", "heavy-branch"], tol=1e-11)
    fe, fec = g.bcc("Fe", 2.87, 3)
    b.pert("feoch_fe_bcc", "Fe_O_C_H", ["C", "H", "O", "Fe"], fe, 0.04, 202, desc="Fe bcc 3x3x3 (54 atoms), periodic", cell=fec, periodic=(True, True, True), tags=["metal", "periodic", "perturbed", "heavy-branch"], tol=1e-11)
    rs, rsc = g.rocksalt_like("V", "O", 4.1, 2)
    b.pert("voch_vo_rocksalt", "V_O_C_H", ["C", "H", "O", "V"], rs, 0.04, 203, desc="VO rock-salt-like 2x2x2 (64 atoms), periodic", cell=rsc, periodic=(True, True, True), tags=["metal", "periodic", "perturbed", "heavy-branch", "ionic"], tol=1e-11, extra={"expected_warnings": [r"Changed valency_val to valency_boc for X"]})
    zn = [{"el": "H", "xyz": [0, 0, 0]}, {"el": "O", "xyz": [0, 0, 0.96]}, {"el": "Zn", "xyz": [0, 0, 0.96 + 1.80]}, {"el": "O", "xyz": [0, 0, 0.96 + 3.60]}, {"el": "H", "xyz": [0, 0, 0.96 + 3.60 + 0.96]}]
    b.pert("znoh_znoh2", "ZnOH", ["H", "O", "Zn"], zn, 0.10, 204, desc="Zn(OH)2 near-linear", tags=["metal", "molecule", "heavy-branch", "perturbed"], extra={"expected_warnings": [r"Changed valency_val to valency_boc for X"]})
    # ---------------------------------------------------------------- periodic CHO
    dia, diac = g.diamond(3.567, 2)
    b.pert("cho_diamond_2x2x2", "cho", CHO, dia, 0.10, 301, desc="diamond 2x2x2 (64 atoms, cell 7.134 < nonbonded cutoff 10): the LAMMPS regression geometry class", cell=diac, periodic=(True, True, True), tags=["periodic", "crystal", "small-cell", "perturbed", "images"], tol=1e-11)
    dia1, dia1c = g.diamond(3.567, 1)
    b.pert("cho_diamond_1x1x1", "cho", CHO, dia1, 0.05, 302, desc="diamond conventional cell (8 atoms, cell 3.567 Å): bonded 3-body/4-body terms across boundaries, self-images", cell=dia1c, periodic=(True, True, True), tags=["periodic", "crystal", "small-cell", "self-image", "perturbed", "images"], tol=1e-11)
    dia3, dia3c = g.diamond(3.567, 3)
    b.pert("cho_diamond_3x3x3", "cho", CHO, dia3, 0.05, 303, desc="diamond 3x3x3 (216 atoms, cell 10.70 > cutoff 10)", cell=dia3c, periodic=(True, True, True), tags=["periodic", "crystal", "perturbed", "large-cell"], tol=1e-10)
    gr, grc = g.graphene_sheet(3, 2)
    b.pert("cho_graphene", "cho", CHO, gr, 0.03, 304, desc="graphene 3x2 orthogonal cell (24 atoms), periodic xy, vacuum z", cell=grc, periodic=(True, True, False), tags=["periodic", "2d", "conjugation", "perturbed", "images"], tol=1e-11)
    # sheared (triclinic) diamond: fractional coordinates of the cubic diamond cell mapped into skewed vectors
    base, _ = g.diamond(3.567, 2)
    frac = np.array([a["xyz"] for a in base]) / 7.134
    vec = np.array([[7.134, 0.0, 0.0], [1.2, 7.0, 0.0], [0.8, 1.0, 7.1]])
    tri = [{"el": "C", "xyz": [float(v) for v in f @ vec]} for f in frac]
    b.pert("cho_diamond_triclinic", "cho", CHO, tri, 0.05, 305, desc="sheared diamond 64 atoms, triclinic cell (restricted form)", cell={"vectors": vec.tolist(), "origin": [0, 0, 0]}, periodic=(True, True, True), tags=["periodic", "triclinic", "crystal", "perturbed", "images"], tol=1e-11)
    # periodic liquid-like water box: 8 molecules on a 2x2x2 lattice, random orientation (seeded)
    rng = np.random.default_rng(401)
    wat = []
    L = 9.6
    for i in range(2):
        for j in range(2):
            for k in range(2):
                w = g.water()
                axis = rng.normal(size=3)
                w = g.rotate(w, axis, float(rng.uniform(0, 360)))
                w = g.translate(w, [(i + 0.5) * L / 2, (j + 0.5) * L / 2, (k + 0.5) * L / 2])
                wat += w
    b.add("cho_water_box_8", "cho", CHO, wat, cell={"lo": [0, 0, 0], "hi": [L, L, L]}, periodic=(True, True, True),
          desc="8 H2O in a 9.6 Å periodic cube (random orientations, seed 401): condensed-phase hbond/Coulomb images", tags=["periodic", "hbond", "polar", "images", "liquid"], tol=1e-11,
          perturbation={"seed": 401, "amplitude": None, "note": "orientations random, positions on lattice"})
    # ---------------------------------------------------------------- tiny periodic cells: self-images, charged, 1-D chains
    nb = {"neighbor": {"skin": 1.0}}
    b.add("cho_chain_1atom_period1.30", "cho", CHO, [{"el": "C", "xyz": [0.5, 5.0, 5.0]}], cell={"lo": [0, 0, 0], "hi": [1.30, 20.0, 20.0]}, periodic=(True, False, False),
          desc="one C per 1.30 A period along x: every bond/angle/torsion partner is an image of the SAME atom (self-image accounting)", tags=["periodic", "self-image", "chain", "images", "small-cell"], extra=nb)
    zz = [{"el": "C", "xyz": [0.4, 5.0, 5.0]}, {"el": "C", "xyz": [1.55, 5.7, 5.0]}]
    b.pert("cho_chain_zigzag_period2.5", "cho", CHO, zz, 0.03, 310, desc="two C per 2.5 A period (zigzag)", cell={"lo": [0, 0, 0], "hi": [2.5, 20.0, 20.0]}, periodic=(True, False, False),
           tags=["periodic", "chain", "self-image", "images", "small-cell", "perturbed"], extra=nb)
    pe = []
    for cx, cz, sg in ((0.0, 0.4307, 1.0), (1.2765, -0.4307, -1.0)):
        pe.append({"el": "C", "xyz": [cx, 5.0, 5.0 + cz]})
        for sy in (1.0, -1.0):
            pe.append({"el": "H", "xyz": [cx, 5.0 + sy * 1.09 * np.sin(54.75 * np.pi / 180), 5.0 + cz + sg * 1.09 * np.cos(54.75 * np.pi / 180)]})
    b.pert("cho_polyethylene_period2.553", "cho", CHO, pe, 0.03, 311, desc="all-trans polyethylene, 2 CH2 per 2.553 A period: charged, bonded across the boundary",
           cell={"lo": [0, 0, 0], "hi": [2.553, 10.0, 10.0]}, periodic=(True, False, False), tags=["periodic", "chain", "images", "small-cell", "perturbed", "polar"], extra=nb, tol=1e-11)
    b.pert("cho_methane_cell6.0", "cho", CHO, g.translate(g.methane(), [3.0, 3.0, 3.0]), 0.05, 312, desc="one CH4 in a 6.0 A periodic cube (charged, many images)",
           cell={"lo": [0, 0, 0], "hi": [6.0] * 3}, periodic=(True,) * 3, tags=["periodic", "images", "small-cell", "perturbed", "polar"], tol=1e-11)
    b.pert("cho_water_cell6.2_hbond_self_image", "cho", CHO, g.translate(g.water(), [3.0, 3.0, 3.0]), 0.05, 313,
           desc="ONE H2O in a 6.2 A periodic cube: the acceptor O is a periodic image of the donor O; pinned LAMMPS drops that H-bond (tag comparison, Q-32)",
           cell={"lo": [0, 0, 0], "hi": [6.2] * 3}, periodic=(True,) * 3, tags=["periodic", "hbond", "images", "small-cell", "perturbed", "polar", "known-reference-quirk"], tol=1e-11)
    b.pert("cho_water_dimer_cell9.0", "cho", CHO, g.translate(g.water_dimer(), [4.5, 4.5, 4.5]), 0.05, 314, desc="(H2O)2 in a 9.0 A periodic cube (supercell-invariant control for Q-32)",
           cell={"lo": [0, 0, 0], "hi": [9.0] * 3}, periodic=(True,) * 3, tags=["periodic", "hbond", "images", "perturbed", "polar"], tol=1e-11)
    return b.cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffield-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); (out / "cases").mkdir(parents=True, exist_ok=True)
    cases = build(a.ffield_dir)
    for c in cases:
        (out / "cases" / (c["id"] + ".json")).write_text(json.dumps(c, indent=1, sort_keys=True) + "\n")
    man = ["# name\tsha256\tsource (relative to the pinned LAMMPS tree, third_party/lammps/PIN.txt)"]
    for f in sorted(set(c["ffield"]["name"] for c in cases)):
        man.append(f"{f}\t{sha(Path(a.ffield_dir) / f)}\tpotentials/{f}")
    (out / "ffield_manifest.tsv").write_text("\n".join(man) + "\n")
    print(f"wrote {len(cases)} cases and a manifest of {len(man) - 1} force fields to {out}")


if __name__ == "__main__":
    main()
