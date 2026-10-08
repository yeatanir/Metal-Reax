# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Self-tests of the M1 reference harness (no LAMMPS needed). They guard the independent checks against silent drift:
if the independent reference were wrong in the same way as the thing it checks, these would not catch it, so each test
compares against an analytic or brute-force quantity computed here by different means."""
import hashlib, itertools, json, math, os, sys, tempfile, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "reaxref"))
import runner, ref_nonbonded, ffield_edit, geometries  # noqa: E402


def synthetic_params(D=0.0, gamma_c=0.5):
    """minimal one-type parameter dump with the columns ref_nonbonded reads"""
    T = [0.0] * 27
    T[18], T[19], T[20], T[21], T[22], T[23], T[24], T[25], T[26] = D, 10.0, 3.0, 1.0, 1.5, 0.0, 9.0, 0.0, 1.0
    S = ["X"] + ["0"] * 20
    S[15], S[16] = "5.0", "7.0"
    G = [0.0] * 39; G[28] = 1.5
    return {"G": G, "meta": "#ntypes 1 vdw_type 1 n_global 39", "T": {(0, 0): T}, "T2": {(0, 0): [gamma_c, 0.0, 0.0]},
            "S": {0: S}}


class TaperTests(unittest.TestCase):
    def test_matches_textbook_polynomial_and_flat_ends(self):
        swb = 10.0
        c = ref_nonbonded.taper_poly(0.0, swb)
        for r in np.linspace(0.0, swb, 11):
            x = r / swb
            textbook = 1 - 35 * x ** 4 + 84 * x ** 5 - 70 * x ** 6 + 20 * x ** 7
            self.assertAlmostEqual(sum(c[k] * r ** k for k in range(8)), textbook, places=12)
        # first three derivatives vanish at both ends
        for end in (0.0, swb):
            for d in (1, 2, 3):
                v = sum(c[k] * math.perm(k, d) * end ** (k - d) for k in range(d, 8))
                self.assertAlmostEqual(v, 0.0, places=8)

    def test_nonzero_lower_radius(self):
        c = ref_nonbonded.taper_poly(2.0, 9.0)
        val = lambda r: sum(c[k] * r ** k for k in range(8))
        self.assertAlmostEqual(val(2.0), 1.0, places=9); self.assertAlmostEqual(val(9.0), 0.0, places=9)


class SelfImageCountTests(unittest.TestCase):
    def test_single_atom_cubic_cell_pair_count_and_coulomb(self):
        L, swb = 3.0, 7.0
        P = synthetic_params(D=0.0)
        ctrl = {"nonb_low": 0.0, "nonb_cut": swb}
        out = ref_nonbonded.nonbonded_reference(np.zeros((1, 3)), np.array([0]), np.array([1.0]), np.eye(3) * L, [True] * 3, P, ctrl, False)
        # brute force: lattice vectors n != 0 with |n| L <= swb; physical pairs = half of them
        vecs = [n for n in itertools.product(range(-4, 5), repeat=3) if n != (0, 0, 0) and L * math.sqrt(sum(v * v for v in n)) <= swb]
        self.assertEqual(out["n_pairs"], len(vecs) // 2)
        self.assertEqual(out["n_pairs_self"], len(vecs) // 2)
        c = ref_nonbonded.taper_poly(0.0, swb)
        e = 0.0
        for n in vecs:
            r = L * math.sqrt(sum(v * v for v in n))
            e += 0.5 * ref_nonbonded.C_ELE * sum(c[k] * r ** k for k in range(8)) / (r ** 3 + 0.5) ** ref_nonbonded.CBRT_EXP
        self.assertAlmostEqual(out["e_ele"], e, places=9)

    def test_two_atom_molecule_counts_one_pair(self):
        x = np.array([[0, 0, 0], [0, 0, 1.2]], float)
        out = ref_nonbonded.nonbonded_reference(x, np.array([0, 0]), np.array([0.3, -0.3]), np.eye(3) * 50.0, [False] * 3, synthetic_params(), {"nonb_low": 0.0, "nonb_cut": 10.0}, False)
        self.assertEqual(out["n_pairs"], 1)


class EemResidualTests(unittest.TestCase):
    def test_solution_has_zero_residual_and_perturbation_does_not(self):
        x = np.array([[0, 0, 0], [0, 0, 1.3], [1.1, 0.2, 0.4]], float)
        chi = np.array([5.0, 7.0, 6.0]); eta = np.array([14.0, 13.0, 12.0]); gam = np.array([0.8, 0.9, 0.7])
        swb = 10.0
        tap = runner.taper_coef(swb)
        n = 3; A = np.diag(eta)
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = np.linalg.norm(x[j] - x[i]); A[i, j] += 14.4 * tap(r) / (r ** 3 + (gam[i] * gam[j]) ** -1.5) ** (1 / 3)
        # minimise chi.q + q.A.q/2 with sum q = 0 : [A 1; 1 0][q; mu] = [-chi; 0]
        K = np.block([[A, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
        sol = np.linalg.solve(K, np.concatenate([-chi, [0.0]])); q = sol[:n]
        res, mu, sq = runner.eem_residual(x, None, chi, eta, gam, q, np.eye(3) * 100.0, [False] * 3, swb)
        self.assertLess(res, 1e-12); self.assertLess(abs(sq), 1e-12)
        q2 = q + np.array([1e-3, -1e-3, 0.0])
        self.assertGreater(runner.eem_residual(x, None, chi, eta, gam, q2, np.eye(3) * 100.0, [False] * 3, swb)[0], 1e-5)


class CellTests(unittest.TestCase):
    def test_restricted_triclinic_roundtrip(self):
        v = [[7.0, 0, 0], [1.2, 6.5, 0], [0.8, 1.0, 6.9]]
        lo, hi, tilt = runner.cell_to_lammps({"vectors": v})
        self.assertEqual(list(tilt), [1.2, 0.8, 1.0]); self.assertEqual(list(hi - lo), [7.0, 6.5, 6.9])
        np.testing.assert_allclose(runner.cell_matrix({"vectors": v}), np.array(v))

    def test_non_restricted_cell_is_refused(self):
        with self.assertRaises(ValueError):
            runner.cell_to_lammps({"vectors": [[7.0, 0.5, 0], [1.2, 6.5, 0], [0.8, 1.0, 6.9]]})


class FFieldEditTests(unittest.TestCase):
    TEXT = """header
 2    ! Nr of bonds; x
         p(be2)
  1  1 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
         a b c d e f
  1  2 1.1 2.1 3.1 4.1 5.1 6.1 7.1 8.1
         a b c d e f
  2    ! Nr of angles
  1  1  1  1.0 2.0 3.0 4.0 5.0 6.0 7.0
  1  2  1  1.5 2.5 3.5 4.5 5.5 6.5 7.5
  0    ! Nr of torsions
"""

    def test_angle_block_edit(self):
        L, k, ang = ffield_edit.angle_block(self.TEXT)
        self.assertEqual(len(ang), 2)
        out = ffield_edit.set_angle_lines(self.TEXT, ang + [ffield_edit.fmt_angle(2, 2, 2, 1, 2, 3, 4, 5, 6, 7)])
        _, _, ang2 = ffield_edit.angle_block(out)
        self.assertEqual(len(ang2), 3); self.assertEqual(ang2[:2], ang)
        self.assertIn("Nr of torsions", out)

    def test_drop_bond_pair(self):
        out = ffield_edit.drop_bond_pair(self.TEXT, 2, 1)
        self.assertTrue(out.lstrip().startswith("header"))
        self.assertEqual([int(l.split()[0]) for l in out.splitlines() if "Nr of bonds" in l], [1])
        self.assertNotIn("1.1 2.1", out); self.assertIn("1.0 2.0", out)
        with self.assertRaises(AssertionError):
            ffield_edit.drop_bond_pair(self.TEXT, 3, 3)


class GeometryTests(unittest.TestCase):
    def test_bond_lengths_and_angles(self):
        w = geometries.water()
        X = np.array([a["xyz"] for a in w])
        self.assertAlmostEqual(np.linalg.norm(X[1] - X[0]), 0.9572, places=9)
        v1, v2 = X[1] - X[0], X[2] - X[0]
        self.assertAlmostEqual(math.degrees(math.acos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2))), 104.52, places=6)
        m = np.array([a["xyz"] for a in geometries.methane()])
        d = [np.linalg.norm(m[k] - m[0]) for k in range(1, 5)]
        self.assertTrue(all(abs(x - 1.09) < 1e-12 for x in d))

    def test_diamond_density(self):
        at, cell = geometries.diamond(3.567, 2)
        self.assertEqual(len(at), 64); self.assertAlmostEqual(cell["hi"][0], 7.134)


class FixtureTests(unittest.TestCase):
    def test_every_case_is_well_formed(self):
        cases = sorted((ROOT / "tests" / "fixtures" / "cases").glob("*.json"))
        if not cases: self.skipTest("no fixtures generated")
        man = {l.split("\t")[0]: l.split("\t")[1] for l in (ROOT / "tests/fixtures/ffield_manifest.tsv").read_text().splitlines() if l and not l.startswith("#")}
        ids = set()
        for f in cases:
            c = json.loads(f.read_text())
            self.assertEqual(c["id"], f.stem); self.assertNotIn(c["id"], ids); ids.add(c["id"])
            self.assertEqual(man[c["ffield"]["name"]], c["ffield"]["sha256"])
            for a in c["atoms"]: self.assertIn(a["el"], c["elements"])
            lo, hi, _ = runner.cell_to_lammps(c["cell"])
            self.assertTrue(np.all(hi > lo))
            self.assertEqual(len(c["periodic"]), 3)
        self.assertGreaterEqual(len(ids), 50)

    def test_golden_references_match_cases(self):
        gdir = ROOT / "tests" / "fixtures" / "reference"
        if not gdir.exists(): self.skipTest("no golden references committed yet")
        for g in sorted(gdir.glob("*.json")):
            ref = json.loads(g.read_text())
            case = json.loads((ROOT / "tests/fixtures/cases" / (ref["case_id"] + ".json")).read_text())
            self.assertEqual(ref["case_sha256"], hashlib.sha256(runner.canonical_json(case)).hexdigest(), g.name)
            n = len(case["atoms"])
            self.assertEqual(len(ref["charges"]), n); self.assertEqual(len(ref["forces"]), n)
            self.assertTrue(ref["valid"]); self.assertEqual(len(ref["energy"]["slots"]), 14)
            self.assertLess(abs(sum(ref["energy"]["slots"].values()) - ref["energy"]["total_pe"]), 1e-8 * max(1.0, abs(ref["energy"]["total_pe"])))
            self.assertTrue(all(math.isfinite(v) for row in ref["forces"] for v in row))
            self.assertEqual(ref["provenance"]["lammps_pinned_commit"], "8de817dd79bfe4525d5d39246a212d833e6dee07")

    def test_tolerance_file_matches_policy(self):
        f = ROOT / "tolerances" / "tolerances.json"
        if not f.exists(): self.skipTest("tolerances not frozen yet")
        t = json.loads(f.read_text())
        c3 = t["C3"]    # owner-set values, NUMERICAL_POLICY 5.2 (copied here deliberately: changing one without the other fails)
        self.assertEqual(c3["energy_per_atom_abs_kcal_mol"], 1e-3); self.assertEqual(c3["energy_significant_category_rel"], 1e-5)
        self.assertEqual(c3["force_rms_kcal_mol_A"], 5e-3); self.assertEqual(c3["force_max_component_kcal_mol_A"], 5e-2)
        self.assertEqual(c3["charge_max_abs_e"], 1e-4); self.assertEqual(c3["charge_rms_e"], 2e-5)


if __name__ == "__main__":
    unittest.main()
