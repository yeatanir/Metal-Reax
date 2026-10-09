// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// CPU-64 reference engine for the bond-order based terms (milestone M4, phase A): bond list, bond orders, bond energy, lone pair
// (with the C2 correction), over- and under-coordination, and the gradient of exactly those terms.
// It follows the pinned LAMMPS traversal (reaxff_forces.cpp Init_Forces_noQEq, reaxff_bond_orders.cpp BO, reaxff_bonds.cpp,
// reaxff_multi_body.cpp, Add_dBond_to_Forces) because the reference's bond_mark rule and owner-computes rules are order dependent
// (ENGINE_SPEC 3.1, 4, 5). Pure formulas are in terms.hpp (shared with the CPU-32 twin and, later, Metal).
// Not yet included: valence angle, torsion, H-bond, nonbonded (later phases). `BondedResult::grad` is the gradient dE/dx of the
// computed terms only; the physical force is -grad (ENGINE_SPEC 1).
#include <array>
#include <cstddef>
#include <vector>

#include "reaxmetal/census.hpp"
#include "reaxmetal/energy_terms.hpp"
#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/neighbor.hpp"
#include "reaxmetal/system.hpp"

namespace reaxmetal {

struct BondedOptions {
  bool enobonds = true;  // pair_style reaxff enobonds yes (default): lone pair / under-coordination energy for atoms without bonds
  bool census = false;   // fill BondedResult::census
  bool per_atom = false; // fill eatom / vatom with the reference's per-atom tallies (ev_tally*, v_tally*; owned and ghost atoms, the host folds ghosts)
};

struct BondedStats {
  std::size_t bonds = 0;          // undirected bonds in the list (owned + ghost rows)
  std::size_t max_bonds_per_atom = 0;
};

struct BondedResult {
  EnergyBreakdown e;              // Bond, LonePair, Over, Under filled (others stay 0)
  std::vector<double> grad;       // 3 * nall, dE/dx of those terms, ghosts included (host folds ghost onto owner)
  std::vector<double> total_bo;   // per atom corrected total bond order (nall)
  DecisionCensus census;          // bonds, angle_sets, torsions, hbonds (nlocal entries), only with census
  std::vector<double> eatom;      // per-atom energy (nall), only with per_atom
  std::vector<std::array<double, 6>> vatom;   // per-atom virial xx yy zz xy xz yz (nall), only with per_atom
  BondedStats stats;
};

// `ctl` supplies bond_cut (control-file/default); bo_cut comes from the force field general parameters.
BondedResult compute_bonded_core(const ForceField& ff, const ControlParams& ctl, const AtomSet& atoms, const FarList& far,
                                 const BondedOptions& opt = {});

// The same engine compiled in single precision (CPU-32 twin, FP32-1): inputs rounded to float on entry, results widened to double.
BondedResult compute_bonded_core_fp32(const ForceField& ff, const ControlParams& ctl, const AtomSet& atoms, const FarList& far,
                                      const BondedOptions& opt = {});

}  // namespace reaxmetal
