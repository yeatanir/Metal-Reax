// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// CPU-64 reference for the nonbonded terms (milestone M5): van der Waals (types 1/2/3, lg), shielded Coulomb with frozen charges, and the
// polarization (self) energy. Owner-computes counting and traversal as pinned LAMMPS vdW_Coulomb_Energy (reaxff_nonbonded.cpp:62-215):
// every owned row, entries with d <= nonb_cut, counted by classify_nonbonded_entry (neighbor.hpp); the gradient goes onto i and j (ghosts included).
// Charges are INPUT (stock `fix qeq/reaxff` solves them in LAMMPS, ADR-015); ghosts use their owner's charge.
#include <array>
#include <vector>

#include "reaxmetal/energy_terms.hpp"
#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/neighbor.hpp"
#include "reaxmetal/system.hpp"

namespace reaxmetal {

struct NonbondedOptions {
  bool lgvdw = false;  // pair_style reaxff lgvdw yes
  bool per_atom = false;   // fill eatom / vatom (ev_tally of the reference: half of each pair term to each atom)
};

struct NonbondedResult {
  EnergyBreakdown e;              // VdW, Coulomb, Polarization filled
  std::vector<double> grad;       // 3 * nall, dE/dx (physical force = -grad)
  std::size_t pairs = 0;          // counted pairs
  std::vector<double> eatom;      // nall, only with per_atom
  std::vector<std::array<double, 6>> vatom;   // nall, only with per_atom
};

// q: charge per OWNED atom (nlocal entries; ghosts use their owner's) or per atom (nall entries, the multi-rank form).
NonbondedResult compute_nonbonded_core(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far,
                                       const std::vector<double>& q, const NonbondedOptions& opt = {});

NonbondedResult compute_nonbonded_core_fp32(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far,
                                           const std::vector<double>& q, const NonbondedOptions& opt = {});

}  // namespace reaxmetal
