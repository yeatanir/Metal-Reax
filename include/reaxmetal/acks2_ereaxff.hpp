// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// ACKS2 charges inside the pair style for eReaxFF mode (pair_style keyword `acks2 yes`), CPU-64, direct solve.
// Re-implemented from the equations of the legacy routine `chargesa` (Margaret/eReaxFF_src/reac.f:4041-4300, "compact" version, and the
// full-system version of the Si source); no legacy code is copied.
//
// Which particles are solved is DATA: SingleBody::flags.charge_solve (eparams `charge=solve|fixed`; default: every species that is not an electron).
// Fixed-charge particles keep their data-file charge and stay OUT of the matrix: their Coulomb field is folded into the right-hand side of the solved
// atoms (the compact legacy treatment), they have no kinetic potential (s = 0). If every particle is solved this is the full-system legacy ACKS2.
//
// Unknowns (m = number of solved atoms, G = number of charge groups):  q (m), s (m), the multiplier of  sum s = 0,  one multiplier per group.
//   row q_i :  2 eta_i q_i + sum_j H_ij q_j + s_i + mu_g(i)           = -chi_i - sum_{j fixed} H_ij q_j
//   row s_i :  q_i + (X s)_i + lambda                                  = Q_g(i) / n_g(i)
//   row  S  :  sum_i s_i                                               = 0
//   row  g  :  sum_{i in g} q_i                                        = Q_g
// H_ij = sum over images of Tap(r) * dielec / (r^3 + gamma_ij)^(1/3); X has the bond-softness bs(r) on the off-diagonal and -sum bs on the diagonal.
// A group is a set of solved atoms with a fixed total charge Q_g (the legacy MOLCHARGE line); the pair style forms the groups from the molecule id.
#include <vector>

#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/neighbor.hpp"
#include "reaxmetal/system.hpp"

namespace reaxmetal {

struct Acks2Options {
  // true: the numeric constants of the legacy executable (single-precision literals dielec = 14.40, evkcmol = 23.02, exponent 1/3 as REAL, and the
  // self-image diagonal of the matrix with half the weight that the Coulomb energy implies). false: constants consistent with the energy terms
  // (dielec = c1c / evkcmol, exponent 1/3 in double, self-image weight consistent), so that the forces are the exact gradient of the energy.
  bool legacy_constants = true;
  int soft_fixed = -1;     // -1: follows legacy_constants (legacy: no kinetic block for pairs with a fixed particle; consistent: yes). 0/1 forces it (analysis).
};

struct Acks2Solution {
  std::vector<double> q;   // nlocal: solved charge, or the input charge for fixed particles
  std::vector<double> s;   // nlocal: kinetic potential (eV), 0 for fixed particles
  std::vector<double> ref; // nlocal: ACKS2 reference charge Q_g / n_g of solved atoms (0 for fixed)
  std::size_t nsolved = 0;
  std::size_t neq = 0;
  double residual = 0.0;   // max |A x - b| of the solved system (eV)
};

constexpr double kLegacyEvKcal = static_cast<double>(23.02f);   // evkcmol = 23.02 as a default-REAL literal (reac.f chargesa, poten.f nonbonacks2)

// q_in: nlocal data-file charges (used for fixed particles and, through group_q, nothing else). group: nlocal entries, the group index (0..G-1) of every
// SOLVED atom (ignored for fixed ones). group_q: G total charges. Throws SystemError for a singular system.
Acks2Solution solve_acks2_ereaxff(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far, const std::vector<double>& q_in,
                                  const std::vector<int>& group, const std::vector<double>& group_q, const Acks2Options& opt = {});

// The charge-dependent self energy of the solved atoms, evkcmol * (chi + eta q) q  (legacy `ech`, without the coupling q*s, which add_acks2_terms supplies).
double acks2_self_energy(const ForceField& ff, const AtomSet& atoms, const std::vector<double>& q, double evkcmol);

}  // namespace reaxmetal
