// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// The pair-side terms of the ACKS2 charge model (fix acks2/reaxff): the kinetic-energy coupling of the polarization energy and the geometry-dependent
// "bond softness" Coulomb term over the far list, as reaxff_nonbonded.cpp:36-60 and :214-265 of the pinned LAMMPS. The kinetic potentials s_i (the second
// half of the fix's solution vector, `s[N + i]` with N = nlocal + nghost) are INPUT. CPU-64; used for backend cpu64 and for the host-side addition to the
// Metal non-bonded kernel.
#include <vector>

#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/neighbor.hpp"
#include "reaxmetal/nonbonded.hpp"
#include "reaxmetal/system.hpp"

namespace reaxmetal {

// s_kin: nall entries (s[N + i]) ; q: charge per owned atom or per atom as for compute_nonbonded_core. Adds into `res` (Coulomb and Polarization energies,
// gradient, and the per-atom tallies if `res` already carries them).
void add_acks2_terms(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far, const std::vector<double>& q,
                     const double* s_kin, NonbondedResult& res);

}  // namespace reaxmetal
