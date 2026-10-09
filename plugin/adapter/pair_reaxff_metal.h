/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan
 *
 * ReaxMetal adapter A1 (milestone M2) + A2 host view (milestone M3): `pair_style reaxff/metal`. Provides what the ReaxFF family of LAMMPS styles must provide
 * to its surroundings -- argument parsing, `pair_coeff` element mapping, the force-field tables (via reaxmetal's own parser),
 * `extract("chi"|"eta"|"gamma")` for the charge fixes, and the host checks -- but NO physics: compute() refuses with an explicit
 * error until the force backend exists (M4). Derived from the LAMMPS Pair interface (pinned stable_30Sep2026 @ 8de817dd,
 * GPL-2.0); structure after examples/plugins/pair_morse2.h (see plugin/probe/pair_reaxff_metal_probe.h for the notice).
 */
#ifndef REAXMETAL_PAIR_REAXFF_METAL_H
#define REAXMETAL_PAIR_REAXFF_METAL_H

#include "pair.h"

#include <memory>
#include <string>
#include <vector>

#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/metal_backend.hpp"
#include "reaxmetal/neighbor.hpp"
#include "reaxmetal/pair_settings.hpp"

namespace LAMMPS_NS {

class PairReaxFFMetal : public Pair {
 public:
  explicit PairReaxFFMetal(class LAMMPS *);
  ~PairReaxFFMetal() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void *extract(const char *, int &) override;
  void init_list(int, class NeighList *) override;

  // used by fix qeq/reaxff/metal (same plugin): the Metal context and the ghost-native host view
  bool uses_metal() const { return settings_.backend == "metal"; }
  reaxmetal::mtl::Context &metal_context();
  reaxmetal::NeighborCutoffs cutoffs() const;
  reaxmetal::Box host_box() const;
  reaxmetal::AtomSet host_atom_set(const reaxmetal::Box &box) const;

 protected:
  reaxmetal::PairSettings settings_;
  std::unique_ptr<reaxmetal::ForceField> ff_;
  std::vector<int> map_;                       // LAMMPS type (1..ntypes) -> force field element index, -1 = NULL
  std::vector<double> chi_, eta_, gamma_;      // extract() arrays, index 0..ntypes (index 0 unused)
  double cutmax_ = 0.0;
  std::unique_ptr<reaxmetal::mtl::Context> ctx_;
  std::vector<std::pair<std::string, double>> profile_;   // REAXMETAL_PROFILE=1: accumulated wall seconds per phase of compute(), printed at destruction
  void prof(const char *name, double seconds);   // Metal device context (backend metal), created on first use
  void allocate();

  // ---- A2 (M3): the ghost-native view of the host data (LAMMPS_INTEGRATION section 4, rules C1-C3, C10) ----
  // host_atom_set: owned atoms [0,nlocal) then ghosts, exactly in LAMMPS' order; throws reaxmetal::SystemError when the host data violate the
  // contract (a ghost that is not owner + lattice shift, duplicate tags among owned atoms, a ghost shell narrower than required)
  // builds the engine's far list for that view and compares it row by row with LAMMPS' own half/newton-off/ghost list
  std::string selfcheck_summary(bool &ok);
};

}    // namespace LAMMPS_NS
#endif
