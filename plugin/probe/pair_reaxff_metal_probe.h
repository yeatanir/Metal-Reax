/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan
 *
 * SPIKE / SCAFFOLDING (M0.5) -- NOT the production adapter. Contains no ReaxFF physics.
 * Purpose: empirically record the LAMMPS host contract (ghosts, neighbor lists, newton, virial, charge-fix
 * coupling) for a Pair-derived style named "reaxff/metal". Derived from the LAMMPS Pair interface
 * (pinned stable_30Sep2026 @ 8de817dd79bfe4525d5d39246a212d833e6dee07); links against GPL-2.0 LAMMPS.
 * Adapted-from: examples/plugins/pair_morse2.h @ 8de817dd79bfe4525d5d39246a212d833e6dee07 (structure of a Pair-derived plugin class;
 *   no morse2 physics retained)
 * Structure follows the LAMMPS plugin example examples/plugins/pair_morse2.h (GPL-2.0, Sandia Corporation notice,
 * "Copyright (2003) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
 * the U.S. Government retains certain rights in this software.").
 */
#ifndef REAXMETAL_PAIR_REAXFF_METAL_PROBE_H
#define REAXMETAL_PAIR_REAXFF_METAL_PROBE_H

#include "pair.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

class PairReaxFFMetalProbe : public Pair {
 public:
  explicit PairReaxFFMetalProbe(class LAMMPS *);
  ~PairReaxFFMetalProbe() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  void init_list(int, class NeighList *) override;
  double init_one(int, int) override;
  void *extract(const char *, int &) override;

 protected:
  enum NeighMode { NONE, HALF, HALF_NEWTON_OFF, FULL, FULL_GHOST, HALF_NEWTON_OFF_GHOST };
  NeighMode neigh_mode;
  double cutmax;
  bool ghostforce, verbose;
  std::vector<int> map;                      // LAMMPS type -> ffield element index (-1 = none)
  std::vector<double> chi_, eta_, gamma_;    // per LAMMPS type, index 1..ntypes (scaffolding parse)
  bigint last_report;
  void allocate();
  void report(int eflag, int vflag);
};

}    // namespace LAMMPS_NS
#endif
