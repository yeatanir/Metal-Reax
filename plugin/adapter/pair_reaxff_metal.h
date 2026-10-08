/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan
 *
 * ReaxMetal adapter A1 (milestone M2): `pair_style reaxff/metal`. Provides what the ReaxFF family of LAMMPS styles must provide
 * to its surroundings -- argument parsing, `pair_coeff` element mapping, the force-field tables (via reaxmetal's own parser),
 * `extract("chi"|"eta"|"gamma")` for the charge fixes, and the host checks -- but NO physics: compute() refuses with an explicit
 * error until the force backend exists (M4). Derived from the LAMMPS Pair interface (pinned stable_30Sep2026 @ 8de817dd,
 * GPL-2.0); structure after examples/plugins/pair_morse2.h (see plugin/probe/pair_reaxff_metal_probe.h for the notice).
 */
#ifndef REAXMETAL_PAIR_REAXFF_METAL_H
#define REAXMETAL_PAIR_REAXFF_METAL_H

#include "pair.h"

#include <memory>
#include <vector>

#include "reaxmetal/forcefield.hpp"
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

 protected:
  reaxmetal::PairSettings settings_;
  std::unique_ptr<reaxmetal::ForceField> ff_;
  std::vector<int> map_;                       // LAMMPS type (1..ntypes) -> force field element index, -1 = NULL
  std::vector<double> chi_, eta_, gamma_;      // extract() arrays, index 0..ntypes (index 0 unused)
  double cutmax_ = 0.0;
  void allocate();
};

}    // namespace LAMMPS_NS
#endif
