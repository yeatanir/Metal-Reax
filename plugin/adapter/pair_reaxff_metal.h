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

#include "REAXFF/pair_reaxff.h"
#include "pair.h"

#include <memory>
#include <string>
#include <vector>

#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/metal_backend.hpp"
#include "reaxmetal/neighbor.hpp"
#include "reaxmetal/pair_settings.hpp"

namespace LAMMPS_NS {

class FixACKS2ReaxFF;

// The host view, device list and far rows of the current positions, built once and shared by the pair style and the GPU charge fix
struct NbView {
  reaxmetal::Box box;
  reaxmetal::AtomSet a;
  reaxmetal::NeighborCutoffs cut;
  reaxmetal::DeviceListInput list;
  std::shared_ptr<const reaxmetal::FarRowsF32> rows;
};

// Derived from the stock PairReaxFF on purpose: every stock command that does dynamic_cast<PairReaxFF *>(force->pair_match("^reax..")) -- fix reaxff/bonds, fix reaxff/species,
// compute reaxff/atom, compute spec/atom, and the stock charge fixes that borrow the pair style's neighbor list -- then works unmodified, provided this class fills the data
// those commands read (api->lists[BONDS], api->workspace->total_bond_order / nlp, api->control->bg_cut, tmpid / tmpbo, eletype, list). Nothing else of the base class is used:
// compute, settings, coeff, init_style, init_one, extract and setup are replaced.
class PairReaxFFMetal : public PairReaxFF {
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
  void setup() override {}                       // the stock setup() allocates the stock engine's data structures, which this style does not use
  double memory_usage() override { return 0.0; }

  // used by fix qeq/reaxff/metal (same plugin): the Metal context and the ghost-native host view
  bool uses_metal() const { return settings_.backend == "metal"; }
  reaxmetal::mtl::Context &metal_context();
  reaxmetal::NeighborCutoffs cutoffs() const;
  reaxmetal::Box host_box() const;
  reaxmetal::AtomSet host_atom_set(const reaxmetal::Box &box) const;
  // rebuilt only when the positions, box or atom counts differ from the cached ones (backend metal only)
  const NbView &nb_view();

 protected:
  mutable bool shell_warned_ = false;
  FixACKS2ReaxFF *acks2_fix_ = nullptr;   // the ACKS2 charge fix, if any: it supplies the kinetic potentials s (pair energy and force terms)
  std::unique_ptr<NbView> view_;
  std::vector<double> view_key_;
  reaxmetal::PairSettings settings_;
  std::unique_ptr<reaxmetal::ForceField> ff_;
  std::vector<int> map_;                       // LAMMPS type (1..ntypes) -> force field element index, -1 = NULL
  std::vector<double> chi_, eta_, gamma_, bcut_acks2_;
  double bond_softness_ = 0.0;                 // general parameter 35 (index 34), the ACKS2 bond softness      // extract() arrays, index 0..ntypes (index 0 unused)
  double cutmax_ = 0.0;
  std::unique_ptr<reaxmetal::mtl::Context> ctx_;
  std::vector<std::pair<std::string, double>> profile_;   // REAXMETAL_PROFILE=1: accumulated wall seconds per phase of compute(), printed at destruction
  void prof(const char *name, double seconds);   // Metal device context (backend metal), created on first use
  void allocate();
  bool need_list_ = false;                      // a stock fix / compute needs this pair style's half neighbor list with ghost rows (requested only then)
  bool need_bonds_ = false;                     // a stock analysis command needs the bond list (fix reaxff/bonds, reaxff/species, compute reaxff/atom, spec/atom)
  std::vector<double> bt_total_, bt_nlp_;       // api->workspace->total_bond_order / nlp storage
  int bond_list_cap_ = 0, bond_list_n_ = 0;     // capacity (entries, atoms) of api->lists[BONDS]
  void publish_bonds(const reaxmetal::BondTable &t);   // copy the engine's bond table into the stock structures the analysis commands read
  void free_bond_list();

  // ---- A2 (M3): the ghost-native view of the host data (LAMMPS_INTEGRATION section 4, rules C1-C3, C10) ----
  // host_atom_set: owned atoms [0,nlocal) then ghosts, exactly in LAMMPS' order; throws reaxmetal::SystemError when the host data violate the
  // contract (a ghost that is not owner + lattice shift, duplicate tags among owned atoms, a ghost shell narrower than required)
  // builds the engine's far list for that view and compares it row by row with LAMMPS' own half/newton-off/ghost list
  std::string selfcheck_summary(bool &ok);
};

}    // namespace LAMMPS_NS
#endif
