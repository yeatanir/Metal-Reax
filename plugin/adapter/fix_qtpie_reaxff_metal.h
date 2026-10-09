/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan
 *
 * fix qtpie/reaxff/metal: the stock QTPIE charge model (derived from FixQtpieReaxFF, pinned stable_30Sep2026 @ 8de817dd, GPL-2.0) for use with pair_style
 * reaxff/metal. The stock fix borrows the half neighbor list WITH ghost rows from pair reaxff; that pair style is not a PairReaxFF here, so this class requests
 * the same kind of list for itself (the stock fallback list has no ghost rows and gives wrong effective electronegativities).
 */
#ifndef REAXMETAL_FIX_QTPIE_REAXFF_METAL_H
#define REAXMETAL_FIX_QTPIE_REAXFF_METAL_H

#include "REAXFF/fix_qeq_rel_reaxff.h"
#include "REAXFF/fix_qtpie_reaxff.h"

namespace LAMMPS_NS {

class FixQtpieReaxFFMetal : public FixQtpieReaxFF {
 public:
  FixQtpieReaxFFMetal(class LAMMPS *lmp, int narg, char **arg) : FixQtpieReaxFF(lmp, narg, arg) {}
  void init() override;
  void init_list(int, class NeighList *) override;
};

// fix qeq/rel/reaxff/metal: QEq-R (stock FixQEqRelReaxFF is a FixQtpieReaxFF with another effective electronegativity) with the same neighbor-list request
class FixQEqRelReaxFFMetal : public FixQEqRelReaxFF {
 public:
  FixQEqRelReaxFFMetal(class LAMMPS *lmp, int narg, char **arg) : FixQEqRelReaxFF(lmp, narg, arg) {}
  void init() override;
  void init_list(int, class NeighList *) override;
};

}    // namespace LAMMPS_NS
#endif
