/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan
 *
 * fix qeq/reaxff/metal: fix qeq/reaxff with the EEM matrix assembly and the matrix-vector product on the Metal GPU (FP32, no atomics, fixed
 * summation order). The preconditioned CG, the history extrapolation and the charge update are the stock fix's, in double. Derived from
 * LAMMPS FixQEqReaxFF (pinned stable_30Sep2026 @ 8de817dd, GPL-2.0). Falls back to the stock CPU path when the pair style is not
 * reaxff/metal with 'backend metal', for fix groups other than 'all', or when the taper radius exceeds nonb_cut.
 */
#ifndef REAXMETAL_FIX_QEQ_REAXFF_METAL_H
#define REAXMETAL_FIX_QEQ_REAXFF_METAL_H

#include "REAXFF/fix_qeq_reaxff.h"

namespace LAMMPS_NS {

class PairReaxFFMetal;

class FixQEqReaxFFMetal : public FixQEqReaxFF {
 public:
  FixQEqReaxFFMetal(class LAMMPS *lmp, int narg, char **arg) : FixQEqReaxFF(lmp, narg, arg) {}

 protected:
  void compute_H() override;
  void sparse_matvec(sparse_matrix *, double *, double *) override;

 private:
  PairReaxFFMetal *pair_ = nullptr;
  bool gpu_ = false;
  bool warned_ = false;
};

}    // namespace LAMMPS_NS
#endif
