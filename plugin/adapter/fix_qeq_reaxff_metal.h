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

#include <string>
#include <vector>

namespace LAMMPS_NS {

class PairReaxFFMetal;

// Extra keywords of this fix, split off before the stock constructor (which rejects unknown keywords):
//   strict          non-convergence of the CG is an error (stock: a warning, and the run continues); the taper radius must lie inside the
//                   ghost shell (stock: silently truncates the matrix, ENGINE_SPEC Q-35)
//   resident        the EEM solve runs on the GPU (single rank): residuals and matrix in double-single arithmetic, correction solves by an FP32 CG on the device,
//                   so the requested tolerance (down to 1e-11) is reached
//   verify <eV>     after every solve the equalisation residual max_i |(H q)_i + chi_i - mu| is evaluated with the matrix of the solve and must not
//                   exceed <eV> (an error otherwise); implies nothing else
struct QeqExtraArgs {
  std::vector<char *> filtered;
  bool strict = false;
  bool resident = false;
  double verify_ev = 0.0;
  std::string problem;
  QeqExtraArgs(int narg, char **arg);
};

class FixQEqReaxFFMetal : private QeqExtraArgs, public FixQEqReaxFF {
 public:
  FixQEqReaxFFMetal(class LAMMPS *lmp, int narg, char **arg) :
      QeqExtraArgs(narg, arg), FixQEqReaxFF(lmp, static_cast<int>(filtered.size()), filtered.data()) { reaxff = nullptr; }   // use the fix's own list, not the pair style's
  void init() override;
  void pre_force(int) override;

 protected:
  void compute_H() override;
  void sparse_matvec(sparse_matrix *, double *, double *) override;
  int CG(double *, double *) override;
  void calculate_Q() override;

 private:
  PairReaxFFMetal *pair_ = nullptr;
  bool gpu_ = false;
  bool warned_ = false;
  bool warned_tol_ = false;
  bool shell_checked_ = false;
  bool resident_ok_ = false;     // this step: resident solve possible
  int resident_iters_t_ = 0;
};

}    // namespace LAMMPS_NS
#endif
