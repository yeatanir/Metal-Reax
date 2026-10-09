/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan */
#include "fix_qeq_reaxff_metal.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "pair_reaxff_metal.h"

#include <cstring>
#include <exception>

using namespace LAMMPS_NS;

void FixQEqReaxFFMetal::compute_H()
{
  pair_ = dynamic_cast<PairReaxFFMetal *>(force->pair_match("^reaxff/metal", 0));
  gpu_ = false;
  if (pair_ && pair_->uses_metal() && igroup == 0) {
    try {
      using namespace reaxmetal;
      auto &ctx = pair_->metal_context();
      const Box box = pair_->host_box();
      const AtomSet a = pair_->host_atom_set(box);
      const NeighborCutoffs cut = pair_->cutoffs();
      if (swb <= cut.nonb) {
        const std::size_t nall = a.nall();
        std::vector<int> types(atom->type, atom->type + nall);
        const QeqDeviceInput in = make_qeq_device_input(cut, a, box, types, eta, gamma, atom->ntypes, swa, swb,
                                                        [&](const DeviceListInput &l) { return ctx.far_rows(l); });
        ctx.qeq_setup(in);
        gpu_ = true;
      }
    } catch (const std::exception &e) {
      error->all(FLERR, "Fix qeq/reaxff/metal: {}", e.what());
    }
  }
  if (!gpu_) {
    if (!warned_ && comm->me == 0) error->warning(FLERR, "Fix qeq/reaxff/metal is using the stock CPU matrix (needs pair reaxff/metal backend metal, group all, taper radius <= nonb_cut)");
    warned_ = true;
    FixQEqReaxFF::compute_H();
  }
}

void FixQEqReaxFFMetal::sparse_matvec(sparse_matrix *A, double *x, double *b)
{
  if (!gpu_) {
    FixQEqReaxFF::sparse_matvec(A, x, b);
    return;
  }
  try {
    pair_->metal_context().qeq_matvec(x, b);
  } catch (const std::exception &e) {
    error->all(FLERR, "Fix qeq/reaxff/metal: {}", e.what());
  }
  const int nall = atom->nlocal + atom->nghost;
  for (int i = atom->nlocal; i < nall; ++i) b[i] = 0.0;   // ghost sums are folded into the owners inside the kernel
}
