/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan */
#include "fix_qeq_reaxff_metal.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "pair_reaxff_metal.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

using namespace LAMMPS_NS;

namespace {
// REAXMETAL_PROFILE=1: wall seconds per phase of the GPU charge fix, printed at exit
struct Prof {
  double view = 0, rows = 0, setup = 0, mv = 0;
  long nmv = 0;
  bool on = std::getenv("REAXMETAL_PROFILE") != nullptr;
  ~Prof() { if (on) std::fprintf(stderr, "[qeq/reaxff/metal] host_view %.3f  input+rows %.3f  setup %.3f  matvec %.3f (%ld calls)\n", view, rows, setup, mv, nmv); }
};
Prof prof;
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}    // namespace

void FixQEqReaxFFMetal::compute_H()
{
  pair_ = dynamic_cast<PairReaxFFMetal *>(force->pair_match("^reaxff/metal", 0));
  gpu_ = false;
  if (pair_ && pair_->uses_metal() && igroup == 0 && comm->nprocs == 1) {
    try {
      using namespace reaxmetal;
      auto &ctx = pair_->metal_context();
      double t0 = now();
      const NbView &view = pair_->nb_view();
      const Box &box = view.box;
      const AtomSet &a = view.a;
      const NeighborCutoffs &cut = view.cut;
      prof.view += now() - t0; t0 = now();
      if (swb <= cut.nonb) {
        const std::size_t nall = a.nall();
        std::vector<int> types(atom->type, atom->type + nall);
        const QeqDeviceInput in = make_qeq_device_input(cut, a, box, types, eta, gamma, atom->ntypes, swa, swb,
                                                        [&](const DeviceListInput &) { return FarRowsF32{}; }, &view.list, view.rows);
        prof.rows += now() - t0; t0 = now();
        ctx.qeq_setup(in);
        prof.setup += now() - t0;
        gpu_ = true;
      }
    } catch (const std::exception &e) {
      error->all(FLERR, "Fix qeq/reaxff/metal: {}", e.what());
    }
  }
  if (!gpu_) {
    if (!warned_ && comm->me == 0) error->warning(FLERR, "Fix qeq/reaxff/metal is using the stock CPU matrix (needs pair reaxff/metal backend metal, one MPI rank, group all, taper radius <= nonb_cut)");
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
    const double t0 = now();
    pair_->metal_context().qeq_matvec(x, b);
    prof.mv += now() - t0; ++prof.nmv;
  } catch (const std::exception &e) {
    error->all(FLERR, "Fix qeq/reaxff/metal: {}", e.what());
  }
  const int nall = atom->nlocal + atom->nghost;
  for (int i = atom->nlocal; i < nall; ++i) b[i] = 0.0;   // ghost sums are folded into the owners inside the kernel
}
