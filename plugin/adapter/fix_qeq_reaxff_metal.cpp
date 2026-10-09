/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan */
#include "fix_qeq_reaxff_metal.h"

#include "atom.h"
#include "comm.h"
#include "update.h"
#include "error.h"
#include "force.h"
#include "pair_reaxff_metal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

QeqExtraArgs::QeqExtraArgs(int narg, char **arg)
{
  for (int i = 0; i < narg; ++i) {
    if (i >= 8 && std::strcmp(arg[i], "strict") == 0) strict = true;
    else if (i >= 8 && std::strcmp(arg[i], "verify") == 0) {
      if (i + 1 >= narg) { problem = "keyword verify needs a residual in eV"; break; }
      verify_ev = std::atof(arg[++i]);
      if (!(verify_ev > 0.0)) problem = "keyword verify needs a positive residual in eV";
    } else filtered.push_back(arg[i]);
  }
}

void FixQEqReaxFFMetal::init()
{
  if (!problem.empty()) error->all(FLERR, "Fix qeq/reaxff/metal: {}", problem);
  FixQEqReaxFF::init();
}

void FixQEqReaxFFMetal::pre_force(int vflag)
{
  if (strict && !shell_checked_) {   // the ghost shell exists only after the first setup
    shell_checked_ = true;
    if (swb > comm->get_comm_cutoff())
      error->all(FLERR, "Fix qeq/reaxff/metal strict: taper radius {} exceeds the ghost cutoff {}; the charge matrix would be silently truncated (increase 'comm_modify cutoff')",
                 swb, comm->get_comm_cutoff());
  }
  FixQEqReaxFF::pre_force(vflag);
}

// The stock CG loop (fix_qeq_reaxff.cpp), kept line for line, with the stopping state reported: strict mode turns non-convergence into an error.
int FixQEqReaxFFMetal::CG(double *b, double *x)
{
  if (!strict) return FixQEqReaxFF::CG(b, x);
  double alpha, beta, b_norm, sig_old, sig_new;
  int i;
  pack_flag = 1;
  sparse_matvec(&H, x, q);
  comm->reverse_comm(this);
  vector_sum(r, 1., b, -1., q, nn);
  for (int jj = 0; jj < nn; ++jj) {
    const int j = ilist[jj];
    if (atom->mask[j] & groupbit) d[j] = r[j] * Hdia_inv[j];
  }
  b_norm = parallel_norm(b, nn);
  sig_new = parallel_dot(r, d, nn);
  for (i = 1; i < imax && std::sqrt(sig_new) / b_norm > tolerance; ++i) {
    comm->forward_comm(this);
    sparse_matvec(&H, d, q);
    comm->reverse_comm(this);
    const double tmp = parallel_dot(d, q, nn);
    alpha = sig_new / tmp;
    vector_add(x, alpha, d, nn);
    vector_add(r, -alpha, q, nn);
    for (int jj = 0; jj < nn; ++jj) {
      const int j = ilist[jj];
      if (atom->mask[j] & groupbit) p[j] = r[j] * Hdia_inv[j];
    }
    sig_old = sig_new;
    sig_new = parallel_dot(r, p, nn);
    beta = sig_new / sig_old;
    vector_sum(d, 1., p, beta, d, nn);
  }
  const double rel = std::sqrt(sig_new) / b_norm;
  if (!(rel <= tolerance))
    error->all(FLERR, "Fix qeq/reaxff/metal strict: CG did not converge after {} iterations (relative residual {:.3e} > tolerance {:.3e}) at step {}", i, rel, tolerance,
               update->ntimestep);
  return i;
}

// Equalisation residual of the charges just computed: (H q)_i + chi_i must be the same chemical potential mu for every atom.
void FixQEqReaxFFMetal::calculate_Q()
{
  FixQEqReaxFF::calculate_Q();
  if (!(verify_ev > 0.0)) return;
  pack_flag = 1;
  sparse_matvec(&H, atom->q, q);       // q (the CG scratch the reverse communication works on) = (diag(eta) + H) atom->q ; ghost q were just communicated
  comm->reverse_comm(this);
  double sum = 0.0, cnt = 0.0;
  for (int ii = 0; ii < nn; ++ii) {
    const int i = ilist[ii];
    if (atom->mask[i] & groupbit) { sum += q[i] + chi[atom->type[i]]; cnt += 1.0; }
  }
  double loc[2] = {sum, cnt}, glob[2];
  MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, world);
  const double mu = glob[1] > 0 ? glob[0] / glob[1] : 0.0;
  double worst = 0.0;
  for (int ii = 0; ii < nn; ++ii) {
    const int i = ilist[ii];
    if (atom->mask[i] & groupbit) worst = std::max(worst, std::fabs(q[i] + chi[atom->type[i]] - mu));
  }
  double gw;
  MPI_Allreduce(&worst, &gw, 1, MPI_DOUBLE, MPI_MAX, world);
  if (gw > verify_ev)
    error->all(FLERR, "Fix qeq/reaxff/metal verify: equalisation residual {:.3e} eV exceeds {:.3e} eV at step {}", gw, verify_ev, update->ntimestep);
}

void FixQEqReaxFFMetal::compute_H()
{
  pair_ = dynamic_cast<PairReaxFFMetal *>(force->pair_match("^reaxff/metal", 0));
  gpu_ = false;
  if (pair_ && pair_->uses_metal() && igroup == 0) {
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
    const double t0 = now();
    pair_->metal_context().qeq_matvec(x, static_cast<std::size_t>(atom->nlocal + atom->nghost), b);
    prof.mv += now() - t0; ++prof.nmv;
  } catch (const std::exception &e) {
    error->all(FLERR, "Fix qeq/reaxff/metal: {}", e.what());
  }
  const int nall = atom->nlocal + atom->nghost;
  for (int i = atom->nlocal; i < nall; ++i) b[i] = 0.0;   // ghost sums are folded into the owners inside the kernel
}
