/* SPDX-License-Identifier: GPL-2.0-only
 * SPIKE / SCAFFOLDING (M0.5) -- see header. Contains no ReaxFF physics: all energies and forces are zero
 * (except the optional `ghostforce yes` probe, which adds a unit x-force to EVERY atom incl. ghosts so the
 * reverse-communication fold-back can be measured). The ffield parse below extracts only chi/eta/gamma and is
 * throw-away test scaffolding, NOT the engine's parser (milestone M2).
 */
#include "pair_reaxff_metal_probe.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <map>

using namespace LAMMPS_NS;

PairReaxFFMetalProbe::PairReaxFFMetalProbe(LAMMPS *lmp) :
    Pair(lmp), neigh_mode(NONE), cutmax(10.0), ghostforce(false), verbose(false), last_report(-1)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;           // as pair reaxff
  centroidstressflag = CENTROID_NOTAVAIL;
  ghostneigh = 0;
  nextra = 14;                    // as pair reaxff: compute pair exposes pvector[0..13]
  pvector = new double[nextra];
  for (int i = 0; i < nextra; ++i) pvector[i] = 0.0;
  no_virial_fdotr_compute = 0;    // virial from sum x.f over owned+ghost atoms (exact for any many-body potential)
}

PairReaxFFMetalProbe::~PairReaxFFMetalProbe()
{
  delete[] pvector;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutghost);
  }
}

void PairReaxFFMetalProbe::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cutghost, n + 1, n + 1, "pair:cutghost");
  map.assign(n + 1, -1);
  chi_.assign(n + 1, 0.0);
  eta_.assign(n + 1, 0.0);
  gamma_.assign(n + 1, 0.0);
}

// pair_style reaxff/metal NULL [cutoff R] [neigh none|half|halfoff|full|fullghost|halfoffghost] [ghostforce yes|no] [verbose yes|no]
void PairReaxFFMetalProbe::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR, "Illegal pair_style command");
  if (strcmp(arg[0], "NULL") != 0) error->all(FLERR, "reaxff/metal probe: control file not supported, use NULL");
  int iarg = 1;
  while (iarg < narg) {
    if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_style reaxff/metal command");
    std::string key = arg[iarg], val = arg[iarg + 1];
    if (key == "cutoff") cutmax = utils::numeric(FLERR, val, false, lmp);
    else if (key == "ghostforce") ghostforce = utils::logical(FLERR, val, false, lmp);
    else if (key == "verbose") verbose = utils::logical(FLERR, val, false, lmp);
    else if (key == "neigh") {
      if (val == "none") neigh_mode = NONE;
      else if (val == "half") neigh_mode = HALF;
      else if (val == "halfoff") neigh_mode = HALF_NEWTON_OFF;
      else if (val == "full") neigh_mode = FULL;
      else if (val == "fullghost") neigh_mode = FULL_GHOST;
      else if (val == "halfoffghost") neigh_mode = HALF_NEWTON_OFF_GHOST;
      else error->all(FLERR, "Illegal neigh keyword {} for pair_style reaxff/metal", val);
    } else error->all(FLERR, "Illegal pair_style reaxff/metal keyword {}", key);
    iarg += 2;
  }
  ghostneigh = (neigh_mode == FULL_GHOST || neigh_mode == HALF_NEWTON_OFF_GHOST) ? 1 : 0;
}

// pair_coeff * * ffield El1 El2 ...   (scaffolding: reads only chi, eta(x2), gamma of 4-line atom blocks)
void PairReaxFFMetalProbe::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  const int n = atom->ntypes;
  if (narg != 3 + n) error->all(FLERR, "Incorrect args for pair coefficients");
  FILE *fp = utils::open_potential(arg[2], lmp, nullptr);
  if (!fp) error->all(FLERR, "Cannot open ffield file {}", arg[2]);
  char line[1024];
  auto next = [&]() -> bool { return fgets(line, sizeof line, fp) != nullptr; };
  auto tok = [&](int idx) -> std::string {
    std::vector<std::string> t = utils::split_words(line);
    return idx < (int) t.size() ? t[idx] : std::string();
  };
  next();                                         // comment line
  next();
  int nglobal = atoi(line);
  for (int i = 0; i < nglobal; ++i) next();
  next();
  int nat = atoi(line);
  next(); next(); next();
  struct El { std::string name; double chi, eta, gamma; };
  std::vector<El> els;
  for (int i = 0; i < nat; ++i) {
    El e;
    next(); e.name = utils::uppercase(tok(0)); e.gamma = utils::numeric(FLERR, tok(6), false, lmp);
    next(); e.chi = utils::numeric(FLERR, tok(5), false, lmp); e.eta = 2.0 * utils::numeric(FLERR, tok(6), false, lmp);
    next(); next();
    els.push_back(e);
  }
  fclose(fp);
  int count = 0;
  for (int t = 1; t <= n; ++t) {
    std::string want = utils::uppercase(arg[2 + t]);
    map[t] = -1;
    for (int k = 0; k < (int) els.size(); ++k)
      if (els[k].name == want) { map[t] = k; chi_[t] = els[k].chi; eta_[t] = els[k].eta; gamma_[t] = els[k].gamma; ++count; }
  }
  if (count != n) error->all(FLERR, "Non-existent ReaxFF type");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 1;
}

void PairReaxFFMetalProbe::init_style()
{
  // host-contract checks the production adapter must make (ENGINE decisions: single rank, newton on)
  if (!atom->q_flag) error->all(FLERR, "Pair style reaxff/metal requires atom attribute q");
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style reaxff/metal requires atom IDs");
  if (force->newton_pair == 0) error->all(FLERR, "Pair style reaxff/metal requires newton pair on");
  if (comm->nprocs != 1) error->all(FLERR, "Pair style reaxff/metal supports a single MPI rank only");
  switch (neigh_mode) {
    case NONE: break;    // deliberately no neighbor request: tests whether LAMMPS permits it
    case HALF: neighbor->add_request(this); break;
    case HALF_NEWTON_OFF: neighbor->add_request(this, NeighConst::REQ_NEWTON_OFF); break;
    case FULL: neighbor->add_request(this, NeighConst::REQ_FULL); break;
    case FULL_GHOST: neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST); break;
    case HALF_NEWTON_OFF_GHOST: neighbor->add_request(this, NeighConst::REQ_GHOST | NeighConst::REQ_NEWTON_OFF); break;
  }
}

void PairReaxFFMetalProbe::init_list(int id, NeighList *ptr) { Pair::init_list(id, ptr); }

double PairReaxFFMetalProbe::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  cutghost[i][j] = cutghost[j][i] = cutmax;
  return cutmax;
}

void *PairReaxFFMetalProbe::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "chi") == 0) return (void *) chi_.data();
  if (strcmp(str, "eta") == 0) return (void *) eta_.data();
  if (strcmp(str, "gamma") == 0) return (void *) gamma_.data();
  return nullptr;
}

void PairReaxFFMetalProbe::report(int eflag, int vflag)
{
  const int nlocal = atom->nlocal, nghost = atom->nghost, nall = nlocal + nghost;
  double **x = atom->x;
  tagint *tag = atom->tag;
  std::map<tagint, int> mult;
  for (int i = 0; i < nall; ++i) ++mult[tag[i]];
  int maxmult = 0;
  for (auto &kv : mult) maxmult = std::max(maxmult, kv.second);

  // ghost = owner + integer lattice shift ?  (orthogonal boxes only)
  int bad = 0, maximg[3] = {0, 0, 0};
  const double prd[3] = {domain->xprd, domain->yprd, domain->zprd};
  for (int g = nlocal; g < nall; ++g) {
    int o = atom->map(tag[g]);
    if (o < 0 || o >= nlocal) { ++bad; continue; }
    for (int d = 0; d < 3; ++d) {
      double s = (x[g][d] - x[o][d]) / prd[d];
      double r = std::round(s);
      if (std::fabs(s - r) > 1e-8) ++bad;
      maximg[d] = std::max(maximg[d], (int) std::fabs(r));
    }
  }
  unsigned long long order_hash = 1469598103934665603ULL;     // FNV-1a over the owned-atom tag ORDER
  for (int i = 0; i < nlocal; ++i) { order_hash ^= (unsigned long long) tag[i]; order_hash *= 1099511628211ULL; }
  long pairs_in = 0, pairs_all = 0;
  int inum = -1, gnum = -1;
  if (list) {
    inum = list->inum; gnum = list->gnum;
    for (int ii = 0; ii < inum + gnum; ++ii) {
      int i = list->ilist[ii];
      for (int jj = 0; jj < list->numneigh[i]; ++jj) {
        int j = list->firstneigh[i][jj] & NEIGHMASK;
        double dx = x[j][0] - x[i][0], dy = x[j][1] - x[i][1], dz = x[j][2] - x[i][2];
        ++pairs_all;
        if (dx * dx + dy * dy + dz * dz <= cutmax * cutmax) ++pairs_in;
      }
    }
  }
  utils::logmesg(lmp,
      "PROBE step={} ago={} eflag={} vflag={} nlocal={} nghost={} maxmult={} ghost_not_lattice_shift={} "
      "max_image=[{},{},{}] nprocs={} newton_pair={} neigh_mode={} list={} inum={} gnum={} pairs_listed={} "
      "pairs_within_cut={} cutforce={:.6f} skin={:.6f} cutneighmax={:.6f} comm_cutoff_user={:.6f} "
      "mult_tag1={} order_hash={:x} nbor_ago_changed_state={}\n",
      update->ntimestep, neighbor->ago, eflag, vflag, nlocal, nghost, maxmult, bad, maximg[0], maximg[1], maximg[2],
      comm->nprocs, force->newton_pair, (int) neigh_mode, list ? 1 : 0, inum, gnum, pairs_all, pairs_in, cutforce,
      neighbor->skin, neighbor->cutneighmax, comm->cutghostuser, mult.count(1) ? mult[1] : 0, order_hash, (int) (neighbor->ago == 0));
}

void PairReaxFFMetalProbe::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (verbose && last_report != update->ntimestep) {
    report(eflag, vflag);
    last_report = update->ntimestep;
  }
  if (eflag_global) for (int i = 0; i < nextra; ++i) pvector[i] = i + 1.0;   // marker values
  // ghost-force fold-back probe: +1 on x for every atom INCLUDING ghosts (LAMMPS must reverse-communicate)
  if (ghostforce) {
    double **f = atom->f;
    const int nall = atom->nlocal + atom->nghost;
    for (int i = 0; i < nall; ++i) f[i][0] += 1.0;
  }
  if (vflag_fdotr) virial_fdotr_compute();
}
