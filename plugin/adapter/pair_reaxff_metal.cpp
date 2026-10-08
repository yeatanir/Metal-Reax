/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan
 * ReaxMetal adapter A1 -- see the header. Behaviour mirrored from pinned LAMMPS src/REAXFF/pair_reaxff.cpp (settings, coeff,
 * init_style, init_one, extract) where the result is observable by users or by the charge fixes; all deviations are deliberate and
 * documented: strict parsing (ENGINE_SPEC Q-09/Q-12), unsupported charge models rejected with an explicit error, no physics yet.
 */
#include "pair_reaxff_metal.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "utils.h"

#include <algorithm>
#include <cstring>

#include "reaxmetal/capabilities.hpp"

using namespace LAMMPS_NS;

PairReaxFFMetal::PairReaxFFMetal(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  ghostneigh = 0;
  centroidstressflag = CENTROID_NOTAVAIL;
  no_virial_fdotr_compute = 0;
  nextra = 14;                       // compute pair reaxff/metal exposes the 14 energy slots like pair reaxff
  pvector = new double[nextra];
  for (int i = 0; i < nextra; ++i) pvector[i] = 0.0;
}

PairReaxFFMetal::~PairReaxFFMetal()
{
  delete[] pvector;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutghost);
  }
}

void PairReaxFFMetal::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cutghost, n + 1, n + 1, "pair:cutghost");
  map_.assign(static_cast<std::size_t>(n) + 1, -1);
  chi_.assign(static_cast<std::size_t>(n) + 1, 0.0);
  eta_.assign(static_cast<std::size_t>(n) + 1, 0.0);
  gamma_.assign(static_cast<std::size_t>(n) + 1, 0.0);
}

void PairReaxFFMetal::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR, "Illegal pair_style command");
  std::vector<std::string> args(arg, arg + narg);
  try {
    settings_ = reaxmetal::parse_pair_style_args(args);
  } catch (const std::exception &e) {
    error->all(FLERR, "{}", e.what());
  }
  if (comm->me == 0)
    for (const auto &n : settings_.notices) error->warning(FLERR, "{}", n);
}

void PairReaxFFMetal::coeff(int nargs, char **args)
{
  if (!allocated) allocate();
  const int n = atom->ntypes;
  if (nargs != 3 + n) error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));

  // read and validate the force field with reaxmetal's own parser (every rank: the file is small and read-only)
  try {
    reaxmetal::FfieldOptions opt;
    opt.lgvdw = settings_.lgvdw;
    ff_ = std::make_unique<reaxmetal::ForceField>(reaxmetal::read_force_field_file(utils::get_potential_file_path(args[2]), opt));
  } catch (const std::exception &e) {
    error->all(FLERR, "{}", e.what());
  }
  if (comm->me == 0)
    for (const auto &w : ff_->warnings()) error->warning(FLERR, "{}", w);

  // map LAMMPS atom types to force-field elements: "NULL" or a case-insensitive match of the (<=3 char, upper-cased) element
  // symbol; like pair reaxff, every match is counted, so a duplicated symbol in the file fails the count check below.
  int itmp = 0;
  std::fill(map_.begin(), map_.end(), -1);
  for (int i = 3; i < nargs; ++i) {
    if (strcmp(args[i], "NULL") == 0) {
      map_[static_cast<std::size_t>(i - 2)] = -1;
      ++itmp;
      continue;
    }
    for (int k : ff_->match_element(args[i])) {
      map_[static_cast<std::size_t>(i - 2)] = k;
      ++itmp;
    }
  }
  if (itmp != n) error->all(FLERR, "Non-existent ReaxFF type");

  // Q-12: pinned LAMMPS zero-fills absent bond-pair blocks and creates phantom bonds; we refuse to use such a pair.
  std::vector<int> used;
  for (int t = 1; t <= n; ++t)
    if (map_[static_cast<std::size_t>(t)] >= 0 && std::find(used.begin(), used.end(), map_[static_cast<std::size_t>(t)]) == used.end())
      used.push_back(map_[static_cast<std::size_t>(t)]);
  try {
    ff_->require_bond_blocks(used);
  } catch (const std::exception &e) {
    error->all(FLERR, "{}", e.what());
  }

  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j) setflag[i][j] = 0;
  int count = 0;
  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j)
      if (map_[static_cast<std::size_t>(i)] >= 0 && map_[static_cast<std::size_t>(j)] >= 0) {
        setflag[i][j] = 1;
        ++count;
      }
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));

  cutmax_ = std::max({ff_->file_control().nonb_cut, settings_.control.hbond_cut, settings_.control.bond_cut});
}

void PairReaxFFMetal::init_style()
{
  // host contract (LAMMPS_INTEGRATION C1-C8); the messages for the first checks match pair reaxff
  if (!atom->q_flag) error->all(FLERR, "Pair style reaxff requires atom attribute q");
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style reaxff requires atom IDs");
  if (force->newton_pair == 0) error->all(FLERR, "Pair style reaxff requires newton pair on");
  if (comm->nprocs != 1) error->all(FLERR, "Pair style reaxff/metal supports a single MPI rank only");

  // charge fix: standard EEM only (fix qeq/reaxff or fix qeq/shielded); the other ReaxFF charge models are Deferred
  const std::size_t acks2 = modify->get_fix_by_style("^acks2/reax").size();
  const std::size_t qtpie = modify->get_fix_by_style("^qtpie/reax").size();
  const std::size_t qeqrel = modify->get_fix_by_style("^qeq/rel/reax").size();
  if (acks2 + qtpie + qeqrel > 0)
    error->all(FLERR, "Pair style reaxff/metal does not support fix acks2/reaxff, qtpie/reaxff or qeq/rel/reaxff (deferred; see docs/FEATURE_MATRIX.md)");
  const std::size_t have_qeq = modify->get_fix_by_style("^qeq/reax").size() + modify->get_fix_by_style("^qeq/shielded").size();
  if (settings_.checkqeq && have_qeq != 1)
    error->all(FLERR, "Pair style reaxff/metal requires use of exactly one of the fix qeq/reaxff or fix qeq/shielded commands");

  if (cutmax_ < 2.0 * settings_.control.bond_cut && comm->me == 0)
    error->warning(FLERR, "Total cutoff < 2*bond cutoff. May need to use an increased neighbor list skin.");
}

double PairReaxFFMetal::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  cutghost[i][j] = cutghost[j][i] = cutmax_;
  return cutmax_;
}

void *PairReaxFFMetal::extract(const char *str, int &dim)
{
  dim = 1;
  if (!ff_) return nullptr;
  auto fill = [&](std::vector<double> &v, auto member) -> void * {
    v[0] = 0.0;
    for (int t = 1; t <= atom->ntypes; ++t) v[static_cast<std::size_t>(t)] = map_[static_cast<std::size_t>(t)] >= 0 ? ff_->single(map_[static_cast<std::size_t>(t)]).*member : 0.0;
    return static_cast<void *>(v.data());
  };
  if (strcmp(str, "chi") == 0) return fill(chi_, &reaxmetal::SingleBody::chi);
  if (strcmp(str, "eta") == 0) return fill(eta_, &reaxmetal::SingleBody::eta);       // 2 x file value, as pair reaxff
  if (strcmp(str, "gamma") == 0) return fill(gamma_, &reaxmetal::SingleBody::gamma);
  return nullptr;
}

void PairReaxFFMetal::compute(int, int)
{
  error->all(FLERR,
             "Pair style reaxff/metal: the force backend is not implemented yet (milestone M4). This adapter (A1, milestone M2) "
             "provides parsing, pair_coeff mapping, extract() and host checks only; it never returns zero energies or forces.");
}
