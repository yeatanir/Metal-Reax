/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan
 * ReaxMetal adapter A1 -- see the header. Behaviour mirrored from pinned LAMMPS src/REAXFF/pair_reaxff.cpp (settings, coeff,
 * init_style, init_one, extract) where the result is observable by users or by the charge fixes; all deviations are deliberate and
 * documented: strict parsing (ENGINE_SPEC Q-09/Q-12), unsupported charge models rejected with an explicit error, no physics yet.
 */
#include "pair_reaxff_metal.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "reaxmetal/bonded.hpp"
#include "reaxmetal/bonded_device.hpp"
#include "reaxmetal/metal_backend.hpp"
#include "reaxmetal/nonbonded_device.hpp"
#include "reaxmetal/capabilities.hpp"
#include "reaxmetal/energy_terms.hpp"
#include "reaxmetal/nonbonded.hpp"

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

  // A2 self-check only: ask LAMMPS for the list the reference kernels use (half, newton off, with ghost rows), to compare against
  if (settings_.selfcheck) neighbor->add_request(this, NeighConst::REQ_GHOST | NeighConst::REQ_NEWTON_OFF);
}

void PairReaxFFMetal::init_list(int id, NeighList *ptr) { Pair::init_list(id, ptr); }

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

reaxmetal::NeighborCutoffs PairReaxFFMetal::cutoffs() const
{
  reaxmetal::NeighborCutoffs c;
  c.nonb = ff_->file_control().nonb_cut;
  c.bond = settings_.control.bond_cut;
  c.hbond = settings_.control.hbond_cut;
  return c;
}

reaxmetal::Box PairReaxFFMetal::host_box() const
{
  return reaxmetal::Box::from_lammps({domain->boxlo[0], domain->boxlo[1], domain->boxlo[2]}, {domain->boxhi[0], domain->boxhi[1], domain->boxhi[2]},
                                     {domain->xy, domain->xz, domain->yz}, {domain->periodicity[0] != 0, domain->periodicity[1] != 0, domain->periodicity[2] != 0});
}

reaxmetal::AtomSet PairReaxFFMetal::host_atom_set(const reaxmetal::Box &box) const
{
  using reaxmetal::SystemError;
  const int nlocal = atom->nlocal, nall = atom->nlocal + atom->nghost;
  // C3 (strict where the reference only warns): the ghost shell must be wide enough for owned-atom results to be those of the reference
  const double need = cutoffs().required_shell(), have = comm->get_comm_cutoff();
  if (have < need)
    throw SystemError("ghost shell too narrow: communication cutoff " + std::to_string(have) + " < required max(nonb_cut, hbond_cut, 2*bond_cut) = " + std::to_string(need) +
                      " (use comm_modify cutoff)");
  reaxmetal::AtomSet a;
  a.nlocal = static_cast<std::size_t>(nlocal);
  a.x.resize(3 * static_cast<std::size_t>(nall));
  a.type.resize(static_cast<std::size_t>(nall));
  a.tag.resize(static_cast<std::size_t>(nall));
  a.owner.resize(static_cast<std::size_t>(nall));
  a.shift.assign(static_cast<std::size_t>(nall), {0, 0, 0});
  std::unordered_map<tagint, int> owned_by_tag;
  owned_by_tag.reserve(static_cast<std::size_t>(nlocal) * 2);
  for (int i = 0; i < nall; ++i) {
    const auto si = static_cast<std::size_t>(i);
    for (int c = 0; c < 3; ++c) a.x[3 * si + static_cast<std::size_t>(c)] = atom->x[i][c];
    const int t = atom->type[i];
    a.type[si] = map_[static_cast<std::size_t>(t)];
    a.tag[si] = atom->tag[i];
    a.owner[si] = i;
    if (i < nlocal && !owned_by_tag.emplace(atom->tag[i], i).second) throw SystemError("duplicate atom id " + std::to_string(atom->tag[i]) + " among owned atoms");
  }
  for (int g = nlocal; g < nall; ++g) {
    const auto sg = static_cast<std::size_t>(g);
    const auto it = owned_by_tag.find(atom->tag[g]);
    if (it == owned_by_tag.end()) throw SystemError("ghost " + std::to_string(g) + " has no owned atom with id " + std::to_string(atom->tag[g]));
    a.owner[sg] = it->second;
    const reaxmetal::Vec3 fg = box.to_fractional(a.position(sg)), fo = box.to_fractional(a.position(static_cast<std::size_t>(it->second)));
    for (std::size_t d = 0; d < 3; ++d) {
      const double s = std::nearbyint(fg[d] - fo[d]);
      if (!box.periodic[d] && s != 0.0) throw SystemError("ghost " + std::to_string(g) + " is shifted along a non-periodic direction");
      a.shift[sg][d] = static_cast<std::int32_t>(s);
    }
  }
  a.validate(box, 1e-8);   // ghost == owner + lattice shift, same tag/type (the contract LAMMPS_INTEGRATION section 4 measured)
  return a;
}

std::string PairReaxFFMetal::selfcheck_summary(bool &ok)
{
  using namespace reaxmetal;
  ok = false;
  if (!list) throw SystemError("no LAMMPS neighbor list was provided to the self-check");
  const Box box = host_box();
  const AtomSet a = host_atom_set(box);
  const NeighborCutoffs cut = cutoffs();
  const FarList ours = build_far_list(a, cut);
  std::size_t lammps_entries = 0, mismatched_rows = 0, rows_seen = 0;
  std::vector<int> row;
  const int nrows = list->inum + list->gnum;
  for (int ii = 0; ii < nrows; ++ii) {
    const int i = list->ilist[ii];
    ++rows_seen;
    const double rc = cut.row_cut(static_cast<std::size_t>(i), a.nlocal), rc2 = rc * rc;
    row.clear();
    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      const int j = list->firstneigh[i][jj] & NEIGHMASK;
      const double dx = atom->x[j][0] - atom->x[i][0], dy = atom->x[j][1] - atom->x[i][1], dz = atom->x[j][2] - atom->x[i][2];
      if (dx * dx + dy * dy + dz * dz <= rc2) row.push_back(j);
    }
    std::sort(row.begin(), row.end());
    lammps_entries += row.size();
    const auto b = ours.nbr.begin() + static_cast<std::ptrdiff_t>(ours.row_start[static_cast<std::size_t>(i)]);
    const auto e = ours.nbr.begin() + static_cast<std::ptrdiff_t>(ours.row_start[static_cast<std::size_t>(i) + 1]);
    if (!std::equal(row.begin(), row.end(), b, e)) ++mismatched_rows;
  }
  const PairCounts pc = count_nonbonded_pairs(a, ours, cut);
  ok = (rows_seen == a.nall()) && mismatched_rows == 0 && lammps_entries == ours.entries();
  return "ReaxMetal A2 self-check " + std::string(ok ? "OK" : "FAILED") + ": nlocal=" + std::to_string(a.nlocal) + " nghost=" + std::to_string(a.nghost()) +
         " rows=" + std::to_string(rows_seen) + " lammps_entries=" + std::to_string(lammps_entries) + " engine_entries=" + std::to_string(ours.entries()) +
         " mismatched_rows=" + std::to_string(mismatched_rows) + " vdw_oo=" + std::to_string(pc.oo) + " vdw_og=" + std::to_string(pc.og) +
         " vdw_self=" + std::to_string(pc.self) + " comm_cutoff=" + std::to_string(comm->get_comm_cutoff());
}

void PairReaxFFMetal::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  std::string selfcheck;
  if (settings_.selfcheck) {
    // development aid (A2): verify the host view and far list against LAMMPS' own list, report through the error text and stop (the
    // C-library harness reads the summary from the message)
    bool ok = false;
    std::string failure;
    try {
      selfcheck = selfcheck_summary(ok);
    } catch (const std::exception &e) {
      failure = e.what();
    }
    if (!failure.empty()) error->all(FLERR, "ReaxMetal A2 self-check could not run: {}", failure);
    if (!ok) error->all(FLERR, "{}", selfcheck);
    error->all(FLERR, "{}; self-check mode stops here (remove reaxmetal_selfcheck to compute)", selfcheck);
  }
  if (eflag_atom || vflag_atom)
    error->all(FLERR, "Pair style reaxff/metal: per-atom energy / virial output is not implemented yet");
  const bool use_metal = settings_.backend == "metal";
  if (use_metal && !reaxmetal::mtl::compiled_with_metal())
    error->all(FLERR, "Pair style reaxff/metal: backend metal requested but this plugin was built without Metal (needs macOS and -DREAXMETAL_ENABLE_METAL=ON); use 'backend cpu64'");

  using namespace reaxmetal;
  reaxmetal::BondedResult br;
  reaxmetal::NonbondedResult nr;
  AtomSet a;
  try {
    const Box box = host_box();
    a = host_atom_set(box);
    const NeighborCutoffs cut = cutoffs();
    std::vector<double> q(a.nlocal);
    for (std::size_t i = 0; i < a.nlocal; ++i) q[i] = atom->q[i];
    BondedOptions bo;
    bo.enobonds = settings_.enobonds;
    NonbondedOptions no;
    no.lgvdw = settings_.lgvdw;
    if (use_metal) {
      // FP32 on the GPU (deterministic, no atomics); bookkeeping and the final sums in FP64 on the host. Charges come from the stock charge fix.
      if (!ctx_) ctx_ = std::make_unique<mtl::Context>();
      br = finish_bonded(ctx_->bonded(make_bonded_device_input(*ff_, settings_.control, a, box, bo)), a.nall());
      const NonbondedDeviceInput nin = make_nonbonded_device_input(*ff_, cut, a, box, q, no, [&](const DeviceListInput &l) { return ctx_->far_rows(l); });
      nr = finish_nonbonded(*ff_, a, q, ctx_->nonbonded(nin));
    } else {
      const FarList far = build_far_list(a, cut);
      br = compute_bonded_core(*ff_, settings_.control, a, far, bo);
      nr = compute_nonbonded_core(*ff_, cut, a, far, q, no);
    }
  } catch (const std::exception &e) {
    error->all(FLERR, "Pair style reaxff/metal: {}", e.what());
  }

  // forces: the engine returns gradients (dE/dx) for owned and ghost atoms; LAMMPS folds the ghost forces onto their owners
  double **f = atom->f;
  const std::size_t nall = a.nall();
  for (std::size_t i = 0; i < nall; ++i)
    for (std::size_t c = 0; c < 3; ++c) f[i][c] -= br.grad[3 * i + c] + nr.grad[3 * i + c];

  EnergyBreakdown e;
  for (std::size_t t = 0; t < kEnergyTermCount; ++t) e.e[t] = br.e.e[t] + nr.e.e[t];
  if (eflag_global) {
    const auto pv = to_lammps_pvector(e);
    for (std::size_t k = 0; k < kPvectorSize; ++k) pvector[k] = pv[k];
    eng_vdwl += e[EnergyTerm::Bond] + e[EnergyTerm::Over] + e[EnergyTerm::Under] + e[EnergyTerm::LonePair] + e[EnergyTerm::Valence] +
                e[EnergyTerm::Penalty] + e[EnergyTerm::Coalition] + e[EnergyTerm::HBond] + e[EnergyTerm::Torsion] + e[EnergyTerm::Conjugation] + e[EnergyTerm::VdW];
    eng_coul += e[EnergyTerm::Coulomb] + e[EnergyTerm::Polarization];
  }
  if (vflag_fdotr) virial_fdotr_compute();
}
