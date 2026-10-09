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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

void PairReaxFFMetal::prof(const char *name, double seconds)
{
  for (auto &e : profile_)
    if (e.first == name) { e.second += seconds; return; }
  profile_.emplace_back(name, seconds);
}

PairReaxFFMetal::~PairReaxFFMetal()
{
  if (std::getenv("REAXMETAL_PROFILE") && !profile_.empty()) {
    std::fprintf(stderr, "ReaxMetal profile (seconds, whole run):");
    for (const auto &e : profile_) std::fprintf(stderr, " %s=%.4f", e.first.c_str(), e.second);
    std::fprintf(stderr, "\n");
  }
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
  a.distributed = comm->nprocs > 1;
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
    if (a.distributed) { a.owner[sg] = -1; continue; }   // multi-rank: the owner may be on another rank
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

const NbView &PairReaxFFMetal::nb_view()
{
  using namespace reaxmetal;
  const std::size_t nall = static_cast<std::size_t>(atom->nlocal + atom->nghost);
  std::vector<double> key;
  key.reserve(3 * nall + 8);
  key.push_back(static_cast<double>(atom->nlocal));
  key.push_back(static_cast<double>(atom->nghost));
  for (int d = 0; d < 3; ++d) { key.push_back(domain->boxlo[d]); key.push_back(domain->boxhi[d]); }
  for (std::size_t i = 0; i < nall; ++i) for (int d = 0; d < 3; ++d) key.push_back(atom->x[i][d]);
  if (view_ && key == view_key_) return *view_;
  auto v = std::make_unique<NbView>();
  auto t0 = std::chrono::steady_clock::now();
  auto lap = [&](const char *name) { const auto t1 = std::chrono::steady_clock::now(); prof(name, std::chrono::duration<double>(t1 - t0).count()); t0 = t1; };
  v->box = host_box();
  v->a = host_atom_set(v->box);
  lap("view_atomset");
  v->cut = cutoffs();
  v->list = make_device_list_input(v->a, v->box, v->cut);
  lap("view_binning");
  v->rows = std::make_shared<const FarRowsF32>(metal_context().far_rows(v->list));
  lap("view_far_rows");
  view_ = std::move(v);
  view_key_ = std::move(key);
  return *view_;
}

reaxmetal::mtl::Context &PairReaxFFMetal::metal_context()
{
  if (!ctx_) ctx_ = std::make_unique<reaxmetal::mtl::Context>();
  return *ctx_;
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
  // Per-atom energy / virial need the reference's term-by-term tallies, which only the CPU-64 engine reproduces: on a step where a compute asks
  // for them, that step is evaluated by the CPU-64 engine even with 'backend metal' (energies and forces of that step are then CPU-64 ones).
  const bool per_atom = eflag_atom || vflag_atom;
  const bool use_metal = settings_.backend == "metal" && !per_atom;
  if (use_metal && !reaxmetal::mtl::compiled_with_metal())
    error->all(FLERR, "Pair style reaxff/metal: backend metal requested but this plugin was built without Metal (needs macOS and -DREAXMETAL_ENABLE_METAL=ON); use 'backend cpu64'");

  using namespace reaxmetal;
  reaxmetal::BondedResult br;
  reaxmetal::NonbondedResult nr;
  AtomSet a;
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();
  auto lap = [&](const char *name) { const auto t1 = clk::now(); prof(name, std::chrono::duration<double>(t1 - t0).count()); t0 = t1; };
  try {
    const NbView *view = use_metal ? &nb_view() : nullptr;
    const Box box = view ? view->box : host_box();
    a = view ? view->a : host_atom_set(box);
    lap("host_view");
    const NeighborCutoffs cut = cutoffs();
    std::vector<double> q(a.nall());   // per atom: ghost charges are the ones LAMMPS communicated (as the reference reads them)
    for (std::size_t i = 0; i < a.nall(); ++i) q[i] = atom->q[i];
    BondedOptions bo;
    bo.enobonds = settings_.enobonds;
    bo.per_atom = per_atom;
    NonbondedOptions no;
    no.lgvdw = settings_.lgvdw;
    no.per_atom = per_atom;
    if (use_metal) {
      // FP32 on the GPU (deterministic, no atomics); bookkeeping and the final sums in FP64 on the host. Charges come from the stock charge fix.
      metal_context();
      const BondedDeviceInput bin = make_bonded_device_input(*ff_, settings_.control, a, box, bo, &view->list);
      lap("bonded_pack");
      if (std::getenv("REAXMETAL_DEBUG_CPU_BONDED")) {   // diagnostic only
        br = compute_bonded_core(*ff_, settings_.control, a, build_far_list(a, cut), bo);
      } else {
      const BondedDeviceOutput bout = ctx_->bonded(bin);
      lap("bonded_device");
      br = finish_bonded(bout, a.nall());
      }
      if (std::getenv("REAXMETAL_DEBUG_CPU_NB")) {   // diagnostic only: nonbonded in FP64 on the host, to attribute FP32 force noise
        nr = compute_nonbonded_core(*ff_, cut, a, build_far_list(a, cut), q, no);
      } else {
      const NonbondedDeviceInput nin = make_nonbonded_device_input(*ff_, cut, a, box, q, no, [&](const DeviceListInput &) { return FarRowsF32{}; }, &view->list, view->rows);
      lap("nonbonded_pack_rows");
      const NonbondedDeviceOutput nout = ctx_->nonbonded(nin);
      lap("nonbonded_device");
      nr = finish_nonbonded(*ff_, a, q, nout);
      }
      lap("finish");
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

  if (per_atom) {
    for (std::size_t i = 0; i < nall; ++i) {
      if (eflag_atom) eatom[i] += br.eatom[i] + nr.eatom[i];
      if (vflag_atom)
        for (std::size_t c = 0; c < 6; ++c) vatom[i][c] += br.vatom[i][c] + nr.vatom[i][c];
    }
  }

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
