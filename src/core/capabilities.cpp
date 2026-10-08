// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/capabilities.hpp"

#include <array>

namespace reaxmetal {
namespace {

constexpr Status P = Status::Planned;
constexpr Status R = Status::Rejected;
constexpr Status D = Status::Deferred;
constexpr Status I = Status::Ignored;

// Keep in lock-step with docs/FEATURE_MATRIX.md (enforced by tests/test_docs_sync.cpp).
// `Implemented` rows have passing tests recorded in docs/VALIDATION.md; no energy term exists yet (M4+).
constexpr std::array kFeatures{
    // ---- energy terms -------------------------------------------------------------------------
    Feature{"term.bond", P, "M4", "ReaxFF::Bonds", "includes terminal-triple-bond stabilisation"},
    Feature{"term.lone_pair", P, "M4", "ReaxFF::Atom_Energy", "includes C2 correction"},
    Feature{"term.over_under", P, "M4", "ReaxFF::Atom_Energy", "over- and under-coordination"},
    Feature{"term.valence", P, "M6", "ReaxFF::Valence_Angles", ""},
    Feature{"term.penalty", P, "M6", "ReaxFF::Valence_Angles", ""},
    Feature{"term.coalition", P, "M6", "ReaxFF::Valence_Angles", "3-body conjugation"},
    Feature{"term.torsion", P, "M6", "ReaxFF::Torsion_Angles", ""},
    Feature{"term.conjugation", P, "M6", "ReaxFF::Torsion_Angles", "4-body conjugation"},
    Feature{"term.hbond", P, "M6", "ReaxFF::Hydrogen_Bonds", ""},
    Feature{"term.vdw.shielded", P, "M5", "vdw_type 1", "shielded Morse"},
    Feature{"term.vdw.inner_wall", P, "M5", "vdw_type 2", "Morse + inner wall, no shielding"},
    Feature{"term.vdw.shielded_inner_wall", P, "M5", "vdw_type 3", "shielded Morse + inner wall"},
    Feature{"term.vdw.lg_dispersion", P, "M5", "pair_style reaxff lgvdw yes", "low-gradient correction"},
    Feature{"term.coulomb", P, "M5", "ReaxFF::vdW_Coulomb_Energy", "taper-shielded"},
    Feature{"term.polarization", P, "M5", "ReaxFF::Compute_Polarization_Energy", "QEq self energy"},
    // ---- parameter file -------------------------------------------------------------------------
    Feature{"ffield.standard", Status::Implemented, "-", "ffield general/atom/bond/angle/torsion/hbond blocks", "39-parameter layout"},
    Feature{"ffield.atom_line5_lgvdw", Status::Implemented, "-", "ffield 5-line atom block", "only with lgvdw"},
    Feature{"ffield.offdiagonal", Status::Implemented, "-", "ffield off-diagonal block", ""},
    Feature{"ffield.torsion_compact", Status::Implemented, "-", "ffield 4-body entry 0-X-Y-0", "order-dependent overwrite, see ENGINE_SPEC 2.5"},
    Feature{"ffield.hbond_block", Status::Implemented, "-", "ffield hydrogen-bond block", "may be absent (LAMMPS warns)"},
    Feature{"ffield.control_file", Status::Implemented, "-", "pair_style reaxff <control file>", "cutoff keywords only"},
    Feature{"ffield.strict_missing_pairs", Status::Implemented, "-", "(no LAMMPS equivalent)", "reject, do not zero-fill, absent 2-body pairs (Q-12: zero-fill creates a phantom bond of BO'=1 at every r <= bond_cut)"},
    Feature{"ffield.reject_three_body_overrun", Status::Implemented, "-", "(no LAMMPS equivalent)", "reject >2 parameter sets for a j==l angle triple (Q-09: LAMMPS doubles the slot count and reads past prm[4])"},
    // ---- LAMMPS-compat flags (element knowledge expressed as data, ADR-003) -----------------------
    Feature{"compat.c2_correction", P, "M4", "strcmp(name,\"C\") in Atom_Energy", "per-type flag derived at load time"},
    Feature{"compat.triple_bond_stabilisation", P, "M4", "gp.l[37]==2 or mass pair 12.0000/15.9990", "per-pair flag"},
    Feature{"compat.light_element_split", P, "M4", "mass > 21 / mass < 21 tests", "per-type flag"},
    Feature{"compat.hbond_donor_image_exclusion", P, "M4", "orig_id[i] != orig_id[k] in Hydrogen_Bonds", "reproduce by default (Q-32): acceptor that is a periodic image of the donor is dropped; identity-based variant needs owner decision"},
    Feature{"compat.ovun_heavy_neighbor_force", P, "M6", "dDelta_lp[j] where the energy uses Delta_lp_temp[j] (Atom_Energy force loop)", "reproduce LAMMPS forces by default (Q-34: analytic force != gradient of the reported energy for heavy atoms with pi bonds); corrected variant is opt-in"},
    // ---- LAMMPS integration (M0.5) ----------------------------------------------------------------
    Feature{"lammps.pair_style_reaxff_metal", P, "M4", "pair_style reaxff/metal", "A1 exists (parse, extract, host checks; compute() refuses explicitly); planned = computes energies/forces"},
    Feature{"lammps.plugin_loadable", Status::Implemented, "-", "plugin load <reaxmetal plugin>", "DSO built against the pinned LAMMPS; version-matched"},
    Feature{"lammps.extract_chi_eta_gamma", Status::Implemented, "-", "Pair::extract(chi|eta|gamma)", "arrays indexed by LAMMPS type 1..ntypes, eta = 2x file value"},
    Feature{"lammps.single_rank_only", Status::Implemented, "-", "comm->nprocs == 1", "multi-rank runs fail explicitly (checked in init_style)"},
    Feature{"lammps.multi_rank", D, "-", "mpirun -np N>1 with reaxff/metal", "deferred; needs distributed ghost/QEq handling"},
    Feature{"lammps.newton_off", R, "-", "newton off (newton_pair off)", "forces on ghosts must be reverse-communicated"},
    Feature{"lammps.ghost_native_contract", P, "M3", "ghost atoms from LAMMPS borders", "engine consumes owned+ghost atom set (LAMMPS_INTEGRATION 4)"},
    Feature{"lammps.virial_fdotr", P, "M6", "Pair::virial_fdotr_compute", "global virial/pressure from forces on owned+ghost atoms"},
    // ---- EEM charge model (naming: EEM == the standard ReaxFF charge model; not a different physics) ---
    Feature{"eem.external_cpu_fix", P, "M4", "fix qeq/reaxff | fix qeq/shielded (stock CPU)", "extract() half verified in M2 (EEM-3); the stock fix drives q once compute() exists"},
    Feature{"eem.charge_verification", P, "M5", "(no LAMMPS equivalent)", "adapter checks the EEM residual so strictness holds with the stock fix"},
    Feature{"eem.taper_within_ghost_shell", P, "M5", "(no LAMMPS equivalent)", "error if the QEq taper radius exceeds the ghost shell: stock fix silently truncates (Q-35)"},
    Feature{"eem.strict_convergence", P, "M5", "(no LAMMPS equivalent)", "default: non-convergence is an error/status, never silently accepted"},
    Feature{"eem.compat_warn_continue", P, "M5", "fix qeq/reaxff default (warn and continue)", "explicit opt-in only"},
    Feature{"eem.gpu_resident", P, "M5", "(no LAMMPS equivalent)", "GPU-resident EEM solve, subject to numerical validation"},
    Feature{"eem.net_charge_nonzero", D, "-", "non-neutral fix group in fix qeq/reaxff", "LAMMPS imposes sum(q)=0; non-zero total charge not supported"},
    Feature{"compat.flag_derivation", Status::Implemented, "-", "compat predicates (ENGINE_SPEC Q-06)", "exact upstream predicates, boundary-tested (tests/test_compat_flags.cpp)"},
    // ---- pair_style options ----------------------------------------------------------------------
    Feature{"opt.enobonds", P, "M4", "pair_style reaxff enobonds yes|no", ""},
    Feature{"opt.checkqeq_no", P, "M5", "pair_style reaxff checkqeq no", "fixed input charges"},
    Feature{"opt.lgvdw", P, "M5", "pair_style reaxff lgvdw yes", ""},
    Feature{"opt.memory_heuristics", I, "-", "safezone / mincap / minhbonds", "LAMMPS allocation heuristics only"},
    Feature{"opt.list_blocking", I, "-", "list/blocking", "Kokkos performance option only"},
    Feature{"opt.tabulate", D, "-", "tabulate N>0 / tabulate_long_range N>0", "deferred; spline tables change the numbers, analytic evaluation only for now"},
    // ---- charge models ---------------------------------------------------------------------------
    Feature{"qeq.reaxff", P, "M5", "fix qeq/reaxff ... reaxff", "standard EEM as implemented by fix qeq/reaxff (ENGINE_SPEC 7)"},
    Feature{"qeq.pertype_file", P, "M5", "fix qeq/reaxff ... <param file>", "per-type chi/eta/gamma override"},
    Feature{"qeq.shielded", P, "M5", "fix qeq/shielded", "LAMMPS-compatible shielded charge equilibration; same kernel as qeq/reaxff (ENGINE_SPEC 7.2); works through extract() today"},
    Feature{"qeq.acks2", D, "-", "fix acks2/reaxff", "deferred; different charge model"},
    Feature{"qeq.qtpie", D, "-", "fix qtpie/reaxff", "deferred; different charge model"},
    Feature{"qeq.relative", D, "-", "fix qeq/rel/reaxff", "deferred"},
    Feature{"qeq.efield", D, "-", "fix efield with fix qeq/reaxff", "deferred; external electric field"},
    Feature{"qeq.group_subset", D, "-", "fix qeq/reaxff on a proper subgroup", "deferred; all atoms are equilibrated"},
    // ---- system description ----------------------------------------------------------------------
    Feature{"sys.pbc_images", P, "M3", "ghost atoms / periodic images", "ghost-native contract: images supplied by the host (LAMMPS ghosts) or by the standalone image expander; never minimum-image only (ADR-004/013)"},
    Feature{"sys.triclinic", P, "M3", "triclinic box", ""},
    Feature{"sys.nonperiodic", P, "M3", "boundary f/s/m", ""},
    Feature{"sys.type_null_mapping", R, "-", "pair_coeff ... NULL", "hybrid placeholder; not planned"},
    Feature{"sys.hybrid", R, "-", "pair_style hybrid[/overlay] with reaxff", "not planned"},
    // ---- outputs ---------------------------------------------------------------------------------
    Feature{"out.energy_breakdown", P, "M4", "compute pair reaxff (pvector[14])", "see energy_terms.hpp"},
    Feature{"out.forces", P, "M6", "atom->f", "analytical"},
    Feature{"out.charges", P, "M5", "atom->q", ""},
    Feature{"out.virial", P, "M6", "virial_fdotr / v_tally*", "needed for pressure"},
    Feature{"out.per_atom_energy", P, "M7", "compute pe/atom with reaxff", "adapter-level per-atom energy/virial; until then requests are refused"},
    Feature{"out.bond_analysis", D, "-", "fix reaxff/bonds, fix reaxff/species", "deferred; LAMMPS analysis tools dynamic_cast to PairReaxFF and refuse other styles"},
    // ---- dynamics --------------------------------------------------------------------------------
    Feature{"md.nve", R, "-", "fix nve", "standalone MD is out of scope: LAMMPS provides integrators"},
    Feature{"md.thermostat", R, "-", "fix nvt / langevin", "standalone MD is out of scope: LAMMPS provides thermostats"},
    Feature{"md.barostat", P, "M6", "fix npt", "enabled once out.virial is validated; until then pressure-controlled runs are refused"},
    Feature{"min.minimize", R, "-", "minimize", "standalone minimiser is out of scope: LAMMPS provides minimize"},
    // ---- backends --------------------------------------------------------------------------------
    Feature{"backend.cpu_fp64", P, "M4", "-", "reference backend"},
    Feature{"backend.cpu_fp32_twin", P, "M4", "-", "same kernels in float; calibrates GPU tolerance"},
    Feature{"backend.metal_fp32", P, "M3", "-", "native Metal; Apple GPUs have no FP64"},
    Feature{"backend.metal_atomic_accum", P, "M8", "-", "benchmark-only option; default path is deterministic"},
    Feature{"backend.metal_batching", P, "M8", "-", ""},
    Feature{"backend.fp16", R, "-", "-", "not planned"},
};

}  // namespace

std::span<const Feature> feature_table() noexcept { return kFeatures; }

const Feature* find_feature(std::string_view id) noexcept {
  for (const Feature& f : kFeatures)
    if (f.id == id) return &f;
  return nullptr;
}

const Feature* find_by_lammps_construct(std::string_view construct) noexcept {
  // "-" (or empty) means "no LAMMPS spelling" and must never be matchable.
  if (construct.empty() || construct == "-") return nullptr;
  for (const Feature& f : kFeatures)
    if (f.lammps_construct == construct) return &f;
  return nullptr;
}

std::string_view to_string(Status s) noexcept {
  switch (s) {
    case Status::Implemented: return "Implemented";
    case Status::Planned: return "Planned";
    case Status::Deferred: return "Deferred";
    case Status::Rejected: return "Rejected";
    case Status::Ignored: return "Ignored";
  }
  return "?";
}

Verdict require_supported(std::string_view id) {
  const Feature* f = find_feature(id);
  if (f == nullptr) throw std::logic_error("require_supported: unknown feature id '" + std::string(id) + "'");
  switch (f->status) {
    case Status::Implemented: return Verdict::Supported;
    case Status::Ignored: return Verdict::IgnoredWithNotice;
    case Status::Planned:
      throw NotImplementedError("feature '" + std::string(id) + "' is planned for milestone " +
                                std::string(f->milestone) + " and is not implemented yet");
    case Status::Deferred:
    case Status::Rejected:
      throw UnsupportedFeatureError("feature '" + std::string(id) + "' (" + std::string(f->lammps_construct) +
                                    ") is not supported: " + std::string(f->note));
  }
  throw std::logic_error("unreachable");
}

}  // namespace reaxmetal
