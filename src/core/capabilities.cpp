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
    Feature{"term.bond", Status::Implemented, "-", "ReaxFF::Bonds", "includes terminal-triple-bond stabilisation"},
    Feature{"term.lone_pair", Status::Implemented, "-", "ReaxFF::Atom_Energy", "includes C2 correction"},
    Feature{"term.over_under", Status::Implemented, "-", "ReaxFF::Atom_Energy", "over- and under-coordination"},
    Feature{"term.valence", Status::Implemented, "-", "ReaxFF::Valence_Angles", ""},
    Feature{"term.penalty", Status::Implemented, "-", "ReaxFF::Valence_Angles", ""},
    Feature{"term.coalition", Status::Implemented, "-", "ReaxFF::Valence_Angles", "3-body conjugation"},
    Feature{"term.torsion", Status::Implemented, "-", "ReaxFF::Torsion_Angles", ""},
    Feature{"term.conjugation", Status::Implemented, "-", "ReaxFF::Torsion_Angles", "4-body conjugation"},
    Feature{"term.hbond", Status::Implemented, "-", "ReaxFF::Hydrogen_Bonds", ""},
    Feature{"term.vdw.shielded", Status::Implemented, "-", "vdw_type 1", "shielded Morse"},
    Feature{"term.vdw.inner_wall", P, "M5", "vdw_type 2", "Morse + inner wall, no shielding"},
    Feature{"term.vdw.shielded_inner_wall", Status::Implemented, "-", "vdw_type 3", "shielded Morse + inner wall"},
    Feature{"term.vdw.lg_dispersion", Status::Implemented, "-", "pair_style reaxff lgvdw yes", "low-gradient correction"},
    Feature{"term.coulomb", Status::Implemented, "-", "ReaxFF::vdW_Coulomb_Energy", "taper-shielded"},
    Feature{"term.polarization", Status::Implemented, "-", "ReaxFF::Compute_Polarization_Energy", "QEq self energy"},
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
    Feature{"compat.c2_correction", Status::Implemented, "-", "strcmp(name,\"C\") in Atom_Energy", "per-type flag derived at load time"},
    Feature{"compat.triple_bond_stabilisation", Status::Implemented, "-", "gp.l[37]==2 or mass pair 12.0000/15.9990", "per-pair flag"},
    Feature{"compat.light_element_split", Status::Implemented, "-", "mass > 21 / mass < 21 tests", "per-type flag"},
    Feature{"compat.hbond_donor_image_exclusion", Status::Implemented, "-", "orig_id[i] != orig_id[k] in Hydrogen_Bonds", "reproduce by default (Q-32): acceptor that is a periodic image of the donor is dropped; identity-based variant needs owner decision"},
    Feature{"compat.ovun_heavy_neighbor_force", Status::Implemented, "-", "dDelta_lp[j] where the energy uses Delta_lp_temp[j] (Atom_Energy force loop)", "reproduce LAMMPS forces by default (Q-34: analytic force != gradient of the reported energy for heavy atoms with pi bonds); corrected variant is opt-in"},
    // ---- LAMMPS integration (M0.5) ----------------------------------------------------------------
    Feature{"lammps.pair_style_reaxff_metal", Status::Implemented, "-", "pair_style reaxff/metal", "computes energies, forces and the virial (backend cpu64 | metal); INT-2 vs stock pair reaxff on 58 fixtures; per-atom energy/virial refused"},
    Feature{"lammps.plugin_loadable", Status::Implemented, "-", "plugin load <reaxmetal plugin>", "DSO built against the pinned LAMMPS; version-matched"},
    Feature{"lammps.extract_chi_eta_gamma", Status::Implemented, "-", "Pair::extract(chi|eta|gamma)", "arrays indexed by LAMMPS type 1..ntypes, eta = 2x file value"},
    Feature{"lammps.single_rank_only", Status::Implemented, "-", "comm->nprocs == 1", "multi-rank runs fail explicitly (checked in init_style)"},
    Feature{"lammps.multi_rank", D, "-", "mpirun -np N>1 with reaxff/metal", "deferred; needs distributed ghost/QEq handling"},
    Feature{"lammps.newton_off", R, "-", "newton off (newton_pair off)", "forces on ghosts must be reverse-communicated"},
    Feature{"lammps.ghost_native_contract", Status::Implemented, "-", "ghost atoms from LAMMPS borders", "adapter A2 builds the owned+ghost view from LAMMPS arrays and verifies ghost = owner + shift (INT-7, 58 fixtures); far list equals LAMMPS' own list row by row"},
    Feature{"lammps.virial_fdotr", P, "M6", "Pair::virial_fdotr_compute", "global virial/pressure from forces on owned+ghost atoms; pressure equals stock pair reaxff in INT-2 (cpu64 1e-10, metal 2.5e-4 relative); INT-4 (NPT) not run"},
    Feature{"lammps.ghost_shell_check", Status::Implemented, "-", "(reference only warns, pair_reaxff.cpp:372-375)", "ghost shell < max(nonb_cut, hbond_cut, 2*bond_cut) is an error (INT-7 negative case)"},
    Feature{"dev.neighbor_selfcheck", Status::Implemented, "-", "(development aid)", "pair_style keyword reaxmetal_selfcheck yes: verify the host view and far list inside LAMMPS, then stop with the summary (development aid)"},
    // ---- EEM charge model (naming: EEM == the standard ReaxFF charge model; not a different physics) ---
    Feature{"eem.external_cpu_fix", Status::Implemented, "-", "fix qeq/reaxff | fix qeq/shielded (stock CPU)", "the stock fix qeq/reaxff drives q through extract() and compute(); charges identical to stock in INT-2"},
    Feature{"eem.charge_verification", P, "M5", "(no LAMMPS equivalent)", "adapter checks the EEM residual so strictness holds with the stock fix"},
    Feature{"eem.taper_within_ghost_shell", P, "M5", "(no LAMMPS equivalent)", "error if the QEq taper radius exceeds the ghost shell: stock fix silently truncates (Q-35)"},
    Feature{"eem.strict_convergence", P, "M5", "(no LAMMPS equivalent)", "default: non-convergence is an error/status, never silently accepted"},
    Feature{"eem.compat_warn_continue", P, "M5", "fix qeq/reaxff default (warn and continue)", "explicit opt-in only"},
    Feature{"eem.gpu_resident", P, "M5", "(no LAMMPS equivalent)", "GPU-resident EEM solve, subject to numerical validation"},
    Feature{"eem.net_charge_nonzero", D, "-", "non-neutral fix group in fix qeq/reaxff", "LAMMPS imposes sum(q)=0; non-zero total charge not supported"},
    Feature{"compat.flag_derivation", Status::Implemented, "-", "compat predicates (ENGINE_SPEC Q-06)", "exact upstream predicates, boundary-tested (tests/test_compat_flags.cpp)"},
    // ---- pair_style options ----------------------------------------------------------------------
    Feature{"opt.enobonds", P, "M4", "pair_style reaxff enobonds yes|no", ""},
    Feature{"opt.checkqeq_no", Status::Implemented, "-", "pair_style reaxff checkqeq no", "fixed input charges"},
    Feature{"opt.lgvdw", Status::Implemented, "-", "pair_style reaxff lgvdw yes", ""},
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
    Feature{"sys.pbc_images", Status::Implemented, "-", "ghost atoms / periodic images", "CPU: standalone image expander (triclinic, mixed periodicity) = LAMMPS ghost set for all 58 fixtures (NBR-2); LAMMPS ghosts verified as owner + lattice shift inside LAMMPS (INT-7). Metal: written only"},
    Feature{"sys.triclinic", Status::Implemented, "-", "triclinic box", "CPU: expander, far list and ghost-native view on triclinic cells (NBR-1/2/3, INT-7)"},
    Feature{"sys.nonperiodic", Status::Implemented, "-", "boundary f/s/m", "CPU: boundary f along any direction (NBR-1, fixtures); shrink-wrapped/m boundaries are not separately tested"},
    Feature{"sys.type_null_mapping", R, "-", "pair_coeff ... NULL", "hybrid placeholder; not planned"},
    Feature{"sys.hybrid", R, "-", "pair_style hybrid[/overlay] with reaxff", "not planned"},
    // ---- outputs ---------------------------------------------------------------------------------
    Feature{"out.energy_breakdown", Status::Implemented, "-", "compute pair reaxff (pvector[14])", "see energy_terms.hpp"},
    Feature{"out.forces", Status::Implemented, "-", "atom->f", "analytical"},
    Feature{"out.charges", P, "M5", "atom->q", ""},
    Feature{"out.virial", P, "M6", "virial_fdotr / v_tally*", "needed for pressure; computed by virial_fdotr in the adapter; matches stock in INT-2; stays Planned until INT-4 (NPT stability)"},
    Feature{"out.per_atom_energy", P, "M7", "compute pe/atom with reaxff", "adapter-level per-atom energy/virial; until then requests are refused"},
    Feature{"out.bond_analysis", D, "-", "fix reaxff/bonds, fix reaxff/species", "deferred; LAMMPS analysis tools dynamic_cast to PairReaxFF and refuse other styles"},
    // ---- dynamics --------------------------------------------------------------------------------
    Feature{"md.nve", R, "-", "fix nve", "standalone MD is out of scope: LAMMPS provides integrators"},
    Feature{"md.thermostat", R, "-", "fix nvt / langevin", "standalone MD is out of scope: LAMMPS provides thermostats"},
    Feature{"md.barostat", P, "M6", "fix npt", "enabled once out.virial is validated; until then pressure-controlled runs are refused"},
    Feature{"min.minimize", R, "-", "minimize", "standalone minimiser is out of scope: LAMMPS provides minimize"},
    // ---- backends --------------------------------------------------------------------------------
    Feature{"backend.cpu_fp64", Status::Implemented, "-", "-", "reference backend: all 13 energy terms and forces equal pinned LAMMPS (FULL-1, INT-2)"},
    Feature{"backend.cpu_fp32_twin", P, "M4", "-", "same kernels in float; calibrates GPU tolerance"},
    Feature{"backend.metal_fp32", Status::Implemented, "-", "-", "native Metal FP32 for neighbor rows, nonbonded and all bonded terms with deterministic gathers (GPU-1 and INT-2 PASS under C3); charges from the stock CPU fix"},
    Feature{"metal.runtime_compile", Status::Implemented, "-", "(no LAMMPS equivalent)", "shaders compiled at run time from source with no Xcode / metal compiler; executed on the M5 Max (MET-1 PASS)"},
    Feature{"metal.device_neighbor_rows", Status::Implemented, "-", "(no LAMMPS equivalent)", "device far-neighbor rows over owned+ghost atoms with grow-and-retry; executed on the M5 Max (NBR-1 PASS, 7 geometries)"},
    Feature{"metal.deterministic_reduction", Status::Implemented, "-", "(no LAMMPS equivalent)", "fixed-order float reductions, bitwise equal to the CPU twin; executed on the M5 Max (FORCE-2, MET-4 PASS)"},
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
