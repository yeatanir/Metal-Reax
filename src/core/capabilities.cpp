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
    Feature{"term.vdw.inner_wall", Status::Implemented, "-", "vdw_type 2", "Morse + inner wall, no shielding; VAR-1: a ffield edited to select it matches stock in-LAMMPS (cpu64 and metal), 6 fixtures"},
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
    Feature{"lammps.pair_style_reaxff_metal", Status::Implemented, "-", "pair_style reaxff/metal", "computes energies, forces, the virial and per-atom energy/virial (backend cpu64 | metal); INT-2 vs stock pair reaxff on 58 fixtures"},
    Feature{"lammps.plugin_loadable", Status::Implemented, "-", "plugin load <reaxmetal plugin>", "DSO built against the pinned LAMMPS; version-matched"},
    Feature{"lammps.extract_chi_eta_gamma", Status::Implemented, "-", "Pair::extract(chi|eta|gamma)", "arrays indexed by LAMMPS type 1..ntypes, eta = 2x file value"},
    Feature{"qeq.gpu_single_rank", Status::Implemented, "-", "(no LAMMPS equivalent)", "fix qeq/reaxff/metal builds the matrix on the GPU only with one MPI rank; with several ranks it uses the stock CPU matrix (the pair style itself runs on any number of ranks)"},
    Feature{"lammps.hybrid", Status::Implemented, "-", "pair_style hybrid/overlay reaxff ... + other styles", "reaxff/metal as a sub-style of hybrid/overlay (type mapping with NULL entries, forces and energies added to the other styles); HYB-1: charge-implicit ReaxFF + tabulated ZBL example equals stock (cpu64 bit-for-bit at printed precision, metal 2e-5 kcal/mol/atom). Keyword shellcheck no accepts a ghost shell narrower than 2*bond_cut as the stock style does"},
    Feature{"lammps.multi_rank", Status::Implemented, "-", "mpirun -np N>1 with reaxff/metal", "ghost atoms whose owner lives on another rank are taken as LAMMPS delivers them (owner-computes rules as the reference); INT-2 vs stock on 58 fixtures with 2 and 4 ranks (cpu64 under C1, metal under C3); 5 184-atom NVT water on 4 ranks agrees with stock"},
    Feature{"lammps.newton_off", R, "-", "newton off (newton_pair off)", "forces on ghosts must be reverse-communicated"},
    Feature{"lammps.ghost_native_contract", Status::Implemented, "-", "ghost atoms from LAMMPS borders", "adapter A2 builds the owned+ghost view from LAMMPS arrays and verifies ghost = owner + shift (INT-7, 58 fixtures); far list equals LAMMPS' own list row by row"},
    Feature{"lammps.virial_fdotr", Status::Implemented, "-", "Pair::virial_fdotr_compute", "global virial/pressure from forces on owned+ghost atoms; pressure equals stock pair reaxff in INT-2 (cpu64 1e-10, metal 2.9e-4 relative); NPT water 40 000 steps agrees with stock"},
    Feature{"lammps.ghost_shell_check", Status::Implemented, "-", "(reference only warns, pair_reaxff.cpp:372-375)", "ghost shell < max(nonb_cut, hbond_cut, 2*bond_cut) is an error (INT-7 negative case)"},
    Feature{"dev.neighbor_selfcheck", Status::Implemented, "-", "(development aid)", "pair_style keyword reaxmetal_selfcheck yes: verify the host view and far list inside LAMMPS, then stop with the summary (development aid)"},
    // ---- EEM charge model (naming: EEM == the standard ReaxFF charge model; not a different physics) ---
    Feature{"eem.external_cpu_fix", Status::Implemented, "-", "fix qeq/reaxff | fix qeq/shielded (stock CPU)", "the stock fix qeq/reaxff drives q through extract() and compute(); charges identical to stock in INT-2"},
    Feature{"eem.charge_verification", Status::Implemented, "-", "(no LAMMPS equivalent)", "fix qeq/reaxff/metal ... verify <eV>: after every solve max_i |(H q)_i + chi_i - mu| is evaluated with the matrix of the solve and an error is raised above <eV> (EEM-1; sees the FP32 matrix error: 1e-7 eV fails on Metal, passes on the double CPU matrix)"},
    Feature{"eem.taper_within_ghost_shell", Status::Implemented, "-", "(no LAMMPS equivalent)", "fix qeq/reaxff/metal ... strict: error if the taper radius exceeds the ghost cutoff (stock silently truncates the matrix, Q-35); without strict the stock behaviour is kept (EEM-1)"},
    Feature{"eem.strict_convergence", Status::Implemented, "-", "(no LAMMPS equivalent)", "fix qeq/reaxff/metal ... strict: CG non-convergence is an error with the residual and the step (EEM-1). Opt-in keyword, not the default, so that the default stays comparable with the stock fix"},
    Feature{"eem.compat_warn_continue", Status::Implemented, "-", "fix qeq/reaxff default (warn and continue)", "the default of fix qeq/reaxff/metal without the keyword strict: warn and continue exactly like the stock fix (EEM-1)"},
    Feature{"eem.gpu_resident", Status::Implemented, "-", "(no LAMMPS equivalent)", "fix qeq/reaxff/metal ... resident: matrix and residual in double-single arithmetic on the GPU, correction solves by an FP32 preconditioned CG that never leaves the device (mixed-precision refinement); charges equal the stock fix's to 1e-10 e (INT-2). Single rank. Speed: neutral to ~20% faster than the host-CG mode at 5k atoms, neutral at 24k, slower at 66k (measured on a loaded machine)"},
    Feature{"eem.net_charge_nonzero", D, "-", "non-neutral fix group in fix qeq/reaxff", "LAMMPS imposes sum(q)=0; non-zero total charge not supported"},
    Feature{"compat.flag_derivation", Status::Implemented, "-", "compat predicates (ENGINE_SPEC Q-06)", "exact upstream predicates, boundary-tested (tests/test_compat_flags.cpp)"},
    // ---- pair_style options ----------------------------------------------------------------------
    Feature{"opt.enobonds", Status::Implemented, "-", "pair_style reaxff enobonds yes|no", "enobonds no matches stock in-LAMMPS on 7 fixtures incl. isolated atoms (VAR-1, cpu64 and metal)"},
    Feature{"opt.checkqeq_no", Status::Implemented, "-", "pair_style reaxff checkqeq no", "fixed input charges"},
    Feature{"opt.lgvdw", Status::Implemented, "-", "pair_style reaxff lgvdw yes", ""},
    Feature{"opt.memory_heuristics", I, "-", "safezone / mincap / minhbonds", "LAMMPS allocation heuristics only"},
    Feature{"opt.list_blocking", I, "-", "list/blocking", "Kokkos performance option only"},
    Feature{"opt.tabulate", Status::Implemented, "-", "tabulate N>0 / tabulate_long_range N>0", "tabulate N / tabulate_long_range N are accepted; no table is built, the non-bonded terms are evaluated analytically (the stock spline table approximates them: a tabulated stock run differs by its interpolation error). A notice says so"},
    // ---- charge models ---------------------------------------------------------------------------
    Feature{"qeq.reaxff", Status::Implemented, "-", "fix qeq/reaxff ... reaxff", "stock fix, or fix qeq/reaxff/metal (EEM matrix and matvec on the GPU, stock CG); charges within 1.6e-5 e of stock"},
    Feature{"qeq.pertype_file", Status::Implemented, "-", "fix qeq/reaxff ... <param file>", "per-type chi/eta/gamma from a parameter file instead of the pair style: identical to stock (CHG-1: 3000-atom water example, stock vs plugin, cpu64 and metal)"},
    Feature{"qeq.shielded", Status::Implemented, "-", "fix qeq/shielded", "LAMMPS-compatible shielded charge equilibration: stock fix, charges drive the plugin (CHG-1: 3000-atom water example, stock vs plugin, cpu64 and metal)"},
    Feature{"qeq.acks2", Status::Implemented, "-", "fix acks2/reaxff", "ACKS2: the stock fix supplies the kinetic potentials; the plugin adds the polarization coupling and the bond-softness Coulomb term (energy, forces, per-atom tallies) to the non-bonded terms (CHG-1: 3000-atom water example, stock vs plugin, cpu64 and metal)"},
    Feature{"qeq.qtpie", Status::Implemented, "-", "fix qtpie/reaxff", "QTPIE: use fix qtpie/reaxff/metal (the stock fix borrows the pair-style list with ghost rows, which a plugin pair style cannot provide; the derived fix requests its own) (CHG-1: 3000-atom water example, stock vs plugin, cpu64 and metal)"},
    Feature{"qeq.relative", Status::Implemented, "-", "fix qeq/rel/reaxff", "QEq-R: use fix qeq/rel/reaxff/metal (same reason as QTPIE) (CHG-1: 3000-atom water example, stock vs plugin, cpu64 and metal)"},
    Feature{"qeq.efield", Status::Implemented, "-", "fix efield with fix qeq/reaxff", "external electric field with every charge model above, also with the GPU charge matrix (CHG-1: 3000-atom water example, stock vs plugin, cpu64 and metal)"},
    Feature{"qeq.group_subset", Status::Implemented, "-", "fix qeq/reaxff on a proper subgroup", "the stock fix handles the group; the plugin does not depend on it (GPU charge matrix falls back to the CPU one) (CHG-1: 3000-atom water example, stock vs plugin, cpu64 and metal)"},
    // ---- system description ----------------------------------------------------------------------
    Feature{"sys.pbc_images", Status::Implemented, "-", "ghost atoms / periodic images", "CPU: standalone image expander (triclinic, mixed periodicity) = LAMMPS ghost set for all 58 fixtures (NBR-2); LAMMPS ghosts verified as owner + lattice shift inside LAMMPS (INT-7). Metal: written only"},
    Feature{"sys.triclinic", Status::Implemented, "-", "triclinic box", "CPU: expander, far list and ghost-native view on triclinic cells (NBR-1/2/3, INT-7)"},
    Feature{"sys.nonperiodic", Status::Implemented, "-", "boundary f/s/m", "CPU: boundary f along any direction (NBR-1, fixtures); shrink-wrapped/m boundaries are not separately tested"},
    Feature{"sys.type_null_mapping", R, "-", "pair_coeff ... NULL", "hybrid placeholder; not planned"},
    Feature{"sys.hybrid", R, "-", "pair_style hybrid[/overlay] with reaxff", "not planned"},
    // ---- outputs ---------------------------------------------------------------------------------
    Feature{"out.energy_breakdown", Status::Implemented, "-", "compute pair reaxff (pvector[14])", "see energy_terms.hpp"},
    Feature{"out.forces", Status::Implemented, "-", "atom->f", "analytical"},
    Feature{"out.charges", Status::Implemented, "-", "atom->q", "charges are LAMMPS atom->q set by the stock fix qeq/reaxff or by fix qeq/reaxff/metal; compared with stock in every INT-2 run"},
    Feature{"out.virial", Status::Implemented, "-", "virial_fdotr / v_tally*", "global virial by virial_fdotr (INT-2, NPT); per-atom virial by the CPU-64 tallies of the reference (PERATOM-1)"},
    Feature{"out.per_atom_energy", Status::Implemented, "-", "compute pe/atom, stress/atom with reaxff", "per-atom energy and virial equal stock on 58 fixtures (PERATOM-1, 5e-10 relative); produced by the CPU-64 engine, so a step on which a compute requests them is evaluated by CPU-64 even with backend metal"},
    Feature{"out.bond_analysis", D, "-", "fix reaxff/bonds, fix reaxff/species", "deferred; LAMMPS analysis tools dynamic_cast to PairReaxFF and refuse other styles"},
    // ---- dynamics --------------------------------------------------------------------------------
    Feature{"md.nve", R, "-", "fix nve", "standalone MD is out of scope: LAMMPS provides integrators"},
    Feature{"md.thermostat", R, "-", "fix nvt / langevin", "standalone MD is out of scope: LAMMPS provides thermostats"},
    Feature{"md.barostat", Status::Implemented, "-", "fix npt", "NPT (Nose-Hoover, iso) with Metal + GPU QEq: <T>, <V>, <P> agree with stock on a 648-atom water box over 40 000 steps (INT-4)"},
    Feature{"min.minimize", R, "-", "minimize", "standalone minimiser is out of scope: LAMMPS provides minimize"},
    // ---- backends --------------------------------------------------------------------------------
    Feature{"backend.cpu_fp64", Status::Implemented, "-", "-", "reference backend: all 13 energy terms and forces equal pinned LAMMPS (FULL-1, INT-2)"},
    Feature{"backend.cpu_fp32_twin", Status::Implemented, "-", "-", "the CPU engine compiled with float (Real = float, term values in float, energy sums in double); FP32 error envelope vs CPU-64 on 58 fixtures: E/atom 6e-5, force max 4.7e-3, RMS 2.2e-3 (FP32-1); Metal stays within it"},
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
