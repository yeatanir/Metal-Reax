# Feature / Capability Matrix

Reference: `pair_style reaxff` + `fix qeq/reaxff` of LAMMPS `stable_30Sep2026` @ `8de817dd…` (see SOURCE_MAP.md).

**Status words** (identical to `reaxmetal::Status` in `include/reaxmetal/capabilities.hpp`; the `docs_sync`
test fails if this file and the code table disagree):

| Word | Meaning |
|---|---|
| Implemented | Exists **and** has passing tests recorded in VALIDATION.md. |
| Planned | Scheduled for the named milestone. Requesting it earlier throws `NotImplementedError`. |
| Deferred | Not supported **now**; may be scheduled later (owner decision). Requesting it throws `UnsupportedFeatureError` (an error, never a warning). |
| Rejected | Deliberately unsupported (no plan). Requesting it throws `UnsupportedFeatureError` (an error, never a warning). |
| Ignored | Accepted; has no effect on physics in this engine; a notice is logged. |

**Implemented so far** (tested, see VALIDATION.md): the force-field/control-file parser rows, the adapter rows, the system-geometry rows (M0.5–M3); **since M4–M6 (this document is checked against the code by `docs_sync`)** all energy terms except vdW type 2 (implemented, but no bundled force field uses it, so untested), the LAMMPS-compat flags, `lammps.pair_style_reaxff_metal` (energies, forces, virial by fdotr; `backend cpu64|metal`), `backend.cpu_fp64`, `backend.metal_fp32`, the three `metal.*` rows (executed on the M5 Max), `opt.lgvdw`, `opt.checkqeq_no` and `eem.external_cpu_fix` (charges come from the stock `fix qeq/reaxff`).
**Implemented never means fast or complete**: charges are not yet solved by this engine; the barostat and `out.virial` stay Planned until INT-4; per-atom energy/virial requests are refused; performance work (M8) has not started.
Milestone letters follow the mission statement (M2 parser … M8 optimisation) as re-planned in `ARCHITECTURE_DECISIONS.md` ADR-013
(M0.5 integration architecture; M7 is now LAMMPS-hosted validation, not a standalone MD engine).

## 1. Energy terms

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `term.bond` | Implemented | - | `ReaxFF::Bonds` | includes terminal-triple-bond stabilisation (ENGINE_SPEC 5.1) |
| `term.lone_pair` | Implemented | - | `ReaxFF::Atom_Energy` | includes the C2 correction (ENGINE_SPEC 5.2) |
| `term.over_under` | Implemented | - | `ReaxFF::Atom_Energy` | LAMMPS reports over+under merged as `ea` |
| `term.valence` | Implemented | - | `ReaxFF::Valence_Angles` | |
| `term.penalty` | Implemented | - | `ReaxFF::Valence_Angles` | |
| `term.coalition` | Implemented | - | `ReaxFF::Valence_Angles` | 3-body conjugation |
| `term.torsion` | Implemented | - | `ReaxFF::Torsion_Angles` | |
| `term.conjugation` | Implemented | - | `ReaxFF::Torsion_Angles` | 4-body conjugation |
| `term.hbond` | Implemented | - | `ReaxFF::Hydrogen_Bonds` | |
| `term.vdw.shielded` | Implemented | - | `vdw_type 1` | derived from ffield contents, not a switch |
| `term.vdw.inner_wall` | Planned | M5 | `vdw_type 2` | |
| `term.vdw.shielded_inner_wall` | Implemented | - | `vdw_type 3` | |
| `term.vdw.lg_dispersion` | Implemented | - | `pair_style reaxff lgvdw yes` | needs 5-line atom blocks |
| `term.coulomb` | Implemented | - | `ReaxFF::vdW_Coulomb_Energy` | taper + gamma shielding |
| `term.polarization` | Implemented | - | `ReaxFF::Compute_Polarization_Energy` | reported by LAMMPS as `eqeq` |

## 2. Parameter files

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `ffield.standard` | Implemented | - | `ffield general/atom/bond/angle/torsion/hbond blocks` | 39-parameter general block |
| `ffield.atom_line5_lgvdw` | Implemented | - | `ffield 5-line atom block` | only meaningful with lgvdw |
| `ffield.offdiagonal` | Implemented | - | `ffield off-diagonal block` | |
| `ffield.torsion_compact` | Implemented | - | `ffield 4-body entry 0-X-Y-0` | order-dependent overwrite (ENGINE_SPEC 2.5) |
| `ffield.hbond_block` | Implemented | - | `ffield hydrogen-bond block` | may be absent; LAMMPS warns and disables |
| `ffield.control_file` | Implemented | - | `pair_style reaxff <control file>` | cutoff keywords only |
| `ffield.strict_missing_pairs` | Implemented | - | `(no LAMMPS equivalent)` | we **reject** absent 2-body pairs; LAMMPS zero-fills creating a phantom bond with BO'=1 at every r <= bond_cut (ENGINE_SPEC Q-12, executed in M1) |
| `ffield.reject_three_body_overrun` | Implemented | - | `(no LAMMPS equivalent)` | reject >2 parameter sets for a j==l angle triple (ENGINE_SPEC Q-09: LAMMPS doubles the slot count and reads past prm[4]; executed in M1) |

## 3. LAMMPS-compat flags (element knowledge expressed as data — ADR-003)

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `compat.c2_correction` | Implemented | - | `strcmp(name,"C") in Atom_Energy` | per-type flag derived at load time; kernels never see element names |
| `compat.triple_bond_stabilisation` | Implemented | - | `gp.l[37]==2 or mass pair 12.0000/15.9990` | per-pair flag |
| `compat.light_element_split` | Implemented | - | `mass > 21 / mass < 21 tests` | per-type flag |
| `compat.hbond_donor_image_exclusion` | Implemented | - | `orig_id[i] != orig_id[k] in Hydrogen_Bonds` | reproduce by default (Q-32): acceptor that is a periodic image of the donor is dropped; identity-based variant needs owner decision |
| `compat.ovun_heavy_neighbor_force` | Implemented | - | `dDelta_lp[j] where the energy uses Delta_lp_temp[j] (Atom_Energy force loop)` | reproduce LAMMPS forces by default (Q-34: analytic force != gradient of the reported energy for heavy atoms with pi bonds); corrected variant is opt-in |

## 4. `pair_style reaxff` options

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `opt.enobonds` | Planned | M4 | `pair_style reaxff enobonds yes\|no` | default yes |
| `opt.checkqeq_no` | Implemented | - | `pair_style reaxff checkqeq no` | fixed input charges |
| `opt.lgvdw` | Implemented | - | `pair_style reaxff lgvdw yes` | |
| `opt.memory_heuristics` | Ignored | - | `safezone / mincap / minhbonds` | allocation heuristics only; notice logged |
| `opt.list_blocking` | Ignored | - | `list/blocking` | Kokkos performance option only |
| `opt.tabulate` | Deferred | - | `tabulate N>0 / tabulate_long_range N>0` | deferred; spline tables change the numbers, analytic evaluation only for now |

## 5. Charge models

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `qeq.reaxff` | Implemented | - | `fix qeq/reaxff ... reaxff` | stock fix, or fix qeq/reaxff/metal (EEM matrix and matvec on the GPU, stock CG); charges within 1.6e-5 e of stock |
| `qeq.pertype_file` | Planned | M5 | `fix qeq/reaxff ... <param file>` | per-type chi/eta/gamma override |
| `qeq.shielded` | Planned | M5 | `fix qeq/shielded` | LAMMPS-compatible shielded charge equilibration; same kernel as qeq/reaxff (ENGINE_SPEC 7.2); works through extract() today |
| `qeq.acks2` | Deferred | - | `fix acks2/reaxff` | deferred; different charge model |
| `qeq.qtpie` | Deferred | - | `fix qtpie/reaxff` | deferred; different charge model |
| `qeq.relative` | Deferred | - | `fix qeq/rel/reaxff` | deferred |
| `qeq.efield` | Deferred | - | `fix efield with fix qeq/reaxff` | deferred; external electric field |
| `qeq.group_subset` | Deferred | - | `fix qeq/reaxff on a proper subgroup` | deferred; all atoms are equilibrated |

## 6. System description

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `sys.pbc_images` | Implemented | - | `ghost atoms / periodic images` | CPU: standalone image expander (triclinic, mixed periodicity) = LAMMPS ghost set for all 58 fixtures (NBR-2); LAMMPS ghosts verified as owner + lattice shift inside LAMMPS (INT-7). Metal: written only |
| `sys.triclinic` | Implemented | - | `triclinic box` | CPU: expander, far list and ghost-native view on triclinic cells (NBR-1/2/3, INT-7) |
| `sys.nonperiodic` | Implemented | - | `boundary f/s/m` | CPU: boundary f along any direction (NBR-1, fixtures); shrink-wrapped/m boundaries are not separately tested |
| `sys.type_null_mapping` | Rejected | - | `pair_coeff ... NULL` | hybrid placeholder |
| `sys.hybrid` | Rejected | - | `pair_style hybrid[/overlay] with reaxff` | |

## 7. Outputs

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `out.energy_breakdown` | Implemented | - | `compute pair reaxff (pvector[14])` | mapping in `energy_terms.hpp`, tested |
| `out.forces` | Implemented | - | `atom->f` | analytical |
| `out.charges` | Planned | M5 | `atom->q` | |
| `out.virial` | Implemented | - | `virial_fdotr / v_tally*` | global virial by virial_fdotr (INT-2, NPT); per-atom virial by the CPU-64 tallies of the reference (PERATOM-1) |
| `out.per_atom_energy` | Implemented | - | `compute pe/atom, stress/atom with reaxff` | per-atom energy and virial equal stock on 58 fixtures (PERATOM-1, 5e-10 relative); produced by the CPU-64 engine, so a step on which a compute requests them is evaluated by CPU-64 even with backend metal |
| `out.bond_analysis` | Deferred | - | `fix reaxff/bonds, fix reaxff/species` | deferred; LAMMPS analysis tools dynamic_cast to PairReaxFF and refuse other styles |

## 8. Dynamics

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `md.nve` | Rejected | - | `fix nve` | standalone MD is out of scope: LAMMPS provides integrators |
| `md.thermostat` | Rejected | - | `fix nvt / langevin` | standalone MD is out of scope: LAMMPS provides thermostats |
| `md.barostat` | Implemented | - | `fix npt` | NPT (Nose-Hoover, iso) with Metal + GPU QEq: <T>, <V>, <P> agree with stock on a 648-atom water box over 40 000 steps (INT-4) |
| `min.minimize` | Rejected | - | `minimize` | standalone minimiser is out of scope: LAMMPS provides minimize |

## 9. Backends

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `backend.cpu_fp64` | Implemented | - | `-` | reference backend |
| `backend.cpu_fp32_twin` | Planned | M4 | `-` | same term functions in float; calibrates GPU tolerance (NUMERICAL_POLICY 5) |
| `backend.metal_fp32` | Implemented | - | `-` | Apple GPUs have no FP64 |
| `metal.runtime_compile` | Implemented | - | `(no LAMMPS equivalent)` | shaders compiled at run time from source (no Xcode / metal compiler); written, NOT built or run on Apple hardware (MET-1) |
| `metal.device_neighbor_rows` | Implemented | - | `(no LAMMPS equivalent)` | device far-neighbor rows over owned+ghost atoms with grow-and-retry; kernel logic verified by CPU emulation only (NBR-1 Metal part) |
| `metal.deterministic_reduction` | Implemented | - | `(no LAMMPS equivalent)` | fixed-order float reductions, bitwise equal to the CPU twin; emulated only (FORCE-2 Metal part) |
| `backend.metal_atomic_accum` | Planned | M8 | `-` | benchmark-only option; default path is deterministic |
| `backend.metal_batching` | Planned | M8 | `-` | |
| `backend.fp16` | Rejected | - | `-` | |

## 10. Variants explicitly out of the supported set

These are the "detect and refuse" cases required by architectural rule 4. Each is detected from the
**input**, not assumed absent:

| Condition in input | Detection point (from M2) | Outcome |
|---|---|---|
| Charge model other than `fix qeq/reaxff` | input/system description names `qeq/shielded`, `acks2`, `qtpie`, `qeq/rel` | `UnsupportedFeatureError` |
| Non-zero tabulation | control file `tabulate_long_range`, pair option `tabulate` | `UnsupportedFeatureError` |
| ffield with unknown block layout / wrong column count | ffield parser | error with file:line (LAMMPS-style) |
| ffield missing bond parameters for a type pair that occurs in the system | parser/system check | error (LAMMPS silently zero-fills; ENGINE_SPEC Q-12) |
| More than 5 three-body (or torsion) parameter sets for one type triple | parser | error (LAMMPS overruns a fixed array; ENGINE_SPEC Q-09) |
| Ignored LAMMPS control keywords (`nsteps`, `dt`, …) | control parser | notice, as LAMMPS warns |

## 11. LAMMPS integration (added in M0.5)

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `lammps.pair_style_reaxff_metal` | Implemented | - | `pair_style reaxff/metal` | computes energies, forces, the virial and per-atom energy/virial (backend cpu64 | metal); INT-2 vs stock pair reaxff on 58 fixtures |
| `lammps.plugin_loadable` | Implemented | - | `plugin load <reaxmetal plugin>` | DSO built against the pinned LAMMPS; version-matched |
| `lammps.extract_chi_eta_gamma` | Implemented | - | `Pair::extract(chi\|eta\|gamma)` | arrays indexed by LAMMPS type 1..ntypes, eta = 2x file value |
| `lammps.single_rank_only` | Implemented | - | `comm->nprocs == 1` | multi-rank runs fail explicitly (checked in init_style) |
| `lammps.multi_rank` | Deferred | - | `mpirun -np N>1 with reaxff/metal` | deferred; needs distributed ghost/QEq handling |
| `lammps.newton_off` | Rejected | - | `newton off (newton_pair off)` | forces on ghosts must be reverse-communicated |
| `lammps.ghost_native_contract` | Implemented | - | `ghost atoms from LAMMPS borders` | adapter A2 builds the owned+ghost view from LAMMPS arrays and verifies ghost = owner + shift (INT-7, 58 fixtures); far list equals LAMMPS' own list row by row |
| `lammps.virial_fdotr` | Implemented | - | `Pair::virial_fdotr_compute` | global virial/pressure from forces on owned+ghost atoms; pressure equals stock pair reaxff in INT-2 (cpu64 1e-10, metal 2.9e-4 relative); NPT water 40 000 steps agrees with stock |
| `lammps.ghost_shell_check` | Implemented | - | `(reference only warns, pair_reaxff.cpp:372-375)` | ghost shell < max(nonb_cut, hbond_cut, 2*bond_cut) is an error (INT-7 negative case) |
| `dev.neighbor_selfcheck` | Implemented | - | `(development aid)` | pair_style keyword reaxmetal_selfcheck yes: verify the host view and far list inside LAMMPS, then refuse to compute |

## 12. EEM charge model (added in M0.5)

"EEM" is the project's name for the standard ReaxFF charge model. It is **not a different physics** from LAMMPS `fix qeq/reaxff`
(ENGINE_SPEC §7.2 records exactly what is and is not established about the relation to AMS-ReaxFF's EEM).

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `eem.external_cpu_fix` | Implemented | - | `fix qeq/reaxff \| fix qeq/shielded (stock CPU)` | extract() half verified in M2 (EEM-3); the stock fix drives q once `compute()` exists |
| `eem.charge_verification` | Planned | M5 | `(no LAMMPS equivalent)` | adapter checks the EEM residual so strictness holds with the stock fix |
| `eem.taper_within_ghost_shell` | Planned | M5 | `(no LAMMPS equivalent)` | error if the QEq taper radius exceeds the ghost shell: the stock fix silently truncates (ENGINE_SPEC Q-35) |
| `eem.strict_convergence` | Planned | M5 | `(no LAMMPS equivalent)` | default: non-convergence is an error/status, never silently accepted |
| `eem.compat_warn_continue` | Planned | M5 | `fix qeq/reaxff default (warn and continue)` | explicit opt-in only |
| `eem.gpu_resident` | Planned | M5 | `(no LAMMPS equivalent)` | GPU-resident EEM solve, subject to numerical validation |
| `eem.net_charge_nonzero` | Deferred | - | `non-neutral fix group in fix qeq/reaxff` | LAMMPS imposes sum(q)=0; non-zero total charge not supported |
| `compat.flag_derivation` | Implemented | - | `compat predicates (ENGINE_SPEC Q-06)` | exact upstream predicates, boundary-tested (tests/test_compat_flags.cpp) |
