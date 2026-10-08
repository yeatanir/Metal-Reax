# Feature / Capability Matrix

Reference: `pair_style reaxff` + `fix qeq/reaxff` of LAMMPS `stable_30Sep2026` @ `8de817dd…` (see SOURCE_MAP.md).

**Status words** (identical to `reaxmetal::Status` in `include/reaxmetal/capabilities.hpp`; the `docs_sync`
test fails if this file and the code table disagree):

| Word | Meaning |
|---|---|
| Implemented | Exists **and** has passing tests recorded in VALIDATION.md. |
| Planned | Scheduled for the named milestone. Requesting it earlier throws `NotImplementedError`. |
| Rejected | Deliberately unsupported. Requesting it throws `UnsupportedFeatureError` (an error, never a warning). |
| Ignored | Accepted; has no effect on physics in this engine; a notice is logged. |

**At M0 no row is Implemented.** M0 delivers the audit, specification, decisions and a build skeleton only.
Milestone letters follow the mission statement (M2 parser … M8 optimisation).

## 1. Energy terms

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `term.bond` | Planned | M4 | `ReaxFF::Bonds` | includes terminal-triple-bond stabilisation (ENGINE_SPEC 5.1) |
| `term.lone_pair` | Planned | M4 | `ReaxFF::Atom_Energy` | includes the C2 correction (ENGINE_SPEC 5.2) |
| `term.over_under` | Planned | M4 | `ReaxFF::Atom_Energy` | LAMMPS reports over+under merged as `ea` |
| `term.valence` | Planned | M6 | `ReaxFF::Valence_Angles` | |
| `term.penalty` | Planned | M6 | `ReaxFF::Valence_Angles` | |
| `term.coalition` | Planned | M6 | `ReaxFF::Valence_Angles` | 3-body conjugation |
| `term.torsion` | Planned | M6 | `ReaxFF::Torsion_Angles` | |
| `term.conjugation` | Planned | M6 | `ReaxFF::Torsion_Angles` | 4-body conjugation |
| `term.hbond` | Planned | M6 | `ReaxFF::Hydrogen_Bonds` | |
| `term.vdw.shielded` | Planned | M5 | `vdw_type 1` | derived from ffield contents, not a switch |
| `term.vdw.inner_wall` | Planned | M5 | `vdw_type 2` | |
| `term.vdw.shielded_inner_wall` | Planned | M5 | `vdw_type 3` | |
| `term.vdw.lg_dispersion` | Planned | M5 | `pair_style reaxff lgvdw yes` | needs 5-line atom blocks |
| `term.coulomb` | Planned | M5 | `ReaxFF::vdW_Coulomb_Energy` | taper + gamma shielding |
| `term.polarization` | Planned | M5 | `ReaxFF::Compute_Polarization_Energy` | reported by LAMMPS as `eqeq` |

## 2. Parameter files

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `ffield.standard` | Planned | M2 | `ffield general/atom/bond/angle/torsion/hbond blocks` | 39-parameter general block |
| `ffield.atom_line5_lgvdw` | Planned | M2 | `ffield 5-line atom block` | only meaningful with lgvdw |
| `ffield.offdiagonal` | Planned | M2 | `ffield off-diagonal block` | |
| `ffield.torsion_compact` | Planned | M2 | `ffield 4-body entry 0-X-Y-0` | order-dependent overwrite (ENGINE_SPEC 2.5) |
| `ffield.hbond_block` | Planned | M2 | `ffield hydrogen-bond block` | may be absent; LAMMPS warns and disables |
| `ffield.control_file` | Planned | M2 | `pair_style reaxff <control file>` | cutoff keywords only |
| `ffield.strict_missing_pairs` | Planned | M2 | `(no LAMMPS equivalent)` | we **reject** absent 2-body pairs; LAMMPS zero-fills (ENGINE_SPEC Q-12) |

## 3. LAMMPS-compat flags (element knowledge expressed as data — ADR-003)

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `compat.c2_correction` | Planned | M4 | `strcmp(name,"C") in Atom_Energy` | per-type flag derived at load time; kernels never see element names |
| `compat.triple_bond_stabilisation` | Planned | M4 | `gp.l[37]==2 or mass pair 12.0000/15.9990` | per-pair flag |
| `compat.light_element_split` | Planned | M4 | `mass > 21 / mass < 21 tests` | per-type flag |

## 4. `pair_style reaxff` options

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `opt.enobonds` | Planned | M4 | `pair_style reaxff enobonds yes\|no` | default yes |
| `opt.checkqeq_no` | Planned | M5 | `pair_style reaxff checkqeq no` | fixed input charges |
| `opt.lgvdw` | Planned | M5 | `pair_style reaxff lgvdw yes` | |
| `opt.memory_heuristics` | Ignored | - | `safezone / mincap / minhbonds` | allocation heuristics only; notice logged |
| `opt.list_blocking` | Ignored | - | `list/blocking` | Kokkos performance option only |
| `opt.tabulate` | Rejected | - | `tabulate N>0 / tabulate_long_range N>0` | spline tables change the numbers; analytic only |

## 5. Charge models

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `qeq.reaxff` | Planned | M5 | `fix qeq/reaxff ... reaxff` | preconditioned CG, history extrapolation (ENGINE_SPEC 7) |
| `qeq.pertype_file` | Planned | M5 | `fix qeq/reaxff ... <param file>` | per-type chi/eta/gamma override |
| `qeq.shielded` | Rejected | - | `fix qeq/shielded` | different model |
| `qeq.acks2` | Rejected | - | `fix acks2/reaxff` | different model |
| `qeq.qtpie` | Rejected | - | `fix qtpie/reaxff` | different model |
| `qeq.relative` | Rejected | - | `fix qeq/rel/reaxff` | |
| `qeq.efield` | Rejected | - | `fix efield with fix qeq/reaxff` | |
| `qeq.group_subset` | Rejected | - | `fix qeq/reaxff on a proper subgroup` | all atoms are equilibrated |

## 6. System description

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `sys.pbc_images` | Planned | M3 | `ghost atoms / periodic images` | explicit image shifts; **never minimum-image only** (ADR-004) |
| `sys.triclinic` | Planned | M3 | `triclinic box` | |
| `sys.nonperiodic` | Planned | M3 | `boundary f/s/m` | |
| `sys.type_null_mapping` | Rejected | - | `pair_coeff ... NULL` | hybrid placeholder |
| `sys.hybrid` | Rejected | - | `pair_style hybrid[/overlay] with reaxff` | |

## 7. Outputs

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `out.energy_breakdown` | Planned | M4 | `compute pair reaxff (pvector[14])` | mapping in `energy_terms.hpp`, tested |
| `out.forces` | Planned | M6 | `atom->f` | analytical |
| `out.charges` | Planned | M5 | `atom->q` | |
| `out.virial` | Planned | M6 | `virial_fdotr / v_tally*` | needed for pressure |
| `out.per_atom_energy` | Rejected | - | `compute pe/atom with reaxff` | |
| `out.bond_analysis` | Rejected | - | `fix reaxff/bonds, fix reaxff/species` | |

## 8. Dynamics

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `md.nve` | Planned | M7 | `fix nve` | |
| `md.thermostat` | Planned | M7 | `fix nvt / langevin` | choice deferred to M7 |
| `md.barostat` | Rejected | - | `fix npt` | |
| `min.minimize` | Planned | M7 | `minimize` | |

## 9. Backends

| Feature | Status | Milestone | LAMMPS construct | Notes |
|---|---|---|---|---|
| `backend.cpu_fp64` | Planned | M4 | `-` | reference backend |
| `backend.cpu_fp32_twin` | Planned | M4 | `-` | same term functions in float; calibrates GPU tolerance (NUMERICAL_POLICY 5) |
| `backend.metal_fp32` | Planned | M3 | `-` | Apple GPUs have no FP64 |
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
