# Source Map

Maps every upstream source the engine is derived from to the engine module that will own its behaviour,
the milestone in which it is implemented, and the evidence used in the M0 audit. Nothing here is
implemented yet (M0).

## 1. Pinned upstream

| Item | Value |
|---|---|
| Repository | `https://github.com/lammps/lammps.git` |
| Release tag | `stable_30Sep2026` (annotated tag object `752e990c8d4c30bb134f6c1b630a2ab056cbc087`) |
| **Commit** | **`8de817dd79bfe4525d5d39246a212d833e6dee07`** (2026-09-30 19:17:59 -0400, "Merge branch 'maintenance' into stable") |
| License of the tree | GNU GPL v2 — `third_party/lammps/COPYING` (byte-identical copy of upstream `LICENSE`, sha256 `be38e38d9482c2beae35…`) |
| Selection rationale | Latest `stable_*` tag on the date of the audit (2026-10-08); `stable_*` tags are LAMMPS' maintained releases. Alternative: an older stable — see ARCHITECTURE_DECISIONS ADR-001 |
| Machine-readable pin | `third_party/lammps/PIN.txt` (checked against `include/reaxmetal/pins.hpp` by `ctest -R pins`) |
| Integrity manifest | `third_party/lammps/SOURCE_HASHES.sha256` — sha256 of all **142** upstream files used (REAXFF sources, Kokkos ReaxFF files, ffields, reaxff unit-test YAMLs, docs; M0.5 added the QEQ and PLUGIN sources, `pair.h/.cpp`, `force.cpp`, `comm.cpp`, `verlet.cpp`, plugin examples/docs and the unit-test harness files) |
| Reproduce | `tools/fetch_lammps.sh <dir>` — shallow sparse clone, verifies commit, tag object and every hash. **No upstream code is vendored in this repository.** |

M0 verified (actual run, see DEVELOPMENT_LOG): `sha256sum -c SOURCE_HASHES.sha256` → 72/72 OK at M0 (142/142 after the M0.5 extension) against the audit
checkout; `tools/fetch_lammps.sh` re-run result recorded in the log.

The reaxff regression YAMLs in this tree record `lammps_version: 2 Sep 2026` and `date_generated: Mon Sep 14 2026`,
i.e. their reference *numbers* were produced by a LAMMPS older than the pinned commit (2026-09-30). We do not
rely on them as exact pins; see §5.

## 2. Papers supplied with the mission

| Paper | What the engine takes from it | What it does **not** settle |
|---|---|---|
| Zheng, Li, Guo, *Algorithms of GPU-enabled ReaxFF molecular dynamics* (GMD-Reax), J. Mol. Graph. Model. 41 (2013) 1–11 | Two-kernel BO split (raw BO + valency, then corrected BO); column-major (coalesced) per-atom bond lists with a bond-count array; ELLPACK + "T threads per row" SpMV for QEq; single-precision math with fast SFU functions reached "<0.01 %" energy deviation on one static step (their Table 1: 1378 and 27 283 atoms, vs Duin's Fortran in LAMMPS) | Only energies of a single step were compared — **no forces, no drift**; periodic boundary via **minimum image** (invalid for cells < 2×cutoff, the LAMMPS unit-test cell is 7.54 Å vs 10 Å cutoff); max ≈40 000 atoms on 3 GB; QEq is the remaining bottleneck (80–90 % of GPU time) |
| Kylasa, Aktulga, Grama, *PuReMD-GPU* (Purdue CS TR 13-005, 2012) | Redundant (full) neighbor/bond/H-bond lists for parallelism; **delaying bond-order derivatives to the end of the step** via accumulated `dEdBO` (this is exactly LAMMPS' `Cdbo`/`CdDelta` scheme); additional memory instead of atomics for conflict-free accumulation (Table 5, FP64: equal for bond-order and nonbonded kernels, but atomics were ≈7× (4-body) to ≈150× (lone pair) slower elsewhere); 1 thread/atom for 3-/4-body kernels but multiple threads/atom for neighbor, H-bond, SpMV, nonbonded (Table 4); lookup tables did **not** help on GPU; atomics were retained only for 4-body force updates | All measurements are FP64 on a 2012 Tesla C2075 — the atomic-vs-memory conclusion is **not** transferable to Apple GPUs / FP32 atomics and must be re-measured (mission rule 9, M8). I read pages 1–26 of the 30-page report in M0; pages 27–30 (accuracy tables/conclusion/refs) were **not** read and should be read before M8 |

## 3. Pinned-source file map

Responsibility → engine module (`src/<module>`, ADR-006) → milestone. "Provenance" is the license text carried
by the upstream file (checked by grep in M0, §6).

### 3.1 `src/REAXFF/` (the oracle)

| Upstream file | Responsibility | Engine target | Milestone | Provenance |
|---|---|---|---|---|
| `reaxff_ffield.cpp` | ffield parser, derived mixing, vdw_type detection | `io/ffield` | M2 | PuReMD, GPL-2.0-or-later text |
| `reaxff_control.cpp` | control file keywords | `io/control` | M2 | PuReMD, GPL-2.0-or-later text |
| `reaxff_types.h`, `reaxff_defs.h`, `reaxff_api.h`, `reaxff_inline.h` | parameter/storage structs, constants | `core/types`, `core/constants` | M2 | LAMMPS-authored, GPL (LICENSE: v2) |
| `pair_reaxff.cpp/.h` | driver: neighbor → lists → forces; option parsing; pvector mapping | `cpu/driver`, `core/energy_terms` | M4–M6 | LAMMPS-authored, GPL v2 |
| `reaxff_forces.cpp` | list construction (`Init_Forces_noQEq`), driver order, `Compute_Total_Force` | `cpu/lists`, `cpu/force_assembly` | M4, M6 | PuReMD |
| `reaxff_bond_orders.cpp` | `BOp`, `BO`, `Add_dBond_to_Forces` | `physics/bond_order`, `cpu/force_assembly` | M4 | PuReMD |
| `reaxff_bonds.cpp` | bond energy + triple-bond stabilisation | `physics/bond_energy` | M4 | PuReMD |
| `reaxff_multi_body.cpp` | lone pair, C2, over/under | `physics/atom_energy` | M4 | PuReMD |
| `reaxff_valence_angles.cpp` | angle, penalty, coalition, `dcos_θ` | `physics/valence` | M6 | PuReMD |
| `reaxff_torsion_angles.cpp` | torsion, 4-body conjugation, `ω` | `physics/torsion` | M6 | PuReMD |
| `reaxff_hydrogen_bonds.cpp` | H-bond | `physics/hbond` | M6 | PuReMD |
| `reaxff_nonbonded.cpp` | vdW, Coulomb, polarization, (tabulated, rejected), ACKS2 part (rejected) | `physics/nonbonded` | M5 | PuReMD |
| `reaxff_init_md.cpp` | taper coefficients, capacity heuristics | `core/taper` | M5 | PuReMD |
| `reaxff_lookup.cpp` | spline tables for tabulation | — (rejected) | — | PuReMD |
| `reaxff_reset_tools.cpp`, `reaxff_allocate.cpp`, `reaxff_list.cpp`, `reaxff_tool_box.cpp` | memory heuristics | — (not needed; ADR-006) | — | PuReMD |
| `fix_qeq_reaxff.cpp/.h` | QEq: H matrix, CG, history extrapolation | `qeq/` | M5 | LAMMPS-authored, GPL v2 |
| `fix_acks2_reaxff`, `fix_qtpie_reaxff`, `fix_qeq_rel_reaxff` | other charge models | — (rejected) | — | LAMMPS-authored |
| `fix_reaxff_bonds/species`, `compute_reaxff_atom`, `compute_spec_atom`, `fix_reaxff` | analysis/bookkeeping | — (rejected / bookkeeping only) | — | LAMMPS-authored |

### 3.2 `src/KOKKOS/` (the closest analogue of a GPU port)

| Upstream file | Used for | Notes from the M0 audit |
|---|---|---|
| `pair_reaxff_kokkos.cpp/.h` (4 415 + 637 lines) | structural precedent for GPU kernels; secondary cross-check (`reaxff/kk`, double) | `ScatterView`: atomics on GPU, duplicated on OpenMP, plain on serial; "blocking" BuildLists and Torsion kernels to reduce divergence; `cbrt` instead of `pow(x,0.33333333333333)`; reduced-exponential rewrite; hard-coded `thb_cutsq = 1e-5` (Q-17); contributors from NVIDIA/AMD optimisation |
| `fix_qeq_reaxff_kokkos.cpp/.h` | GPU QEq precedent | fused two-vector CG (`s` and `t` solved together), optional matrix-free H (`matfree`), `KK_FLOAT` precision |
| `kokkos_type.h` | precision switch | `KOKKOS_PREC = double \| mixed \| single` (`cmake/Modules/Packages/KOKKOS.cmake:15-25`) — LAMMPS can build ReaxFF/kk in `mixed` (FP32 compute, FP64 accumulate) and `single`; it skips its regression tests for `single` and tests `mixed` only at a vacuous tolerance (relative 2.0; Q-25) |

## 4. Fixture inventory candidates (hashes in `SOURCE_HASHES.sha256`)

Bundled ffields that exercise different branches (survey by `tools/survey_ffield.py`, read-only analysis of the
pinned files, M0 run). *Elements are listed only to document coverage; no element is special-cased anywhere.*

| ffield | Elements (in file order) | Notable branches exercised |
|---|---|---|
| `ffield.reax.cho` | C H O | minimal organic; 1 H-bond triple; 5 compact torsions |
| `ffield.reax.rdx` | C H N O | `gp[37]=2` (stabilisation for all pairs); 4 H-bond rows |
| `ffield.reax.lg` | C H O N S | 5-line atoms, `vdw_type 3`, needs `lgvdw yes` |
| `ffield.reax.AB` | H O N B X | `vdw_type 3` without lg; compact torsions; `X` dummy element with `valency_val ≠ valency_boc` |
| `ffield.reax.FC` | C H O N S F Pt Cl Ni X | 10 types; 107 three-body rows incl. 3 duplicate/mirror sets; widest combinatorics |
| `ffield.reax.Fe_O_C_H` | C H O Fe | metal (`mass>21`), mirror 3-body duplicate |
| `ffield.reax.V_O_C_H`, `.ZnOH`, `.AuO` | V / Zn / Au + H O (+C) | metal oxides/hydroxides, 1 H-bond row, no compact torsion in ZnOH |
| `ffield.reax.mattsson`, `.budzien` | C H O N (S) | 9 H-bond rows (mattsson) |

Survey facts (all from the M0 run, parser-branch coverage only, **not** validation): 3-body rows with `j==l`
exist in every file (6–33 rows), so the Q-09 double-increment path is exercised by every real ffield; compact
`0-X-Y-0` torsions exist in 10 of 11; `vdw_type` ∈ {1, 3} (type 2-only not covered by bundled files);
only `ffield.reax.rdx` sets `gp[37]=2`.

## 5. Independent regression data shipped with the pinned LAMMPS

`unittest/force-styles/tests/atomic-pair-reaxff{,_noqeq,_lgvdw,_tabulate,_tabulate_flag,-acks2,-qtpie,…}.yaml`:
64-atom diamond lattice (`lattice diamond 3.77`, 2×2×2 → **7.54 Å cell, smaller than the 10 Å taper cutoff**, so
multi-image handling is mandatory), 2–3 atom types, `fix qeq/reaxff 1 0.0 8.0 1.0e-20 reaxff`, record `init_vdwl/init_coul`
and per-atom forces with CI epsilon between **1e-11 and 2e-10**. They are *not* our oracle (we generate our own,
hashed, instrumented fixtures in M1) but are a free cross-check of the M1 generator and a measure of LAMMPS'
own cross-compiler noise floor (NUMERICAL_POLICY §3). These YAMLs `skip_tests: kokkos_*_single`; error metric of the harness is relative, `|a−b|/min(|a|,|b|)` (`unittest/force-styles/test_main.h:37-44`).

**Granularity gap for M1** (design input, not yet solved): the unmodified LAMMPS exposes (a) 14 energy slots
with over+under merged (`ea`), (b) total forces, (c) charges. It does **not** expose bond orders, `Δ`, `nlp`,
or per-term forces. Localising a mismatch to a term therefore needs an *instrumented* LAMMPS build via a small
patch kept in this repository and hash-pinned. Open question Q1 in the M0 report.

## 6. License provenance of audited files (grep of the pinned tree, M0)

* **PuReMD-derived, text "GPL version 2 … or (at your option) any later version", © 2010 Purdue University**
  (H. M. Aktulga, J. Fogarty, S. Pandit, A. Grama): `reaxff_{allocate,bond_orders,bonds,control,ffield,forces,
  hydrogen_bonds,init_md,list,lookup,multi_body,nonbonded,reset_tools,tool_box,torsion_angles,valence_angles}.cpp`
  and `REAXFF/README`.
* **LAMMPS-authored, "distributed under the GNU General Public License" + LAMMPS `LICENSE` (GPL v2)**:
  `pair_reaxff.*`, `fix_qeq_reaxff.*`, `fix_acks2_reaxff.*`, `fix_qtpie_reaxff.*`, `fix_qeq_rel_reaxff.*`,
  `fix_reaxff*.`, `compute_*`, `reaxff_{api,defs,inline,types}.h`, and all four `src/KOKKOS/*reaxff*` files.
* **Consequence (not legal advice, flagged for the owner — ADR-010):** code adapted *only* from the PuReMD-derived
  physics files may carry `GPL-2.0-or-later`; as soon as anything adapted from a LAMMPS-authored file
  (`fix_qeq_reaxff`, `pair_reaxff`, Kokkos) is combined, the combination is `GPL-2.0-only`. Every adapted source
  file in this repository must carry the original copyright/notice block (template:
  `docs/NOTICE_TEMPLATE.txt`) plus an `Adapted-from:` line naming upstream file and commit. M0 adapts no code.
* The ffield files in LAMMPS `potentials/` are distributed with LAMMPS; their scientific provenance
  (Aidan Thompson contributor lines, publications in `potentials/README.reax`) must be cited wherever they are
  used as fixtures. Whether they are copied into this repository or fetched on demand is decided in M1.

## 7. LAMMPS host-API sources audited in M0.5 (all hash-pinned in `SOURCE_HASHES.sha256`)

| Upstream file | What it settled | Doc |
|---|---|---|
| `doc/src/Developer_plugins.rst`, `doc/src/plugin.rst`, `src/PLUGIN/plugin.cpp`, `src/lammpsplugin.h`, `examples/plugins/*` | plugin entry point, registry (global since 2 Sep 2026), version message (not enforced), build flags per platform | LAMMPS_INTEGRATION §2 |
| `src/pair.h`, `src/pair.cpp` (`init`, `ev_setup`, `virial_fdotr_compute`, `init_style`) | pair-style flags, eflag/vflag decoding, fdotr virial over owned+ghost, default neighbor request | §3 |
| `src/force.cpp` (`pair_match`) | `^reaxff` regex matches `reaxff/metal` | §3.3 |
| `src/neighbor.h/.cpp`, `src/comm.cpp` (`get_comm_cutoff`) | request flags; ghost shell = `max(comm_modify cutoff, cutforce+skin)` | §4 |
| `src/verlet.cpp` | per-step order: neighbor/comm → `force_clear` → `pre_force` → `pair->compute` → `reverse_comm` | §3.1 |
| `src/QEQ/fix_qeq*.{cpp,h}` (`fix qeq/shielded`) | same kernel/constants as `qeq/reaxff`; full list, H×0.5, history, `swa=0` | §7 |
| `src/REAXFF/fix_qeq_reaxff.*` (re-read for access/extract) | protected state, no `extract`, own-list fallback when the pair is not `PairReaxFF` | §7.1, §7.4 |
| `unittest/force-styles/test_main.h`, `test_pair_style.cpp` | LAMMPS' own tolerance semantics (relative `|a−b|/min`), reduced-precision epsilon scaling | NUMERICAL_POLICY §5 |

**License provenance (extended, M0.5):** `third_party/lammps/LICENSE_AUDIT.tsv` (generated by `tools/audit_licenses.py`) classifies 130 files: 76 LAMMPS-GPL, 17 PuReMD GPL-2.0-or-later, 1 MIT-style (`src/fmt/format.h`, only *included* by plugin builds), 35 with no per-file notice
(ffields, regression YAML, plugin-example build files; tree-level GPLv2 only). `lammpsplugin.h` and `library.h` carry the plain LAMMPS GPL notice (not LGPL). Decision: ADR-010 (GPL-2.0-only, approved).

**External (non-pinned, secondary) sources used for the EEM discussion:** AMS ReaxFF documentation (scm.com) — *page fetch blocked by the sandbox proxy; only web-search summaries seen*; LAMMPS manual pages in the pinned tree; a LAMMPS-forum thread (secondhand). None is treated as specification.

## M1 additions — reference harness and instrumentation (this repository's own code; upstream files only *patched*)
| Item | Role |
|---|---|
| `third_party/lammps/patches/0001-reaxmetal-diagnostics.patch` (+ `PATCHES.sha256`) | adds `src/REAXFF/reaxff_diag.h` and observation hooks in `reaxff_{ffield,bond_orders,bonds,multi_body,valence_angles,torsion_angles,hydrogen_bonds,nonbonded,forces}.cpp`, `fix_qeq_reaxff.cpp`; hunks keep the upstream (Sandia / PuReMD) notices |
| `third_party/lammps/patches/experiments/EXP-0001-hbond-image-identity.patch` | `orig_id[i] != orig_id[k]` → `i != k` (confirms Q-32) |
| `third_party/lammps/patches/experiments/EXP-0002-ovun-dDelta_lp_temp.patch` | `dDelta_lp[j]` → `dDelta_lp_temp[j]` in the over/under force loop (confirms Q-34) |
| `tools/build_lammps_instrumented.sh`, `tools/build_reference_matrix.sh` | patched / multi-compiler builds (gcc, clang, FMA, -O0, MPI, Kokkos-serial) |
| `tools/reaxref/runner.py` | case → LAMMPS input → `result.json` (14 slots, q, F, diagnostics, provenance, validity rules, independent EEM residual) |
| `tools/reaxref/{geometries,fixtures}.py`, `tests/fixtures/` | independently constructed geometries; 58 cases; force-field manifest (names + SHA-256) |
| `tools/reaxref/ref_nonbonded.py`, `check_nonbonded.py` | independent explicit-image vdW / Coulomb / E_pol reference and pair-count check |
| `tools/reaxref/{conditioning,noise_floor,make_tolerances,compare_builds}.py` | conditioning classifier, noise floor, frozen tolerances, bitwise comparison |
| `tools/reaxref/exp_*.py`, `fd_check.py` | periodic-image, ghost-range, Q-09/Q-12/Q-32/Q-34 and finite-difference experiments |
| `tools/reaxref/m1_gate.py`, `tools/m1_reproduce.sh` | acceptance gate and end-to-end reproduction |

## M2 additions — parser, settings, adapter A1 (this repository's own code; upstream semantics re-implemented, no upstream source copied)
| Item | Mirrors (pinned LAMMPS) | Role |
|---|---|---|
| `src/io/text_reader.hpp`, `src/io/ffield_parser.cpp`, `include/reaxmetal/forcefield.hpp` | `src/REAXFF/reaxff_ffield.cpp` (+ `TextFileReader`, `ValueTokenizer` semantics) | ffield reader, tables, compat flags, strict-mode checks |
| `src/io/control_parser.cpp` | `src/REAXFF/reaxff_control.cpp` | control file |
| `src/io/pair_settings.cpp`, `include/reaxmetal/pair_settings.hpp` | `PairReaxFF::settings` (`pair_reaxff.cpp`) | `pair_style` keywords; capability gating |
| `src/io/table_dump.cpp`, `tools/ffield_dump.cpp`, `tools/reaxref/canonical_tables.py` | patch 0001 `params.txt` | canonical lossless table dump on both sides (PARSE-1) |
| `src/core/sha256.cpp` | — | FIPS 180-4 SHA-256 for table hashes |
| `tools/reaxref/parse_tables_check.py`, `parse_diff.py` | — | PARSE-1 and the differential mutation fuzz (PARSE-3) |
| `plugin/adapter/*` | `PairReaxFF::{settings,coeff,init_style,init_one,extract}` | adapter A1 (no physics); `lammps_fmt_abi.h` see LAMMPS_INTEGRATION addenda |
| `tests/lammps/run_a1.py` | — | adapter differential test through the LAMMPS C library |

## M3 additions — geometry, neighbor lists, Metal layer (this repository's own code)
| Item | Mirrors (pinned LAMMPS) | Role |
|---|---|---|
| `include/reaxmetal/system.hpp`, `src/core/system.cpp` | `Domain` box description | `Box` (general/triclinic), `AtomSet` (owned+ghost with owner and lattice shift) |
| `include/reaxmetal/neighbor.hpp`, `src/neighbor/neighbor.cpp` | `Comm::borders` (ghost slab rule); `NPairBinGhost<HALF>` + `write_reax_lists` (`pair_reaxff.cpp:629-680`); `vdW_Coulomb_Energy` ownership rules (`reaxff_nonbonded.cpp:104-117`) | image expander, cell grid, far list, pair-class counting, device-list input, row verification, fixed-order reductions |
| `src/metal/shaders/reaxmetal_m3.metal`, `reaxmetal_m3_types.h` | — | MSL kernels (bring-up, `rm_far_rows`, reductions); runtime-compiled |
| `src/metal/metal_backend.mm`, `metal_backend_stub.cpp`, `metal_source.cpp`, `include/reaxmetal/metal_backend.hpp` | — | Objective-C++ host (Apple) / explicit "unavailable" stub / embedded shader text |
| `tests/metal_shim/`, `tests/objc_stub/`, `tests/check_objcxx_syntax.sh` | — | CPU emulation of the kernels; stub headers for a syntax-only check of the Objective-C++ |
| `tools/neighbor_tool.cpp`, `tools/m3_systems.hpp`, `tools/metal_check.cpp`, `tools/mac/*.sh` | — | CLI for fixture comparisons; shared geometries; on-device checks and the Apple-machine scripts |
| `plugin/adapter/*` (A2 additions) | `PairReaxFF::init_style` (`pair_reaxff.cpp:360-376`) | ghost-native host view, self-check |
| `tests/lammps/run_ghosts.py`, `run_a2.py`, `tests/python/test_neighbor_fixtures.py` | M1 reference tallies; stock LAMMPS ghosts | NBR-2, INT-7, NBR-3 |
