# Development Log

Append-only. Each entry: what was done, what was actually run and its result, what went wrong, decisions taken.
Never edit a past result; add a correction entry instead.

---
## 2026-10-08 — M0: source audit, specification, decisions, skeleton

**Scope executed:** M0 only. M1 not started.

### Inputs
* Mission statement (milestones M0–M8, architectural rules 1–10).
* Paper 1: Zheng, Li, Guo (GMD-Reax), J. Mol. Graph. Model. 41 (2013) 1–11 — read pages 1–11 (all).
* Paper 2: Kylasa, Aktulga, Grama (PuReMD-GPU), Purdue CS TR 13-005 — read pages 1–26 of 30.
  **Pages 27–30 were not read** (accuracy tables, conclusion, references). To be read before M8.
* LAMMPS `stable_30Sep2026` (commit `8de817dd…`) obtained by `git clone --depth 1 --filter=blob:none --sparse`
  into the session scratchpad (outside the repository). Direct `https://github.com/...` web fetches returned HTTP 403
  through the sandbox proxy; `git` over HTTPS worked.

### Audit performed (read in full unless noted)
`reaxff_types.h, reaxff_defs.h, reaxff_inline.h, reaxff_ffield.cpp, reaxff_control.cpp, pair_reaxff.cpp, reaxff_forces.cpp,
reaxff_bond_orders.cpp, reaxff_bonds.cpp, reaxff_multi_body.cpp, reaxff_valence_angles.cpp (from line 28),
reaxff_torsion_angles.cpp (from 28), reaxff_hydrogen_bonds.cpp (from 28), reaxff_nonbonded.cpp (from 28),
reaxff_init_md.cpp, reaxff_reset_tools.cpp, fix_qeq_reaxff.cpp (from 28)`; license headers of the whole `src/REAXFF` set
by grep. **Read only partially (searched, not read through):** `src/KOKKOS/pair_reaxff_kokkos.cpp` (4 415 lines) and
`fix_qeq_reaxff_kokkos.cpp` — structure, error checks, atomics/ScatterView use, precision types, constants, and the
differences listed in ENGINE_SPEC Q-02/Q-17 were located by grep and excerpted; the Kokkos kernels were **not** reviewed line by line.
**Not read:** `reaxff_lookup.cpp` (tabulation; only the spline function names checked), `reaxff_allocate/list/tool_box.cpp`
(memory heuristics), `fix_acks2/qtpie/qeq_rel`, species/bonds analysis (all out of scope / rejected).

### Findings that changed the plan (details in ENGINE_SPEC §10, NUMERICAL_POLICY §4, ADRs)
1. The reference contains element logic inside kernels (`strcmp(name,"C")`, masses 12.0000/15.9990, `mass>21`) → ADR-003 turns it into load-time flags.
2. Constants are truncated and mutually inconsistent (`14.4·23.02 ≠ 332.06371`, 0.17 %) → charges are not an exact stationary point of the reported energy → two FD modes (frozen/relaxed).
3. LAMMPS unit tests use a 7.54 Å cell with a 10 Å cutoff → minimum image is wrong; explicit images are mandatory (ADR-004). GMD-Reax used minimum image.
4. FP32 hazards verified by arithmetic (`tools/fp32_hazards.py`): `exp(−p_boc1·Δ')` overflows FP32 for Δ' < −1.774 (p_boc1 = 50) — reachable by any weakly bonded atom of valence ≥ 2.
5. LAMMPS does not validate FP32 ReaxFF: `single` Kokkos tests skipped; `mixed` run at ε×2e9 (relative tolerance 2.0). No upstream FP32 accuracy reference exists.
6. The reference zero-fills missing 2-body parameters (derived by reading: phantom full bonds) and overruns a fixed array for >5 three-body sets → engine rejects (D-2, D-3). **Not yet confirmed by running LAMMPS.**
7. LAMMPS exposes only 14 summed energy slots and total forces; per-term attribution needs an instrumented build (open question).
8. Deterministic force assembly is a scatter problem for all multi-body terms, not just torsion (ADR-006).
9. Parameter-name trap: C++ `valency_boc`/`valency_val` are swapped relative to the ffield header labels; gp[14] labelled `p(val7)` but used as `p_val6`.

### Mistakes made and corrected during M0 (recorded so they are not repeated)
* First sparse checkout failed (`LICENSE` is a file; cone mode accepts directories only) — fixed, no impact.
* My ffield survey script initially indexed `p_hbond` and the core-vdW columns wrongly (reported every element as H-bonding and
  wrong vdW types). Caught by sanity-checking the output against the file, fixed, re-run; only the corrected output is used in the docs.
* ENGINE_SPEC draft claimed the `lgvdw` term was independent of the vdW type; the code nests it inside the inner-wall branch — corrected.
* NUMERICAL_POLICY draft stated "overflows by 10^20.5" for `exp(200)` in FP32; correct factor is ≈2·10^48 — corrected.
* Q-25 initially said LAMMPS skips FP32 ReaxFF tests; correct statement: it skips `single`, tests `mixed` at a vacuous tolerance — corrected.
* Torsion/valence scatter description in ADR-006 was first written too narrowly — generalised.
* `test_capabilities` failed at first build: `find_by_lammps_construct("-")` matched the first row without a LAMMPS spelling.
  Fixed in the library (`"-"`/empty never matches) and the test now states the rule precisely (only engine-internal `backend.*` rows may lack a construct).

### Decisions taken (owner may override; see ADRs)
Pin = latest stable tag (ADR-001); element knowledge as load-time flags (ADR-003, *proposed*); explicit images (ADR-004);
shared term functions (ADR-005, *proposed*); license policy GPL-2.0-only provisional, no `LICENSE` file yet (ADR-010, *proposed*).

### Verification (actual runs)
Environment: Linux 6.18 x86-64 VM (4 cores), CMake 3.28.3, g++ 13.3.0, clang++ 18.1.3, Python 3.13. **No Apple platform.**
* Clean configure + build, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`: g++ OK, 0 warnings; clang++ OK, 0 warnings.
* `ctest` on both compilers: **4/4 passed** (`pins`, `capabilities`, `energy_terms`, `docs_sync`), 0.02 s total each.
* **Mutation check** (scratch copy, not committed): 9 deliberate corruptions — FEATURE_MATRIX status changed, milestone changed,
  row deleted; pin commit changed in the header; pin tag changed in `PIN.txt`; a pvector slot stripped of its term; over+under
  merge broken; `opt.tabulate` Rejected→Ignored; `qeq.acks2` Rejected→Planned — were each caught by the intended test;
  the unmutated control passed 4/4. (Evidence that the tests can fail, not that physics is right.)
* `tools/fetch_lammps.sh <fresh dir>`: `OK: stable_30Sep2026 @ 8de817dd79bfe4525d5d39246a212d833e6dee07 verified; 72 files match manifest` (11 s).
  It refuses a non-empty destination (exit 2).
* `cmake -DREAXMETAL_ENABLE_METAL=ON` on Linux: configure error "requires an Apple platform" (by design).
* `tools/fp32_hazards.py` and `tools/survey_ffield.py` run from the repository tree; their outputs are the ones quoted in the docs.
* **Not run:** anything on a GPU; anything that computes physics.

### Not done / not claimed
* No physics, no parser, no neighbor list, no QEq, no Metal code. No fixture exists. No Metal-capable machine was available.
* Claims marked "by reading" in ENGINE_SPEC (notably Q-09, Q-12) have not been confirmed by executing LAMMPS.
* The Kokkos path was not audited line by line.

---
## 2026-10-08 — M0.5: LAMMPS integration architecture (after the owner's M0 review)

**Scope executed:** M0.5 only; M1 not started. M0 work preserved (nothing discarded).

### Owner decisions received (summary)
1 license GPL-2.0-only approved, complete the per-file audit; 2 element/mass behaviour → per-type compatibility flags with regression tests; 3 target = Apple M5 Max (CLT only, no Xcode/`metal` compiler); 4 instrumented LAMMPS approved (M1) incl.
experimental Q-09/Q-12; 5 FP32 policy modified (twin not the only criterion; independent tolerances before Metal; stable formulations, derivative-tested); 6 strict charge convergence by default; 7 pin approved; 8 scope modified
(standard EEM/QEq and LAMMPS-compatible shielded QEq **not** rejected; defer ACKS2/QTPIE/efield/alt models/tabulation; NPT after virial). **Architectural change:** LAMMPS integration is primary (`reaxff/metal` plugin; no standalone MD engine).

### What was done
* Audited the plugin mechanism and the `Pair`/`Force`/`Neighbor`/`Comm`/`Verlet` contracts in the pinned tree, and the charge fixes (`qeq/reaxff`, `qeq/shielded`, plus access/extract coupling of `qtpie`, `acks2`, analysis fixes).
* Built the **stock pinned LAMMPS** (serial, shared, REAXFF+QEQ+PLUGIN) with the new reusable `tools/build_lammps_reference.sh` (refuses a non-pristine tree; ≈3 min on 4 cores).
* Wrote and ran a **`Pair`-derived probe plugin `reaxff/metal`** (no physics; `plugin/probe/`) and a stdlib-only harness (`tests/lammps/`), then the same through CTest (`lammps_probe`, opt-in).
* License audit (`tools/audit_licenses.py` → `LICENSE_AUDIT.tsv`, 130 files), `LICENSE`, `LICENSES/`, `REUSE.toml`, SPDX everywhere, `license_headers` test.
* `compat_flags.hpp/.cpp` + `test_compat_flags.cpp` (exact upstream predicates, boundaries/exceptions); capability table moved to the owner's scope (`Deferred` status; `lammps.*`, `eem.*` rows).
* `tools/eem_dense_check.py` (independent dense explicit-image EEM), `tools/stab_f1_check.py` (stable overcoordination factor, values only).
* Documents: new `LAMMPS_INTEGRATION.md`; updated ENGINE_SPEC (ghost-native §3, EEM §7.1–7.4, Q-26…Q-31, D-1/D-5), ARCHITECTURE_DECISIONS (approvals, ADR-004 amendment, ADR-013/014/015, re-planned milestones), NUMERICAL_POLICY (§4.4, §5), FEATURE_MATRIX, SOURCE_MAP (§7), VALIDATION, README.

### Findings that changed the plan
1. **A `Pair`-derived style works with the stock charge fixes** (bit-identical charges vs stock `reaxff`; `qeq/shielded` agrees to 3.3e-15) — no need to derive from `PairReaxFF`.
2. **LAMMPS gives the host everything:** ghosts are owner + lattice shift on every step; ghost forces fold back exactly; the topology (order, `nghost`) changes only when `neighbor->ago == 0`, including under `atom_modify sort`; a pair neighbor request is optional → device-built lists are legitimate. This made the **ghost-native** engine the right pivot (ADR-013); the explicit-image-shift mechanism of ADR-004 was dropped (its "never minimum-image" rule stands: minimum-image is wrong by 0.06–7.8 e here).
3. The documented `fix qeq/reaxff` small-cell restriction is **not reproduced** in 4 cubic cases (dense explicit-image EEM agrees to ≤2.7e-14). Open for triclinic/degenerate cases.
4. **Strict EEM cannot be obtained from the stock fix** (no status, protected state, no `extract`) → derived-fix plan (ADR-015).
5. The FP32 reference form of the overcoordination factor is **non-finite for 30 % of the physical Δ′ domain**; the stable form is finite everywhere and equal in FP64 to 4.4e-16 (values only).
6. Plugin interface headers are plain-LAMMPS-GPL (not LGPL) in this tree; 35 audited files carry no per-file notice (ffields, YAML, example build files).
7. The owner's Mac has **no Xcode / `metal` compiler** → runtime shader compilation is the baseline (unverified).
8. AMS EEM details could not be retrieved (`scm.com` blocked); only search summaries were seen → no equivalence claimed.

### Mistakes made and corrected during M0.5
* I first wrote that the plugin loader enforces version equality; the source shows it only logs a mismatch (`plugin.cpp:179-182`). Corrected; ABI/symbol compatibility is the real constraint.
* Quoted "≈7 min" for the stock build; file timestamps show ≈3 min. Corrected.
* My one-line `qtpie`/`qeq/rel` probe used the wrong argument count, so those fixes were **not** tested; their rows are [U], not [V].
* A mutation "survivor" (`mass pair: only one order`) was a **no-op sed** (the `||` sat at a line break); re-run properly it was caught (12/12).
* Regenerating `SOURCE_HASHES.sha256` truncated the file before `cat` read it (72 → 70 entries). Caught by the count; rebuilt from git as the union (142 entries), all 72 originals preserved and all hashes verified against the full tree.
* The M0 documents called ghost handling a deliberate *deviation* (D-1) — superseded: it is now a replication (R).
* A documentation-edit script aborted on a text-pattern mismatch (ADR-004 wording) before writing anything — no partial edits; the pattern was corrected and the script re-run.

### Verification (actual runs, Linux x86-64, g++ 13.3 / clang++ 18.1)
* Default build: **6/6 CTest pass** on both compilers, 0 warnings (`pins`, `capabilities`, `energy_terms`, `docs_sync`, `license_headers`, `compat_flags`).
* Plugin-enabled build: **7/7** including `lammps_probe` (≈10 s). Probe summary: failures=0.
* Mutation checks: compat flags 12/12 caught; license/provenance 3/3; `lammps_probe` detects a plugin that skips ghost forces (fold-back FAIL → CTest red).
* `tools/fetch_lammps.sh`: sparse and `--full`, 142/142 hashes. `tools/build_lammps_reference.sh`: OK on a pristine pinned tree.
* **Not run:** anything on macOS/Metal; any multi-rank run (serial LAMMPS only — the `nprocs` check is unexecuted); `qtpie`, `qeq/rel`; real adapter physics; derivative tests of stable forms; AMS comparison.

### Awaiting owner confirmation (proposals, not decisions)
ADR-005 (shared term functions), ADR-015 (derived strict fix), the numeric acceptance criteria of NUMERICAL_POLICY §5.2 (frozen at M1 start), copyright-holder wording in `REUSE.toml`, whether ffields may be vendored, whether AMS reference charges can be provided.



## M1 — LAMMPS reference oracle, fixtures, periodic-image accounting (2026-10-08)

### Owner decisions applied at the start of M1
ADR-005 (shared pure term functions, backend-specific traversal/reduction; equivalence test if sharing is unsafe) and ADR-015 (derived strict `qeq/reaxff/metal` at M5; M1 records QEq settings/diagnostics and **excludes unconverged calculations from the golden set**) approved; FP32 acceptance revised (C1 floor + strict parity, C2 characterisation only, C3 owner values on well-conditioned fixtures, originals kept as stretch); copyright = `SPDX-FileCopyrightText: 2026 Anirban Phukan` for original files with upstream notices preserved (ADR-018); hybrid fixture policy (ADR-017); cloud-first / Mac-later with status taxonomy (ADR-016).

### What was built
Hashed observation-only LAMMPS diagnostics patch + patch-aware builder (ADR-019); build matrix (gcc, clang, FMA-contracting, -O0, -O2/nofma, OpenMPI, Kokkos-Serial); reference runner with validity rules and an independent EEM residual; independent explicit-image non-bonded reference; 58 independently constructed fixtures (11 force fields, 10 elements); conditioning classifier; noise-floor, equivalence, periodic, ghost, Q-09/Q-12/Q-32/Q-34, finite-difference experiments; frozen tolerances; 58 golden references; CTest `tolerances_frozen`, `fixtures_frozen`, `patch_frozen`, `reaxref_selftest` (14 unit tests); acceptance gate; `tools/m1_reproduce.sh`.

### Findings that change the plan (details: ENGINE_SPEC Q-32…Q-37, VALIDATION §6)
1. **Image accounting is right** in pinned LAMMPS (pair counts, bonded multi-body across boundaries, QEq images, down to one atom per 1.30 Å) — "sum over all owned+ghost pairs" is *not* the rule; the engine replicates owner-computes + tag ordering (ENGINE_SPEC §3.1).
2. **Three reference defects/hazards to reproduce or reject:** Q-32 (H-bond tag comparison drops self-donor images), Q-34 (analytic force ≠ gradient for heavy atoms with π bonds), Q-35 (QEq taper beyond the ghost shell silently truncated); plus Q-33 (near-linear ill-conditioning) and Q-37 (energy jump at `thb_cutsq`).
3. Q-12 was mis-stated in M0 (π=ππ=1 for every pair): measured BO′ = (1,0,0) for H pairs, (1,1,1) for C–O/C–C.
4. Q-27 "small-cell QEq restriction" is again not reproduced (triclinic, 2-D, 1-atom cells included).
5. The pre-registered well-conditioned definition is not sufficient: exact-collinear fixtures give 1.5e-2 force noise between FMA and non-FMA builds → amendment proposed (ADR-020).

### Gate deviation — please judge
My gate G1 said "instrumented bitwise-equals stock on all fixtures". It holds for the strict-IEEE builds (gcc, clang: 58/58, diagnostics on and off) but **not** for the two FMA-contracting builds (11–12 fixtures, ≤1.3e-13 in forces). I split G1 into G1a (required, strict-IEEE) and G1b (reported) *after* seeing this result. The instrumented oracle is the strict-IEEE build; FMA builds are used only as stock noise-floor probes.

### Mistakes made and corrected during M1
* Single-element crystals (diamond, Au, Fe, graphene) have q ≈ 1e-15 by symmetry, so my first "periodic QEq" and ghost tests on them were vacuous; charged systems (water boxes, CH₄, polyethylene) were added and the supercell test compares across runs, not only across replicas.
* The first Q-12 run failed on a directory name with a space; the first nonbonded/param checks used the LAMMPS type index instead of the force-field index (wrong for non-CHO orderings, caught by the FC fixtures); the first conditioning rule flagged saturated sp³ carbon as an `nlp` kink (only non-zero integer Δe/2 are kinks) and every large crystal for a pair distance at `bond_cut` (a distance edge counts only where it gates an interaction).
* A tag-only FC fixture claimed Q-34; the FC force field does not trigger it (FD 1e-7) — retagged as a control.
* An MPICH-based "multi-rank" run silently launched singleton ranks and printed identical results; discovered from the processor-grid line, discarded, redone with OpenMPI.
* I edited a shell script while bash was executing it (spurious `=on: command not found`), and a driver step used a repo-relative path after `git -C` (experiment builds skipped); both fixed and the driver re-run.
* A first Kokkos/MPI noise comparison included a case LAMMPS itself rejects for np≥2 (sub-domain < skin); such cases are now excluded for all builds and listed.

### Verification (actual runs)
Gate (`tools/reaxref/m1_gate.py`) — see VALIDATION §6. **CTest: 10/10 pass with g++ 13.3 and clang++ 18.1, 0 warnings.** Mutation check of the freeze: altering one tolerance value makes `tolerances_frozen` fail; altering a fixture makes `fixtures_frozen` fail (both restored). Reproduction driver run warm end-to-end (experiment builds included); a cold full rebuild of all 12 LAMMPS variants by the driver was **not** repeated from scratch (the early variants were built with the same scripts and flags manually).

### Not run
macOS/Metal anything; engine physics (none exists); AMS comparison; Kokkos OpenMP/GPU; fixtures >216 atoms for the floor; long-bond force fields for the ghost-shell rule; `qtpie`, `qeq/rel`.

### Awaiting owner confirmation
(1) the G1 split above; (2) conditioning amendment X1/X2 (ADR-020); (3) C3 neutrality/equalization-residual thresholds `2e-5 e/atom`, `2e-3 eV` (proposed in the frozen file; not derived from FP64 noise); (4) ADR-021: reproduce Q-32/Q-34 by default (and whether to report them upstream); (5) ADR-005/015 follow-through at M4/M5 as approved.

## M2 — ffield/control parser, tables with compat flags, adapter A1 (2026-10-08)

### Instruction and how I read it
"M1 approved, proceed to M2 only." I treated this as accepting M1 **including the four proposals I had flagged**: (1) the G1 split into G1a (strict-IEEE, bitwise, required) and G1b (FMA builds, reported); (2) the conditioning amendment X1/X2 (ADR-020); (3) the proposed C3 neutrality/equalization-residual thresholds (2e-5 e/atom, 2e-3 eV); (4) ADR-021, reproduce Q-32/Q-34 by default behind compat flags. **This is an interpretation; if you meant only the gate result, say so and I will reopen those items.** M3 was not started.

### What was built
* `reaxmetal::ForceField` and `parse_force_field` / `read_force_field_file` (`src/io/ffield_parser.cpp`, `include/reaxmetal/forcefield.hpp`): token-for-token re-implementation of the pinned reader's semantics (ENGINE_SPEC §2.6), tables as `SingleBody/TwoBody/ThreeBodySet/FourBody/HBondParams`, per-type and per-pair compat flags computed once. Strict by default (Q-09 overrun, non-finite numbers, truncation, <38 general parameters); Q-12 per used element pair via `require_bond_blocks`.
* Control-file parser, `pair_style` settings parser with capability gating (`tabulate` etc. raise `UnsupportedFeatureError`), SHA-256, canonical lossless table dump (C++ and Python sides), `reaxmetal_ffield_dump` tool.
* Adapter A1 (`plugin/adapter`): `reaxff/metal` with parse, `pair_coeff` element mapping, `extract(chi|eta|gamma)`, host checks, explicit refusal in `compute()`.
* Tests: `ffield_parser`, `pair_settings` (unit), `ffield_tables` (PARSE-1 hashes; opt-in via `REAXMETAL_FFIELD_DIR`), `lammps_a1` (opt-in via `REAXMETAL_LAMMPS_PREFIX`), `tools/reaxref/parse_diff.py` (not in CTest: needs the instrumented LAMMPS build).

### Results (actual runs; VALIDATION PARSE-1/2/3, EEM-3, LINT-1, INT-3, INT-6)
* PARSE-1: 11/11 bundled force fields give tables bit-identical to those LAMMPS stored (full and portable dumps).
* PARSE-3: 1200 differential mutants (3 seeds), 0 problems; class counts in VALIDATION.
* EEM-3: `extract()` bit-identical to stock for 56 `(ffield, type map)` combinations; 11 rejected by both; 10 rejected by us only for Q-12 (justified, counted separately); 0 mismatches.
* Host checks, explicit `compute()` refusal and the 2-rank refusal (OpenMPI) pass.
* **CTest: 15/15 pass with g++ 13.3 and clang++ 18.1, 0 warnings** (fresh build trees, with the two opt-in LAMMPS tests enabled).

### Mistakes made and corrected during M2
* First C++ build: `parse_force_field` could not reach private members (friend declaration missing). Synthetic `lgvdw` test file lacked the extra off-diagonal column. A six-set three-body test wrote six lines under a count of five. All fixed.
* **A parser mutation survived**: rounding instead of truncating `p_hbond` (bundled values are integers). A dedicated truncation test now catches it; all 5 parser mutations tried are caught.
* The fuzz driver first reported 6 BAD-ACCEPT: LAMMPS' "Non-existent ReaxFF type" error was hidden by `-screen none`; the driver now captures stdout. Its `crlf` operator returned a bare string (unpack error).
* The ctypes harness sent multi-line strings to `lammps_command` (does nothing), loaded `liblammps` without `RTLD_GLOBAL` (plugin symbols unresolved → "Unrecognized pair style"), and two of my own host-check expectations were wrong (qtpie needs a Gaussian-exponent file; the control-file message text). All harness bugs, not adapter bugs.
* **The harness first accepted a broken adapter**: dropping the Q-12 check left it passing (it counted the 10 combinations as parity). The harness now fails when ours accepts an element pair without a bond block; the mutation is caught (10 failures). Halving `eta` in `extract()` is caught (56 failures).
* The clang++ build of the plugin failed to load (fmt/`std::format` ABI split, LAMMPS_INTEGRATION M2 addenda) and `plugin load` hides that. Found only because I built with both compilers; fixed with `lammps_fmt_abi.h`, and the harness now fails with a clear message. Also: clang `-Wsign-conversion` in `sha256.cpp`, and the capability-table convention (Implemented rows use milestone `-`) caught by the `capabilities` test.

### Not run
macOS/Metal anything (the plugin has not been built with Apple clang or loaded in a macOS LAMMPS); engine physics (none); `acks2`, `qeq/rel`, efield and non-neutral-group refusals; vdw-conflict / lg-on-type-1 diagnostics (not implemented); the parser mutation check against the PARSE-1 hashes beyond the 5 unit mutations; multi-rank beyond np=2; LAMMPS builds with C++20/`std::format` (`REAXMETAL_LAMMPS_STD_FORMAT=ON` untested).

### Awaiting owner confirmation
(a) the reading of "M1 approved" above; (b) ADR-022: Q-12 enforced per used element pair rather than per file; (c) whether the adapter should keep the real style name `reaxff/metal` while `compute()` refuses (my choice) or stay unregistered until M4.

## M3 — geometry/neighbor core, Metal layer, adapter A2 (2026-10-08)

### Instruction
"Yes M3 next. Also b) Metal code alongside the CPU path in M3. You would run a short test script on the Mac and send me the output after each step. — write the metal code". So: CPU path validated here; Metal code written here and *never run*; three Apple-machine steps prepared (`tools/mac`, `docs/MAC_VALIDATION.md`). Status per ADR-016: **CPU: validated on Linux. Metal: written (and emulated/static-checked); not compiled, not executed.**

### What was built
* CPU: `Box`/`AtomSet`, image expander (triclinic, mixed periodicity), cell grid, far neighbor list with the reference's row cutoffs, owner-computes pair counting, device-list input with the float margin contract, row verification, grow-and-retry, fixed-order reductions (`include/reaxmetal/{system,neighbor}.hpp`).
* Metal: MSL kernels (`saxpy`, `math_probe`, `rm_far_rows`, `rm_partial_sums`, `rm_sum_partials`), shared parameter header, Objective-C++ host with run-time shader compilation, plain-C++ interface and a "not available" stub, on-device check tool and step scripts, CPU emulation shim for the kernels, fake backend for testing the check tool, ObjC++ stub headers for a syntax check.
* Adapter A2: ghost-native host view of LAMMPS' own arrays, strict ghost-shell check, `reaxmetal_selfcheck` row-by-row comparison with LAMMPS' list.

### Results (actual runs)
* NBR-1 CPU: cell-grid list bitwise equal to brute force on 4 box types × 3 cutoff sets. NBR-2: ghost sets equal LAMMPS' for 58/58 fixtures (32 313 ghosts). NBR-3: pair classes equal the M1 tallies for 58/58. INT-7: inside LAMMPS the far list equals LAMMPS' list row by row for 58/58 fixtures; ghost = owner + shift verified on every ghost.
* Kernels by emulation: 7 shared geometries (up to 20 197 atoms, 2.6 M entries), 0 missing / 0 illegal; reductions bitwise equal to the CPU twin. **This is emulation, not Metal.**
* **CTest: 24/24 pass with g++ 13.3 and with clang++ 18.1, 0 warnings** (fresh build trees, all opt-in LAMMPS tests enabled); the default configuration (no LAMMPS tree) passes its 18 tests.
* Mutation checks: neighbor core (tag order, tie-break sign, shell width, strict `<`, ghost-row cutoff) — all caught after adding a cutoff-boundary unit test (the first set missed `<` vs `<=` in the unit test and missed orientation flips in the fixture test); 6 kernel mutations all caught; NBR-2 shell mutation caught; A2 ghost perturbation caught.

### Decisions I took that you should confirm
1. **ADR-023: no metal-cpp.** It is Apache-2.0 (I downloaded and read the license); the FSF treats Apache-2.0 as incompatible with GPL-2.0-only. The host layer is Objective-C++ against the system frameworks. This overrides the ADR-008 wording.
2. **ADR-024: binning stays on the host** (device receives cell ids and CSR cells); only the per-atom row search and reductions run on the GPU in M3. This narrows "device cell lists" in the milestone table.
3. Canonical reduction order is chunked-sequential (slow but bit-identical to a CPU twin).

### Mistakes made and corrected during M3
* **Wrong test expectation in my own Mac tool**: the "order-sensitive vector" gave the same sum in canonical and left-to-right order (5 = 5), so step 3 would have reported a failure on your machine for a non-bug. Found only because I built a fake backend and ran the tool end to end; vector replaced and its premise is now a unit test.
* The first fixture driver wrote numpy `repr` (`np.float64(...)`) into the case files, and omitted `--lgvdw` for the lg fixtures. The shim namespace `metal` clashed with my own `reaxmetal::metal` (renamed `mtl`). My shader lint matched comments and `std::uint32_t` in the shared header (comments are now stripped and the header avoids `std::`). A `sed` edit of the test dropped a function once (restored).
* **A2 checks its own row-cutoff function against itself**: mutating `row_cut` (ghost rows = `nonb_cut`) is not caught by `lammps_a2`; the cutoffs are covered by a unit test and source reading only until M4 bond lists give an end-to-end check. Recorded in VALIDATION INT-7.
* 3 fixtures (`ab_ammonia_borane`, `lg_*`) are rejected by the strict Q-12 rule when the type map contains an element that has no atoms; the A2 harness retries with only the elements that have atoms. This is a (conservative) consequence of ADR-022 that users will meet: it is per element *type*, not per element actually present in the data.
* Capability table: Implemented rows must use milestone `-` (the `capabilities` test caught it again after I forgot).

### Not run
**Everything Metal**: the Objective-C++ has never been compiled with the Apple SDK; no shader has been seen by the Metal compiler; no GPU kernel has run; INT-5 (plugin on macOS); NBR-1/FORCE-2 Metal parts; GPU timing. Also not run: shrink-wrapped/`m` boundaries; MPI beyond the existing refusal; a LAMMPS built as C++20 (`REAXMETAL_LAMMPS_STD_FORMAT=ON`).

### What I need from you
Run `tools/mac/step1_bringup.sh` first and send `mac-reports/step1-*.txt`. Then steps 2 and 3 (details in `docs/MAC_VALIDATION.md`). I will not claim any Metal result before that, and step 4 (macOS plugin build, INT-5) will be prepared after the Metal layer builds. Also decide: ADR-023 (no metal-cpp), the narrowed host-binning scope (ADR-024), and the three pending items from M2 (ADR-022 Q-12 per used pair; real style name while `compute()` refuses).

## M3 validation on the M5 Max, step 4 (INT-5) and the physics (M4–M6) — 2026-10-08 (Claude Code on the Mac)

### Instruction
"START WORKING ON IT - MAKE THE PHYSICS RUN PROPERLY WITH METAL", plan approved: ADR-013 order (M4 → M5 → M6), stay inside the frozen rules (ADR-021 quirks reproduced, frozen tolerances, no FP atomics in the validated path, no tolerance edits).

### What was run before any new code
`tools/mac/step1..3` all PASS on the M5 Max (MET-1, NBR-1, FORCE-2, MET-4). Step 4 (INT-5): the pinned LAMMPS built unmodified (serial, shared, Apple clang 17), the plugin built and loaded; probe, A1, A2 pass. No code change was needed for any of it.

### What was built
* `include/reaxmetal/terms.hpp` (dual-use: host C++ via `terms_host.hpp`, and MSL): pure term functions templated on the scalar — bond order, bond-order correction with derivative coefficients, atom quantities, bond energy, triple-bond stabilisation, lone pair, C2, over/under, taper (reference Horner form on the host, scaled form `1−35x⁴+84x⁵−70x⁶+20x⁷` for float), vdW/Coulomb/polarization pair terms. `safe_exp` caps float exponent arguments at 40 (double: 700).
* `src/cpu/bonded.cpp`, `src/cpu/nonbonded.cpp`: the CPU-64 engine, following the pinned traversal (bond_mark, tag-order rules, three-body lists, H-bond lists, `Add_dBond_to_Forces`): all 13 energy terms and the gradient.
* `plugin/adapter`: `compute()` now evaluates the force field (energies, `pvector`, forces, virial by fdotr); keyword `backend cpu64|metal`; per-atom energy/virial requests are refused.
* Metal: `rm_nb_pairs/rm_nb_gather` (nonbonded) and the `rm_b_*` / `rm_h_build` kernels (bond list, bond orders, bond/lp/over/under, valence/penalty/coalition, torsion/conjugation, H-bond, force assembly), `Context::nonbonded/bonded`, host orchestration with grow-and-retry for bond/H-bond capacity (`bonded_device.cpp`), and the same orchestration run against a CPU emulation backend in `test_metal_emulation`.
* Tests: `bonded_fixtures` (BOND-1), `nonbonded_fixtures` (NB-1), `full_fixtures` (FULL-1), `gpu_fixtures` (GPU-1), `lammps_int2`, `lammps_int2_metal` (INT-2), emulation of the new kernels. CTest 31/31 on the M5 Max (Apple clang, Metal ON).

### Results (actual runs; see VALIDATION)
* CPU-64 vs pinned LAMMPS: 13 slots 3.5e-15 relative, forces 3.4e-12 (limits 2e-9 / 1e-9); FD reproduces the documented reference behaviours (Q-34 16.500, Q-33 0.009/0.013) exactly.
* In-LAMMPS A/B (INT-2): cpu64 within frozen C1 on all 58 fixtures; **metal within the owner-set C3** on all 58 (PE ≤ 9.7e-5 kcal/mol/atom, forces 4.0e-3 max / 9.1e-4 RMS). A 24-atom periodic water box, 200 NVE steps with `fix qeq/reaxff` and neighbor rebuilds: cpu64 identical to stock to 10 digits; metal agrees to ~6 digits and conserves energy like stock (−1898.5595 → −1898.5601 for both).
* Metal runs are deterministic (two launches bitwise identical on every fixture).

### Mistakes made and corrected
* The first fixture test mapped LAMMPS types to force-field indices wrongly (element order differs); every energy "failed" and some runs threw `vector::at`. The mapping now goes through `ForceField::match_element` like `pair_coeff`.
* The hydrogen-bond list was built after the `bond_cut` filter, so acceptors between 5 Å and 7.5 Å were missed (periodic water boxes failed); fixed to the reference's `max(hbond_cut, bond_cut)` row cutoff.
* The Metal compiler rejected reference parameters without an address space (the CPU emulation cannot see this class of error); fixed with `RM_THREAD`. This is the first evidence the shared-header design works on the real compiler.
* **GPU-1 first failed on `cho_co`**: force RMS 5.4e-3 vs the owner's 5e-3. Diagnosis: float coordinates relative to the box corner (up to 17 Å here) quantise at 1.9e-6 Å, times the C–O force constant. Fix: hi/lo float positions, differences formed as `(hi_j−hi_i)+(lo_j−lo_i)`; all 58 fixtures then pass with margin. No tolerance was changed.
* The A1/A2 harnesses encoded "compute() refuses"; they were updated deliberately (A1: runs complete; A2 self-check stops with the summary).
* The `capabilities` test caught that `lammps.virial_fdotr`/`out.virial` may not be Implemented while the barostat gate (INT-4) is open; reverted to Planned with the INT-2 evidence in the note.

### Decisions I took that you should confirm
1. **FP32 stability by capped exponents (cap 40), not the algebraic STAB-n forms of NUMERICAL_POLICY 4.4.** It is finite on all 58 fixtures including the isolated atoms and the C=O molecule; the capped and true values differ only where the term saturates far below FP32 precision. The derivative-test protocol (STAB-1..n) was NOT run: the evidence is GPU-1 agreement, not per-expression tests.
2. **hi/lo coordinates** for all device kernels (resolves NUMERICAL_POLICY 4.1 item 6 as an M3 experiment: needed).
3. **Bond lists on the GPU are symmetric by construction**: each directed bond is corrected from its own end (the correction is symmetric) and `bond_mark` is not reproduced; it only changes bonds of atoms ≥ 4 bonds from an owned atom (CPU-64 keeps it). Torsion entries reproduce the reference's three-body-list gate (a later slot always, an earlier one only if k or l is owned) with slots sorted by neighbor index.
4. `backend` defaults to `cpu64` (the complete reference); `backend metal` must be requested.
5. ADR-022's "adapter refuses to compute until M4" is now superseded: compute works; per-atom outputs and non-fdotr virial paths still refuse.

### Not run / known gaps
NPT (INT-4), NVE drift protocol (NVE-1) and minimisation (MIN-1) beyond the 200-step water-box sanity run; the CPU-32 twin and FP32-1 envelope; decision-mismatch census; vdW type 2 and `enobonds no`; the GPU QEq (charges still come from the stock CPU fix); `reaxff/metal` per-atom energy/virial; MPI > 1 rank; systems larger than ~3 000 atoms were only timed, not validated. **Performance is not good yet**: a 3 000-atom water box takes 0.48 s of pair time for 20 steps on Metal vs 0.39 s for stock CPU `reaxff` — the pipeline re-uploads, re-bins and re-runs everything every step and the torsion/valence kernels use one thread per atom. That is the M8 work.

## GPU charge equilibration (fix `qeq/reaxff/metal`) — 2026-10-08

* New plugin fix derived from the stock `FixQEqReaxFF`: `compute_H()` and `sparse_matvec()` are overridden; the preconditioned CG, history extrapolation and charge update stay the stock double-precision code. H assembly (`rm_qeq_h`) and `y = (diag(eta)+H)x` (`rm_qeq_mv`) run on the GPU in FP32 over the engine's owned far rows, with the symmetric product as a gather (row part with ghost images mapped to their owner + column part from the host column index), so no atomics and a fixed summation order. Falls back to the stock CPU matrix for groups other than `all`, a taper radius above `nonb_cut`, or `backend cpu64`.
* INT-2 with `--gpu-qeq` (CTest `lammps_int2_metal_qeq`): 58 fixtures PASS under the C3 criteria; charges differ from the stock fix by at most 1.6e-5 e (limit 1e-4). CTest 32/32.
* 24k-atom water box, 10 NVE steps, `qeq/reaxff` tolerance 1e-10: Pair 1.06 s stock / 0.27 s Metal; Modify 1.34 s stock / 0.29 s with the GPU fix; total wall about 2.4 s stock, 1.6 s Metal pair only, 0.5 s with both.
* Not claimed: the FP32 matvec limits the attainable CG residual, so the requested tolerance is honoured only as far as the recursion allows; kernels have no CPU-emulation test (validated against the stock fix on the device only).

### Dynamics gates, leak and the oxide NVE failure — 2026-10-09
* **Memory leak found and fixed**: every Metal entry point leaked all its buffers when called from a plain C++ loop (no autorelease pool); a 648-atom MD run reached ~28 GB in 3000 steps and the 80 000-step NVE run was killed. RAII autorelease pool in every `Context` method; test `metal_no_leak` (fails without the pool, verified). Earlier 10-step benchmarks could not show it.
* **MIN-1 PASS**, **NPT (INT-4, water) PASS**, **NVE-1 organic PASS**: see VALIDATION rows. The water box heats to ~730 K in NVE for stock and plugin alike (an unrelaxed gas-phase CHO system), so it is a faithful but not a 300 K test.
* **NVE-1 oxide FAIL (open, needs an owner decision).** The Metal path's total energy is much noisier than stock on the VO crystal (RMS 11×, drift 4e-4 kcal/mol/atom/ps against the 2e-4 floor). Attribution by swapping parts to FP64 on the host (env `REAXMETAL_DEBUG_CPU_NB` / `REAXMETAL_DEBUG_CPU_BONDED`, diagnostics only): the FP32 bonded pipeline causes it; FP32 nonbonded and the GPU QEq matvec do not (the latter adds some drift). Force error vs CPU-64 on a thermalised frame: RMS 8.2e-4, max 1.3e-2 kcal/mol/Å (|F| RMS 28). It is per-term FP32 rounding in the bond-order derivative chain, not just summation, so compensated sums would not fix it. Options: (a) accept FP32 and set the MD criterion for Metal separately (thermostatted runs are fine); (b) bonded terms in double-single arithmetic (large rewrite); (c) bonded on CPU-64, nonbonded + QEq on GPU (exact, but the bonded part is the larger share of CPU time). Not decided; no criterion changed.
* Harness fixes: `run_nve.py` logged through pipes, which stalled the plugin run behind the stock run; now files.

### Speed pass and MD-stability policy — 2026-10-09
* **Owner decision**: accept FP32 for Metal; MD stability is judged under NVT, NVE drift is reported but not gated. NVT runs of the water box and the VO oxide (40 000 steps) agree with stock within block σ (VALIDATION).
* **Speed work**: one neighbor view (host atom set, device list, far rows) is built per step and shared by the pair style, the bonded input and the GPU QEq fix (`NbView`; rows shared through `shared_ptr`); nonbonded and far-row buffers are persistent; the row kernel initialises its own unused tail; the far-row capacity is remembered between steps (no regrow launch). All 35 CTest tests pass after the change.
* **Measured** (idle M5 Max, serial LAMMPS, CHO water, NVT, 100 steps, QEq 1e-6; stock `reaxff`+`qeq/reaxff` vs `reaxff/metal backend metal`+`qeq/reaxff/metal`), ms/step: 648 atoms 5.1 vs 5.6 (0.9×); 5 184: 32.6 vs 9.3 (3.5×); 24 000: 153 vs 21.2 (7.2×); 65 856: 411 vs 39.1 (10.5×). Stock is single-core; one system type; no oxide timing.
* Remaining host cost per step (24k atoms, profile `REAXMETAL_PROFILE=1`): bonded kernels ≈ 0.45 s/100 steps, the neighbor view ≈ 0.35, QEq matvec round trips ≈ 0.3 (13 per step), column index and packing ≈ 0.2 each. Next candidates: device-resident rows (no download/upload), fused s/t CG solves, device binning.
