# Architecture Decision Records

Format: **Status** is *Decided* (made in M0 on the evidence cited; revisit only with a log entry) or
*Proposed* (needs the owner's confirmation — also listed as an open question in the M0 report).
Nothing below has been implemented beyond the build skeleton.

---
## ADR-001 — Reference oracle and pin  · *Decided — owner approved `stable_30Sep2026` @ `8de817dd…` (M0 review)*
**Decision.** Primary oracle = `pair_style reaxff` (CPU, `src/REAXFF`) + `fix qeq/reaxff` of LAMMPS tag
`stable_30Sep2026`, commit `8de817dd79bfe4525d5d39246a212d833e6dee07`. Secondary cross-check = `reaxff/kk`
(Kokkos, double). The Kokkos pair style is **not** the oracle: it restructures math (`cbrt`, reduced
exponentials) and hard-codes `thb_cutsq` (Q-17).
**Why.** Latest stable tag at audit time; the CPU path is the documented reference implementation ("pair style
reaxff follows the reference implementation"). **Alternatives.** An older stable (e.g. `stable_22Jul2025_update6`)
if the owner needs to match an installed LAMMPS; moving the pin = new fixtures + log entry. Nothing in the audit found a
reason to prefer an older release, but I only read the pinned tree — no diff against other releases was made.

## ADR-002 — Reproduce the reference, not the textbook  · *Decided*
Every behaviour in ENGINE_SPEC §10 has a disposition: **R**eplicate, **D**eviate by design (and say so), **F**lag.
Deviations are limited to places where the reference is undefined or silently wrong (zero-filled missing pairs,
array overrun, conflicting vdW types) or where it is an artefact of domain decomposition (ghost bookkeeping).
No deviation may change a result that the reference computes correctly.

## ADR-003 — Element-agnostic parameter tables; LAMMPS' element knowledge becomes data  · *Decided — owner approved (M0 review #2)*
**Context.** Rule 1 forbids hard-coding elements, yet the reference contains element logic in kernels
(Q-06): `strcmp(name,"C")` for the C2 correction, exact masses 12.0000/15.9990 for the C–O triple-bond
stabilisation, and `mass>21` / `mass<21` for the first-row/second-row split.
**Decision.** Atoms carry an integer *type* (index into dense parameter tables). At ffield-load time the loader
evaluates the **same predicates as the reference** and stores the outcome as per-type / per-pair *flags*
(`compat.c2_correction`, `compat.triple_bond_stabilisation`, `compat.light_element_split`). Kernels read flags,
never names or masses. Flags are logged at load, and overridable explicitly. Consequence: bit-for-bit LAMMPS
behaviour is kept while no kernel mentions any element. **Alternative:** drop these corrections (violates rule 3;
changes energies of C/O systems — rejected). **Owner decision:** approved. Flags must preserve the exact source behaviour (including exact-equality mass tests and strict `<`/`>` thresholds) and are *compatibility flags, not new chemical rules*;
kernels stay generic. **Done in M0.5:** `compat_flags.hpp` with one regression-test block per predicate (boundaries 20.999…/21.0/21.000…1, `"C"` vs `CL`/`CA`/lower case/>3 chars, exact `12.0000`/`15.9990` incl. `nextafter` neighbours and either order, `(int)gp[37]` truncation incl. negatives/NaN/out-of-range/short `gp`, strict `gp[5] > 0.001`) — 12 deliberate mutations all caught.
Parameter storage: `n_types`-indexed dense tables for 1/2-body; index-table + packed parameter sets for 3/4-body/H-bond
(multiple 3-body sets per triple are *summed*, Q-09); size limit on `n_types` (dense `n⁴` index) measured in M2.

## ADR-004 — Periodicity: never minimum-image  · *Decided; the explicit-image mechanism is amended by ADR-013 (ghost-native)*
**Amendment (M0.5):** with LAMMPS as the host, periodic images arrive as ghost atoms (ADR-013); the *engine* does no shift arithmetic. What stays: minimum-image is forbidden
(measured: wrong by 0.06–7.8 e in QEq charges for the LAMMPS regression cells, LAMMPS_INTEGRATION §7.3). The text below describes the standalone/image-expander variant.

**Evidence.** The LAMMPS reaxff unit tests use a 7.54 Å cell against a 10 Å taper cutoff; GMD-Reax used minimum
image (invalid there). **Decision.** Neighbor lists carry integer lattice shifts; a pair is (i, j, shift);
cell matrix supports triclinic; non-periodic dimensions are first-class. Each physical interaction (including
self-image pairs i–i′) is counted exactly once. Lists that depend on bond orders (bonds, 3-body, H-bond)
are rebuilt every evaluation as in the reference; the geometric (cutoff) list uses a skin with a displacement
trigger. All list capacities are explicit and **overflow is detected and reported** (grow-and-retry), replacing the
reference's `safezone/mincap` heuristics, which can fail with "bondchk failed". Sufficiency of LAMMPS' ghost shell for
the oracle is established by REF-GHOST (VALIDATION).

## ADR-005 — Shared term functions, independent drivers  · *Decided — owner approved (M1 review)*
**Decision.** The *pure* term functions (BO correction & derivative coefficients, atom terms, angle, torsion,
H-bond, pair vdW/Coulomb) are written once, templated on scalar, in an MSL-compatible C++ subset header
(no exceptions/RTTI/STL; address-space macros). CPU-64 and CPU-32 instantiate them; the Metal kernels include the same
header. **Traversal, accumulation and decomposition are separate and backend-specific**: CPU follows the reference's
serial loops for auditability; GPU uses per-atom gather kernels (ADR-006). Rule 6 (independently testable) holds:
each backend is its own executable path with its own tests; the CPU-32 twin isolates FP32 error from GPU effects.
**Risk.** Shared formulas are a common-mode failure channel for *formula* bugs → mitigated by validating CPU-64
against pinned LAMMPS and by per-term FD tests, both independent of the shared code's author.
**Alternative.** Hand-written MSL kernels (rejected: two copies of ~1 500 lines of equations will drift).

## ADR-006 — Metal execution architecture  · *Proposed (design only; nothing run)*
* **Persistent buffers.** Positions/types/charges/forces, all lists and coefficient arrays live in `MTLBuffer`s
  allocated once, grown on overflow; one command buffer per evaluation; no per-step host↔device copies except
  scalars and (optionally) positions/forces.
* **Layout.** SoA, 32-bit indices, column-major per-atom bond slots (GMD-Reax observation for coalescing),
  **redundant (full) bond/H-bond lists with `sym_index`** (PuReMD-GPU), counts via prefix sums (deterministic fill).
* **Pipeline** (mirrors `Compute_Forces`): cell/neighbor build → raw BO' + ΣBO' + bond-list fill → BO correction
  (needs complete Δ') → atom quantities (Δ, nlp) → bond/atom energies + `Cdbo/CdDelta` → valence → torsion →
  H-bond → nonbonded (needs q) ; QEq solve before nonbonded → **bond-derivative gather** (`Add_dBond_to_Forces`) → reductions.
  The QEq solve depends only on geometry, so it can overlap with the bonded kernels.
* **Open design problem (M6) — deterministic scatter.** Every multi-body term in the reference scatters onto atoms
  and bond slots other than the one that owns the work item (all serial there; verified by reading):
  valence centred on `j` adds forces to `f[i], f[k]` and `CdDelta[i], CdDelta[k]` (coalition) although its `Cdbo`
  writes stay in `j`'s own bond list; **torsion for bond (j,k)** adds forces to `i,j,k,l`, `CdDelta[j],[k]`, and `Cdbo`
  of the `k→l` slot in *k's* list, which is also written by torsions of every other neighbour `j'` of `k`; H-bond writes
  forces on `i,j,k` (its `Cdbo` write stays in the hydrogen's own `j→i` slot). A naive GPU port therefore needs atomics (PuReMD-GPU kept atomics
  only for the 4-body kernel; its Table 5 shows what they cost in FP64). Deterministic options to evaluate in M6:
  (a) per-work-item scratch slots + fixed-order segmented reduction (memory ∝ #quadruples ≈ N·b³, b = bonds/atom),
  (b) owner-computes with recomputation of shared sub-expressions, (c) bond colouring/batching so concurrent work
  items never share a target. The choice is made on measured memory and time; FP32 atomics are only the M8
  benchmark option (rule 9) and are *not assumed to exist or to be fast* on Apple GPUs (to be verified at M3).
* **Threading.** 1 thread/atom for BO, atom terms, valence, torsion (PuReMD-GPU found multi-thread-per-atom slower there,
  Table 4); multiple threads/atom (SIMD-group cooperative) for neighbor search, H-bond, SpMV, nonbonded. Block sizes
  tuned in M8 only.
* **QEq.** Kernel set: H assembly (CSR or matrix-free), SpMV (SIMD-group per row), fused two-vector CG
  (`s` and `t` together, as in Kokkos), device-side dot products with fixed reduction trees; convergence flag read
  back each *k* iterations. FP32 strategy decided at the M5 gate (NUMERICAL_POLICY 4.3).

## ADR-007 — Precision and determinism  · *Decided; owner-modified in M0 review #5; numbers proposed in NUMERICAL_POLICY §5 for confirmation before M1 freezes them*
See NUMERICAL_POLICY. **Owner modification:** the CPU-FP32 twin characterises floating-point behaviour but is **not** the sole GPU acceptance criterion; independent energy, force, charge, derivative and MD-stability tolerances are specified *before* any Metal result is evaluated; separate CPU-64↔LAMMPS, CPU-32↔CPU-64 and Metal-32↔CPU-64 comparisons; mathematically unstable expressions (the identified `exp` overflow) are replaced by algebraically equivalent stable forms and derivative-tested. Summary: CPU-64 strict-FP reference; CPU-32 twin; Metal-32; FP32 range audit per `exp/pow`
(M0 found real overflow cases); no FP atomics in the validated path; tolerances frozen before comparison; Metal
tolerance derived from the CPU-32 envelope, never from GPU output.

## ADR-008 — Build system and Metal toolchain  · *Decided; Metal parts untested — updated with the owner's machine facts*
CMake ≥ 3.24, C++20, no Objective-C++ in the core. The Metal backend (M3) exists only on Apple platforms
(`REAXMETAL_ENABLE_METAL` currently hard-fails by design). Host API via **metal-cpp** (Apple-distributed header
package; pinned by version + sha256 when M3 starts — `developer.apple.com/metal/cpp` was reachable from the M0
sandbox, version/licence not yet recorded). Shaders: compiled offline (`xcrun metal` → `.metallib`) as a CMake step, with an
optional runtime-compile path for debugging. **The development sandbox is Linux: the Metal backend cannot be built or run
here**; every Metal result comes from the owner's **Apple M5 Max (40-core GPU, macOS 26.3.1, SDK 26.2, Apple clang 17, CMake 4.2.3)**.
**Owner-reported constraint: only Command Line Tools are installed — no Xcode and no `metal` compiler (`xcrun` cannot find it).** Consequence: the *offline* `.metallib` path is unavailable, so the baseline is **runtime compilation from source** through metal-cpp (`newLibrary(source…)`); whether that works without Xcode on that machine is the first M3 check (unverified). The offline path stays an optional CMake step for machines with Xcode. The GPU reports Metal 4; no Metal 4 API is used until verified — the baseline is whatever the runtime compiler accepts. Details: LAMMPS_INTEGRATION §11.

## ADR-009 — Tests, fixtures, scripting  · *Decided*
CTest; at M0 a ~60-line assertion header (no third-party dependency); a framework (Catch2/GoogleTest via a
pinned `FetchContent`) may be adopted at M2. Fixtures are plain text + a hashed manifest. Python is confined to
`tools/` (fixture generation, survey/analysis) until the M7 bindings (nanobind/pybind11 + ASE calculator); the core
has no Python dependency. Doc/code drift is a failing test (`docs_sync`).

## ADR-010 — License and notices  · *Decided — owner approved GPL-2.0-only (M0 review #1); audit and enforcement implemented in M0.5*
Audit facts (SOURCE_MAP §6): physics-core files carry PuReMD "GPL v2 or any later version"; LAMMPS-authored
files (incl. `fix_qeq_reaxff`, `pair_reaxff`, Kokkos) carry LAMMPS' GPL (v2, `LICENSE`).
**Proposed:** project license `GPL-2.0-only` (compatible with adapting anything in the tree), `SPDX` identifier in every file,
verbatim upstream notice blocks + `Adapted-from: <file>@8de817dd` in every adapted file (`docs/NOTICE_TEMPLATE.txt`,
`THIRD_PARTY_NOTICES.md`). No LAMMPS code is vendored or adapted in M0.
**Done in M0.5 (owner: "do not assume every upstream file has identical terms"):** `tools/audit_licenses.py` classifies the notice text of **130 upstream files** (derived-from, linked-against, or tested-with) into
`third_party/lammps/LICENSE_AUDIT.tsv`: 76 LAMMPS-GPL, 17 PuReMD GPL-2.0-or-later, 1 MIT-style (`fmt`, headers only included by the plugin build), 35 with **no per-file notice** (ffields, regression YAML, plugin
example build files — covered only by the tree-level GPLv2). The plugin interface headers carry the plain LAMMPS GPL (not LGPL) in this tree. Top-level `LICENSE`, `LICENSES/GPL-2.0-only.txt`, `REUSE.toml`, SPDX in every authored
source file, and a `license_headers` test (every authored file has SPDX; every `Adapted-from:` file keeps an upstream notice and is listed in `THIRD_PARTY_NOTICES.md`; license text identical across copies) enforce it.
**Not decided / flagged:** the ffield parameter sets have scientific provenance (papers) but no file-level license — they are fetched at test time, not vendored, until the owner decides; the copyright holder name in `REUSE.toml` is a placeholder ("ReaxMetal contributors"). If the owner wants a more permissive
licence, the equations/conventions must be re-derived without adapting source (a clean-room process has to be
set up *before* M2) — this is a legal/strategy decision that I cannot make; I have not added a top-level
`LICENSE` file until it is made (the `SPDX` tags in the skeleton are provisional).

## ADR-011 — Capability gating as data  · *Decided*
`reaxmetal::feature_table()` is the single machine-readable capability matrix; `require_supported()` is the gate used
by parsers and APIs; `docs_sync` keeps `FEATURE_MATRIX.md` identical. Unsupported variants are errors (rule 4).

## ADR-012 — Repository layout  · *Decided*
```
include/reaxmetal/      public headers (pins, capabilities, energy_terms … ; later: forcefield, system, engine)
src/core/               types, constants, registries            (built at M0)
src/io/                 ffield / control / system readers       (M2)
src/neighbor/           cell lists, image shifts                (M3)
src/physics/            shared pure term functions              (M4–M6)
src/cpu/                CPU-64 / CPU-32 reference drivers       (M4–M6)
src/qeq/                CG solver + H assembly                  (M5)
src/metal/              runtime + kernels (Apple only)          (M3+)
plugin/                 LAMMPS plugin: adapter + DSO (probe in M0.5)       (A0 done; A1 at M2)
python/                 optional standalone evaluator + ASE      (after M7)
tools/                  fetch_lammps.sh, survey_ffield.py, fp32_hazards.py, (M1) fixture generator
fixtures/               reference fixtures + hashed manifests   (M1)
third_party/lammps/     PIN.txt, SOURCE_HASHES.sha256, COPYING (no vendored code)
docs/                   the eight documents (incl. LAMMPS_INTEGRATION.md)
tests/                  CTest suite;  tests/lammps/ = in-LAMMPS integration probe (gated by REAXMETAL_LAMMPS_PREFIX)
```

---
## ADR-013 — LAMMPS-first architecture and re-planned milestones  · *Decided (owner directive, M0 review)*
**Context.** The owner requires ReaxMetal to be usable *inside LAMMPS* and forbids duplicating LAMMPS' MD machinery.
**Decision.** Layers: (1) backend-independent ReaxFF library (`reaxmetal_core`: parsers, tables, term functions, lists, CPU-64/CPU-32 reference, EEM reference solver);
(2) Metal backend; (3) thin LAMMPS adapter `pair_style reaxff/metal` shipped as a **loadable plugin**; (4) CPU reference path for validation; (5) standalone evaluator/Python later.
The engine is **ghost-native** (consumes LAMMPS' owned+ghost atom set; replicates the reference's owner-computes rules; the standalone harness uses an image expander) — this *replaces*
the explicit-image-shift design as the mechanism and turns ENGINE_SPEC D-1 into a replication (R). Neighbor lists are built on the device in production. Single MPI rank; others fail explicitly. Full contract: `LAMMPS_INTEGRATION.md`.
**Evidence (executed, Linux):** a `Pair`-derived plugin loads; ghost forces fold back exactly; the stock charge fixes work against it; no pair neighbor request is required; topology changes only at rebuilds.
**Consequences.** (a) no `src/md`; M7 becomes LAMMPS-hosted validation (NVE drift, minimiser, per-atom outputs, NPT after virial) plus an optional standalone evaluator;
(b) the primary oracle is **in-LAMMPS A/B** (`pair_style reaxff` vs `reaxff/metal backend cpu64` on identical input) with hashed fixtures for localisation;
(c) an A-series for the adapter. **Re-planned milestones:**

| Milestone | Content (changes from the original plan in bold) |
|---|---|
| M0 ✓ | audit, spec, decisions, skeleton (pushed) |
| **M0.5 ✓** | **LAMMPS integration architecture, probe plugin, license audit, compat-flag derivation, EEM/strictness/scope decisions** |
| M1 | pinned fixture generator **+ instrumented LAMMPS patch (hashed)**; experiments Q-09, Q-12, REF-GHOST, REF-QEQ-CELL, noise floor; **tolerance freeze (owner-confirmed numbers)** |
| M2 | ffield/control parser, tables with compat flags, **adapter A1: real parse + `extract(chi/eta/gamma)` + host checks, compared to stock** |
| M3 | Metal runtime (runtime-compiled shaders), device cell/neighbor lists over owned+ghost, deterministic reductions; **first run on the M5 Max** |
| M4 | bond orders + stable formulations, atom terms; CPU-64/CPU-32; derivative tests; adapter A2 (partial energies) |
| M5 | EEM (reference solver, GPU solver, **derived strict fix**), nonbonded; strict/compat policy |
| M6 | bonded terms + forces + **virial (enables NPT)**; in-LAMMPS A/B complete |
| **M7** | **LAMMPS-hosted validation: NVE drift, minimiser, per-atom outputs; optional standalone evaluator/ASE** (no standalone MD engine) |
| M8 | optimisation, batching, benchmarks (atomics benchmark option) |

## ADR-014 — EEM naming, scope and strict convergence  · *Decided (owner M0 review #6, #8)*
The internal charge-model abstraction is **EEM**, defined as the LAMMPS-compatible model (`fix qeq/reaxff` ≡ `fix qeq/shielded` kernel, Q-26) — no new physics. In scope: standard EEM/QEq **and LAMMPS-compatible shielded charge equilibration**;
deferred with explicit errors: ACKS2, QTPIE, `qeq/rel`, external fields, alternative models, tabulated interactions, non-neutral groups. **Non-convergence is an error/status by default**; a separate explicit compatibility mode
warns and continues like LAMMPS; an unconverged solution is never silently accepted. Barostats: enabled through LAMMPS once the pair style reports validated global virial (`lammps.virial_fdotr`, M6); until then pressure-controlled runs are refused.
The stock fix cannot report convergence (Q-28), hence ADR-015. AMS equivalence is *not claimed* (LAMMPS_INTEGRATION §7.2).

## ADR-015 — Plugin class structure: `Pair`-derived style; derived QEq fix for strictness  · *Decided — owner approved; implement at M5 (not before)*
**Decision.** `reaxff/metal` derives from **`Pair`** (verified to work with the stock charge fixes, LAMMPS_INTEGRATION §3.3). For strict EEM, a second plugin style `fix qeq/reaxff/metal` **subclasses `FixQEqReaxFF`** (protected state accessible; REAXFF package required in the host — it is the oracle build anyway)
and verifies the true residual after the base solve (LAMMPS_INTEGRATION §7.4); later replaceable by a GPU-resident own fix.
**Alternatives.** Derive the pair from `PairReaxFF` (rejected: drags in the PuReMD host machinery and heuristics, not needed [V]); a built-in package patch to LAMMPS (rejected for now: keeps the stock tree pristine, ADR-001); adapter-side residual check with user-repeated parameters (fragile).
**Risk.** The derived fix depends on `protected` layout of one pinned LAMMPS version → it is pinned to `stable_30Sep2026` and compile-checked; moving the pin means revisiting it.

### Amendments recorded after the M1 review
**ADR-005 (approved, with conditions).** Pure mathematical term functions are shared across CPU-64, CPU-32 and Metal-32 *wherever practical*; traversal, neighbor construction, reductions and accumulation stay backend-specific and independent. Sharing prevents drift but **does not prove the formulas
correct**, so three independent checks are mandatory: (1) term outputs against the pinned LAMMPS reference (M1 instrumentation provides them); (2) analytical derivatives against finite differences; (3) CPU-32 vs CPU-64, then Metal-32 against both.
**Restriction:** the CPU implementation's mathematical correctness is never compromised to fit Metal syntax; a function that cannot be shared cleanly (or whose sharing would introduce unsafe Metal behaviour) gets **two implementations with an explicit equivalence test**.
**ADR-015 (approved, M5).** No derived-fix work before M5. **M1 consequence:** the reference harness records QEq tolerance, `maxiter`, iteration counts and warnings for every fixture, computes an independent equalization residual, and **rejects any fixture whose convergence is unverified or that emitted a solver warning** — an unconverged reference calculation is *invalid* and never enters the golden dataset.
Reference charge formulation = standard LAMMPS-compatible ReaxFF EEM/QEq. AMS EEM cross-validation is a separate optional task, not a blocker.

## ADR-016 — Development environments, verification-status taxonomy, branch policy  · *Decided (owner, M1 review)*
Development is **cloud Claude Code on Linux first; the M5 Max later**. Planned split:

| Milestone | Cloud Claude Code (Linux) | M5 Max MacBook Pro |
|---|---|---|
| M1 LAMMPS oracle | full development and testing | not needed |
| M2 generic parser | full development and testing | not needed |
| M3 Metal backend | write shaders, host code, static tests | compile, execute, debug |
| M4 bond orders | CPU FP64/FP32 tests; Metal source development | GPU numerical tests |
| M5 EEM + nonbonded | CPU/reference tests | Metal QEq tests |
| M6 full forces | CPU/reference tests | GPU force validation |
| M7 LAMMPS dynamics | CPU NVE/NVT and integration tests | full Metal MD |
| M8 performance | benchmark harness and analysis | M5 Max profiling |

**Status taxonomy (mandatory in every report, per backend):** *written* (source exists) → *compiled* (built by a named compiler on a named platform) → *executed* (ran, with recorded output) → *validated* (executed and met the frozen criteria). Metal code is at most *written* until it is compiled and run on macOS hardware; **CPU and Metal verification statuses are tracked separately** and a Metal milestone is never closed from the cloud.
**Branch policy:** M3 is **not** a permanent blocker for M4–M6 CPU physics. If Metal is unavailable, CPU physics proceeds on its own line of work (separate CPU-physics branch when the Metal side has unmerged/unverified code) while the GPU verification gates stay open. When M3 begins, a **Mac validation checklist** is produced. Cloud work must not stall waiting for hardware.
M1/M2 are **not** blocked by Metal or Xcode.

## ADR-017 — Fixture and data policy  · *Decided (owner, M1 review)*
Hybrid: **commit** small, independently generated geometry fixtures and numerical reference outputs with complete provenance (case spec, ffield name + SHA-256, LAMMPS commit/patch/build identifiers, solver settings, convergence diagnostics, hashes).
**Public LAMMPS potential files** are vendored only if redistribution is permitted; the M0.5 audit found **no per-file notice** on them (35 such files), so they are **fetched at a pinned version with SHA-256 verification** and not committed. **No proprietary or uncertain-license AMS force fields or private parameterisations are ever committed**; optional gitignored local fixtures
(`fixtures/local/`) are supported by the runner. AMS EEM cross-validation waits for a reference calculation from the owner.

## ADR-018 — Copyright attribution  · *Decided (owner, M1 review)*
Original ReaxMetal files carry `SPDX-FileCopyrightText: 2026 Anirban Phukan` (assuming no university or third party holds those rights) beside `SPDX-License-Identifier: GPL-2.0-only`. **Actual upstream copyright notices and license information are preserved verbatim for derived files and are never reassigned to the project owner**
(e.g. the structural adaptation in `plugin/probe/pair_reaxff_metal_probe.h` keeps the Sandia notice). The `license_headers` test checks the attribution line.

## ADR-019 — Reference oracle = pinned LAMMPS + a hashed, observation-only diagnostics patch  · *Implemented (M1)*
The oracle is the pinned `stable_30Sep2026` tree. Instrumentation is **one patch** (`third_party/lammps/patches/0001-reaxmetal-diagnostics.patch`, SHA-256 in `PATCHES.sha256`, applied to a private clone by `tools/build_lammps_instrumented.sh`; the pristine tree is never modified and the builder refuses a tree whose diff hash differs).
The hooks only *copy values out* (parameter tables after parsing, BO′ before correction, per-atom Δ/nlp/Clp, bond lists, force-stage `workspace->f`, per-class interaction tallies, QEq iteration counts and recursive residuals) and are inert unless `REAXMETAL_DIAG_DIR` is set.
**Oracle configuration:** strict IEEE builds (no FMA contraction: gcc and clang on baseline x86-64 are bit-identical on all fixtures). Instrumented ≡ stock **bitwise** for those builds, diagnostics on or off. For `-march=native -ffp-contract=fast` builds the instrumented binary is *not* bitwise equal to stock (≤ 1.3e-13 in forces; the compiler's fusion choices depend on the surrounding code) — such builds are used only **as stock** noise-floor probes, never as the instrumented oracle.
Throwaway **experiment patches** (`patches/experiments/EXP-000x-*.patch`, applied on top of 0001) change reference *behaviour* to confirm a root cause (Q-32, Q-34); builds from them are never used for fixtures.

## ADR-020 — Tolerance freeze and the conditioning-definition amendment (proposal)  · *Frozen file written; amendment awaits owner confirmation*
`tolerances/tolerances.json` (+ `TOLERANCES.sha256`, enforced by the `tolerances_frozen` CTest) carries C1 thresholds = ⌈10 × measured noise floor⌉ (one significant digit), the owner's C3 values verbatim, the QEq/neutrality thresholds, the conditioning classification of every fixture and the FD protocol.
M1 execution showed that the **pre-registered** well-conditioned definition (NUMERICAL_POLICY §5.2) lets through exactly collinear fixtures whose forces differ by 1.5e-2 kcal/mol/Å between FMA and non-FMA builds of the *same* source (a 1e-2 floor would make C1 meaningless). Two classes are therefore **proposed** as additions (X1 near-linear angle, X2 BO-product within 1e-3 of `thb_cutsq`); the file records C1 under both definitions, **primary = with the amendment**. The owner decides; nothing in C3 was changed.

## ADR-021 — Reference defects are reproduced by default, flagged, and never silently corrected  · *Proposed (consequence of M1 findings)*
Q-32 (H-bond acceptor that is a periodic image of the donor is dropped, tag comparison), Q-34 (analytic force ≠ gradient of the reported energy for heavy atoms with π bonds: `dDelta_lp[j]` instead of `dDelta_lp_temp[j]`) are defects of the pinned reference, confirmed by modified-source experiments. The engine's contract is *parity with pinned LAMMPS*: both are reproduced by default behind named compat flags (`compat.hbond_donor_image_exclusion`, `compat.ovun_heavy_neighbor_force`), documented as defects, with a corrected variant opt-in and reported separately in validation. Q-09 overrun and Q-12 zero-fill are *rejected* with an explicit error (undefined behaviour / phantom physics cannot be a parity target), and Q-35 (QEq taper beyond the ghost shell) is an error. Whether to report Q-32/Q-34 upstream is the owner's call.

## ADR-022 — Strict parsing with per-used-pair Q-12 enforcement; adapter A1 ships no physics and refuses to compute  · *Implemented (M2); consequence of M1 findings*
* The reader reproduces everything pinned LAMMPS parses correctly (bit-identical tables, PARSE-1/3) and rejects what LAMMPS silently corrupts (Q-09 overrun, non-finite numbers, truncation, <38 general parameters). Rejecting absent bond-pair blocks at file level would make five bundled force fields unusable for the element sets they were built for, so Q-12 is enforced for the element pairs a run actually uses (`pair_coeff` time). Alternative rejected: zero-fill as LAMMPS does (creates phantom bonds, Q-12).
* A1 registers the real style name so that scripts, `extract()` consumers and host checks are exercised against the stock fixes early, but `compute()` throws an explicit error until M4; a zero-energy stub was rejected because it can silently pass a smoke test.
* Plugins must be built with the same `fmt` branch as the target LAMMPS (LAMMPS_INTEGRATION M2 addenda); the default matches a default (C++17) LAMMPS build.

## ADR-023 — The Metal host layer is Objective-C++ against the system frameworks, not metal-cpp  · *Decided by me in M3 — supersedes the metal-cpp sentence of ADR-008; please confirm*
ADR-008 chose Apple's header-only **metal-cpp** for the host API. In M3 I downloaded it (`metal-cpp_26.zip`, sha256 `4df3c078…a3a4`, 144 files) and read its `LICENSE.txt`: it is the **Apache License 2.0**. The FSF considers Apache-2.0 compatible with GPLv3 but **not** with GPL-2.0-only, the project license (ADR-010). Whether an `-only` work may include such headers in a distributed binary is a legal question I cannot settle, so I did not vendor or include it. The Metal host (`src/metal/metal_backend.mm`) is therefore Objective-C++ calling the system `Metal.framework` / `Foundation.framework` directly (system libraries; no third-party code), behind a plain-C++ interface (`include/reaxmetal/metal_backend.hpp`, no Apple types) so that the core and the adapter stay C++. Consequences: no download or pinning of metal-cpp; ARC (`-fobjc-arc`) manages the Metal objects; the code is a thin layer (~250 lines). Alternative kept open: if you want metal-cpp anyway (e.g. for the Apache NOTICE obligations being acceptable to you), only `metal_backend.mm` changes. Shaders are compiled **at run time from source** (`newLibraryWithSource:`), with fast-math disabled, because Xcode's `metal` compiler is not installed on the target machine (ADR-008); whether that works there is the first thing step 1 checks.

## ADR-024 — Neighbor data model for M3: ghost-native far list, host-side binning, float rows with an explicit margin contract  · *Implemented (M3), Metal side written only*
* **Far list** = what the reference gives its kernels (`write_reax_lists`, built from a half, newton-off list with ghost neighbors): row *i* of **every** owned and ghost atom holds the atoms *j > i* within the **row cutoff** (`nonb_cut` for owned rows, `bond_cut` for ghost rows, Q-08). Owner-computes counting (tag order, coordinate tie-break with `SMALL = 1e-4`) is applied by the consumer (`classify_nonbonded_entry`), not baked into the list, exactly as the reference does. Verified three ways: against brute force; against the 58 M1 tallies (NBR-3); and, inside LAMMPS, row by row against LAMMPS' own list (INT-7).
* **Cell binning stays on the host** (O(N), only when the topology changes, `ago == 0`); the device receives the cell id of every atom plus the CSR cell contents and runs one thread per atom over the 27 surrounding cells. A deterministic *device* counting sort needs scan primitives and atomics-free segmented kernels; that is an optimisation (M8), not needed for correctness, and I did not want to write unverified GPU sorting code before the simple path has run once. This narrows the ADR-006/M3 wording "device cell lists" — flagged.
* **Float rows with a margin contract.** Positions are float relative to the box lower corner (NUMERICAL_POLICY 4.1); the device cutoff is `cutoff + margin`, `margin = max(1e-3 Å, 8·FLT_EPSILON·max|x|)` (float error of a distance is below `4·FLT_EPSILON·max|x|`). Contract (`compare_rows_to_far_list`): every CPU-64 pair is present (superset); every device pair has CPU-64 distance ≤ cutoff + 2·margin; stored `r2` equals the double value within `2(d+margin)margin`. Pairs in the margin band are legitimate and must be re-filtered by the consumer in its own precision; the band was 0.03 % of entries in the test geometries.
* **Fixed-capacity rows + status** instead of device-side compaction: each row stores its true count even when it exceeds `cap`; the host grows `cap` to the exact need and relaunches (`build_far_rows_with_growth`, LAMMPS_INTEGRATION S6). Results are a function of the inputs only (no atomics): row order = (z,y,x cell, ascending index).
* **Reductions**: canonical fixed order (chunked sequential float sums, then sequential over partials). Slower than a tree reduction, and deliberately so: it is bit-identical on CPU and GPU, which makes the first on-device comparison unambiguous. A faster tree order can replace it once the CPU twin implements the same tree.
