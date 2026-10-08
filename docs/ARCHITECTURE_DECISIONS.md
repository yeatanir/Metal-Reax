# Architecture Decision Records

Format: **Status** is *Decided* (made in M0 on the evidence cited; revisit only with a log entry) or
*Proposed* (needs the owner's confirmation — also listed as an open question in the M0 report).
Nothing below has been implemented beyond the build skeleton.

---
## ADR-001 — Reference oracle and pin  · *Decided (owner may override)*
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

## ADR-003 — Element-agnostic parameter tables; LAMMPS' element knowledge becomes data  · *Proposed (needs owner OK)*
**Context.** Rule 1 forbids hard-coding elements, yet the reference contains element logic in kernels
(Q-06): `strcmp(name,"C")` for the C2 correction, exact masses 12.0000/15.9990 for the C–O triple-bond
stabilisation, and `mass>21` / `mass<21` for the first-row/second-row split.
**Decision.** Atoms carry an integer *type* (index into dense parameter tables). At ffield-load time the loader
evaluates the **same predicates as the reference** and stores the outcome as per-type / per-pair *flags*
(`compat.c2_correction`, `compat.triple_bond_stabilisation`, `compat.light_element_split`). Kernels read flags,
never names or masses. Flags are logged at load, and overridable explicitly. Consequence: bit-for-bit LAMMPS
behaviour is kept while no kernel mentions any element. **Alternative:** drop these corrections (violates rule 3;
changes energies of C/O systems — rejected). **Open:** owner to confirm this reading of rule 1.
Parameter storage: `n_types`-indexed dense tables for 1/2-body; index-table + packed parameter sets for 3/4-body/H-bond
(multiple 3-body sets per triple are *summed*, Q-09); size limit on `n_types` (dense `n⁴` index) measured in M2.

## ADR-004 — Periodicity: explicit images, never minimum-image  · *Decided*
**Evidence.** The LAMMPS reaxff unit tests use a 7.54 Å cell against a 10 Å taper cutoff; GMD-Reax used minimum
image (invalid there). **Decision.** Neighbor lists carry integer lattice shifts; a pair is (i, j, shift);
cell matrix supports triclinic; non-periodic dimensions are first-class. Each physical interaction (including
self-image pairs i–i′) is counted exactly once. Lists that depend on bond orders (bonds, 3-body, H-bond)
are rebuilt every evaluation as in the reference; the geometric (cutoff) list uses a skin with a displacement
trigger. All list capacities are explicit and **overflow is detected and reported** (grow-and-retry), replacing the
reference's `safezone/mincap` heuristics, which can fail with "bondchk failed". Sufficiency of LAMMPS' ghost shell for
the oracle is established by REF-GHOST (VALIDATION).

## ADR-005 — Shared term functions, independent drivers  · *Proposed*
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

## ADR-007 — Precision and determinism  · *Decided in principle; numbers pending*
See NUMERICAL_POLICY. Summary: CPU-64 strict-FP reference; CPU-32 twin; Metal-32; FP32 range audit per `exp/pow`
(M0 found real overflow cases); no FP atomics in the validated path; tolerances frozen before comparison; Metal
tolerance derived from the CPU-32 envelope, never from GPU output.

## ADR-008 — Build system and Metal toolchain  · *Decided (Metal parts untested in M0)*
CMake ≥ 3.24, C++20, no Objective-C++ in the core. The Metal backend (M3) exists only on Apple platforms
(`REAXMETAL_ENABLE_METAL` currently hard-fails by design). Host API via **metal-cpp** (Apple-distributed header
package; pinned by version + sha256 when M3 starts — `developer.apple.com/metal/cpp` was reachable from the M0
sandbox, version/licence not yet recorded). Shaders: compiled offline (`xcrun metal` → `.metallib`) as a CMake step, with an
optional runtime-compile path for debugging. **The M0 sandbox is Linux: the Metal backend cannot be built or run
here**, so every Metal result must come from an Apple-silicon machine (open question).

## ADR-009 — Tests, fixtures, scripting  · *Decided*
CTest; at M0 a ~60-line assertion header (no third-party dependency); a framework (Catch2/GoogleTest via a
pinned `FetchContent`) may be adopted at M2. Fixtures are plain text + a hashed manifest. Python is confined to
`tools/` (fixture generation, survey/analysis) until the M7 bindings (nanobind/pybind11 + ASE calculator); the core
has no Python dependency. Doc/code drift is a failing test (`docs_sync`).

## ADR-010 — License and notices  · *Proposed (needs owner decision)*
Audit facts (SOURCE_MAP §6): physics-core files carry PuReMD "GPL v2 or any later version"; LAMMPS-authored
files (incl. `fix_qeq_reaxff`, `pair_reaxff`, Kokkos) carry LAMMPS' GPL (v2, `LICENSE`).
**Proposed:** project license `GPL-2.0-only` (compatible with adapting anything in the tree), `SPDX` identifier in every file,
verbatim upstream notice blocks + `Adapted-from: <file>@8de817dd` in every adapted file (`docs/NOTICE_TEMPLATE.txt`,
`THIRD_PARTY_NOTICES.md`). No LAMMPS code is vendored or adapted in M0. If the owner wants a more permissive
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
src/md/                 integrators, minimiser                  (M7)
python/                 bindings + ASE calculator               (M7)
tools/                  fetch_lammps.sh, survey_ffield.py, fp32_hazards.py, (M1) fixture generator
fixtures/               reference fixtures + hashed manifests   (M1)
third_party/lammps/     PIN.txt, SOURCE_HASHES.sha256, COPYING (no vendored code)
docs/                   the seven documents
tests/                  CTest suite
```
