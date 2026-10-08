# Numerical Policy

Status at M0: **policy and protocol only; no number below is a measured result of this engine** unless it says
"computed in M0" (those come from `tools/fp32_hazards.py` or from reading the pinned sources).
Governing rules: scientific correctness before performance (rule 10); never change a reference or tolerance
to conceal a mismatch; deterministic reductions for validation (rule 8); atomics only as a benchmarked option (rule 9).

## 1. Precision tiers

| Tier | Scalar | Purpose | Status |
|---|---|---|---|
| **CPU-64** | `double` | reference backend; compared against pinned LAMMPS | M4–M6 |
| **CPU-32 twin** | `float`, same term functions and same traversal as CPU-64 | measures the *intrinsic* FP32 error of the algorithm, independent of any GPU | M4–M6 |
| **Metal-32** | `float` (Apple GPUs expose no FP64) | production GPU backend | M3–M8 |

Term functions (bond-order correction, angle, torsion, H-bond, pair, …) are written **once** as pure
inputs→outputs functions templated on the scalar type (ADR-005) and instantiated for CPU-64 and CPU-32;
the Metal kernels use the same formulas through a restricted-C++ header shared with MSL. Traversal,
accumulation order and parallel decomposition are backend-specific.

## 2. CPU-64 reference rules

1. Built with `-ffp-contract=off -fno-fast-math` (`cmake/ReaxMetalFlags.cmake`). Results must not depend on `-O` level.
2. Constants exactly as in the reference, **including truncated ones** (ENGINE_SPEC §1, Q-01/Q-02/Q-03). All
   constants live in one header with the upstream line reference beside each.
3. Expression order follows the reference where it is free to do so (taper by Horner, `pow` forms), but
   **bitwise equality with LAMMPS is not a goal**: LAMMPS is built with default flags (on FMA targets GCC
   contracts by default), MPI/OpenMP reductions reorder sums, and Kokkos uses `cbrt` where the CPU path uses
   `pow(x, 0.33333333333333)`.
4. Run-to-run **bitwise determinism** is required (single thread, fixed traversal order, fixed-order sums).
   Any future threaded CPU path must use fixed-order tree reductions and pass the same bitwise test.
5. Every sum is accumulated in `double` in the documented order; no compensated sum is needed in FP64.

## 3. Tolerances and the freeze protocol

### 3.1 Metrics (defined now so that no metric can be chosen after seeing data)
* Scalars (energies per category, total): `|Δ| ≤ atol + rtol·|ref|`.
* Vectors (forces, charges): per-component `|Δ| ≤ atol + rtol·|ref|` **and** a normwise bound
  `max_i |ΔF_i| / max(F_rms, F_floor)`. Pure per-component relative error (LAMMPS' own harness metric is
  `|a−b|/min(|a|,|b|)`, `unittest/force-styles/test_main.h:37-44`) is unusable for symmetry-zero components.
* Forces are compared in kcal/mol/Å, charges in e, energies in kcal/mol.

### 3.2 CPU-64 vs pinned LAMMPS — provisional, to be replaced by a measured floor
Provisional starting point: LAMMPS' own CI uses a relative epsilon of **2e-10** (QEq case, 1e-11…2e-10 across the
three base variants) × 5 for Kokkos. These are **placeholders**, not requirements.

Protocol (executed at M1 *before the first engine-vs-LAMMPS comparison*, results written to VALIDATION.md):
1. Run each fixture through LAMMPS built several ways (gcc and clang; `-O0`, `-O2`, `-ffp-contract=off`;
   1 and N MPI ranks; `reaxff` and `reaxff/kk` double). The spread is the **reference noise floor** `ε_ref`
   per quantity.
2. Set `rtol = atol_scaled = 10 × ε_ref` (factor 10 recorded and justified there), per quantity class.
3. The tolerance file is hashed and committed. Changing it requires a DEVELOPMENT_LOG entry with the measured
   error that motivated the change **and a physical or numerical explanation**. A mismatch is never closed by
   editing a tolerance or a reference fixture.

### 3.3 QEq-specific
Fixtures are generated with LAMMPS QEq tolerance far below the comparison tolerance (the LAMMPS unit tests use
`1.0e-20`, i.e. run to `maxiter` or stagnation), so the history-dependent initial guess (Q-23) cannot influence
charges beyond `ε_ref`. A second, separate test class compares **at LAMMPS' production tolerance** (e.g. 1e-6 …
1e-8) to verify the stopping rule, `√(r·M⁻¹r)/‖b‖`, and the iteration count (`fix qeq/reaxff` scalar output =
matvecs/2).

## 4. FP32 (CPU-32 twin and Metal-32)

### 4.1 Hazards identified in M0 (computed, `tools/fp32_hazards.py`)
1. **Exponent overflow in the bond-order correction.** `f2 = exp(−p_boc1·Δ'_i) + exp(−p_boc1·Δ'_j)` with
   `p_boc1 = 50` (e.g. `ffield.reax.cho`): FP32 overflows for `Δ' < −1.774` (FP64: `< −14.2`). An atom of valence 2
   with total `BO' < 0.23` already overflows; valence 4 (isolated carbon: `exp(200) = 7.2e86`) exceeds the FP32 maximum (3.4e38) by a factor ≈ 2·10⁴⁸.
   The expression `(val+f2)/(val+f2+f3)` then evaluates `inf/inf = NaN`. **FP32 requires an algebraically
   equivalent stable form** (e.g. `1/(1 + f3/(val+f2))` with exponent clamping and matching zero derivatives).
   This is not an exotic corner: weakly bonded atoms are the normal state near bond breaking.
2. **Lone-pair exponential** `exp(−75·Δlp)`: FP32 overflows for `Δlp < −1.183`; an isolated O has `Δlp = −1.0`
   (`exp(75) = 3.7e32`, 10⁻⁶ of FP32 max) — passes with almost no margin. Same class for `exp(p_ovun2·Δlpcorr)`,
   `exp(p_pen4·Δ)`, `exp(p_tor4(Δj+Δk))`. **Every `exp`/`pow` in ENGINE_SPEC §4–§5 gets an explicit FP32 range audit
   recorded in VALIDATION.md before its kernel is accepted** (deliverable of M4 for BO/atom terms, M6 for the rest).
3. **Taper polynomial.** Horner evaluation of the LAMMPS coefficient form has 9e-6 absolute error in FP32
   (swb=10; computed). The same polynomial in the scaled variable `x=(r−swa)/(swb−swa)`,
   `1 − 35x⁴ + 84x⁵ − 70x⁶ + 20x⁷`, equals the LAMMPS form to 3e-14 in FP64 for swa∈{0,0.5}, swb∈{8,10} (computed) and is
   the candidate FP32 form; its FP32 error is **not yet measured** (M5).
4. **Large sums.** Total energies are O(10⁵) kcal/mol for 10³ atoms (GMD-Reax Table 1: bond energy −2.4·10⁵ for 1378
   atoms); FP32 spacing at 2.4·10⁵ is 0.016 kcal/mol. A naive FP32 total is unacceptable.
5. **Hard thresholds** (ENGINE_SPEC §8): an interaction within one FP32 rounding of `BO=thb_cut`, `0.01`, `bond_cut`,
   `hbond_cut` may be included by one backend and excluded by the other. Validation therefore classifies
   **decision mismatches** (reported separately, bounded by the size of the jump) from **numerical error**.
6. **Coordinates.** FP32 absolute coordinates in a large box lose precision (spacing 7.6e-6 Å at 100 Å).
   Displacements are formed from wrapped coordinates plus **exact integer image shifts**; a hi/lo (double-float)
   position option is a M3 experiment to quantify sensitivity.

### 4.2 Accumulation and determinism on the GPU (default path)
* **No floating-point atomics in the default path.** Forces are gathered per atom over fixed-order lists
  (the engine keeps full/redundant bond lists à la PuReMD-GPU, ADR-006); coefficient arrays (`Cdbo`, `CdDelta`, …)
  are written by exactly one thread.
* Energies: per-atom (or per-interaction-block) FP32 partials written to a buffer; the final reduction is a
  **fixed-shape tree** (or host FP64 sum of the partial array in index order). Run-to-run bitwise reproducibility
  on the same device and OS/driver is a **tested requirement** (run twice, compare bits). Cross-device identity is not promised.
* The **atomic accumulation variant** (`backend.metal_atomic_accum`, M8) is selectable and benchmarked; it is
  expected to be non-deterministic and is excluded from validation runs. Whether Metal exposes FP32 atomic
  add on the target OS/GPU families is **to be verified at M3**, not assumed.
* Metal math-library mode: MSL compiles with fast-math **on** unless disabled (to be confirmed against the SDK
  at M3). Validation builds use precise math. `fast::`/fast-math is an M8 option evaluated with forces and drift,
  not just energies (GMD-Reax reported <0.01 % single-step *energy* deviation with fast SFU math; that is not evidence about forces).

### 4.3 QEq in FP32
Conjugate-gradient residuals stall at a floor set by `ε₃₂·κ(H)`. The target tolerance for Metal-32 cannot be
assumed to equal the FP64 value (PuReMD-GPU used 1e-10 in FP64). Candidate strategies, **to be decided at an M5
gate from measured data**: (a) FP32 CG with compensated dot products; (b) FP32 CG + FP64 residual refinement on the
host (cost: one matvec transfer per refinement); (c) matrix-free FP32 like Kokkos `matfree`. Charge error budget is
derived from the energy budget through `E_ele ∝ C_ele·q_iq_j` at the chosen force tolerance.

### 4.4 Stable formulations (owner requirement: fix unstable expressions algebraically; derivative-test them)
Every `exp`/`pow` identified by the §4.1 audit is replaced by an **algebraically equivalent** form that cannot overflow/underflow into NaN in FP32 over the whole physical domain. Each replacement is a named
variant `STAB-n` with four mandatory tests (VALIDATION): **(i)** equivalence with the reference expression in FP64 on the domain where the reference is finite, relative `≤ 1e-12` for values *and* derivative coefficients;
**(ii)** analytic-vs-central-FD derivative test in FP64 across the full physical range, including the former overflow region; **(iii)** FP32 evaluation is finite and within the twin envelope of the FP64 value at the extremes
(isolated atom `Δ′ = −val`, `Δlp = −1.2…`, `BO′ → bo_cut`); **(iv)** continuity/monotonicity where the reference is monotone. Known candidates:
* **Bond-order overcoordination factor `f1` (`p_boc1·Δ′` up to ±200).** With `a_x = exp(−p₁Δ′_x)` and `m = min(Δ′_i, Δ′_j)`: for `m ≥ 0` the reference form is already safe (`a ≤ 1`); for `m < 0` factor out the dominant exponential,
  `a_x = e^{−p₁m}·α_x`, `α_x = e^{−p₁(Δ′_x−m)} ≤ 1`, and write `(val+f2)/(val+f2+f3) = (val·e^{p₁m} + α_i+α_j)/((val+f3)·e^{p₁m} + α_i+α_j)` with `e^{p₁m} ≤ 1` (underflowing harmlessly to 0). The derivative
  coefficients `Cf1_ij, Cf1_ji` contain `a_i/u²`-type terms that are bounded when written through `ρ = a/(val+f2+f3) ∈ [0,1]`. *Derivative coefficients: sketch only; derived and tested in M4.*
  **Value check done in M0.5** (`tools/stab_f1_check.py`, `p₁=50`, `p₂=9.5469`, 20 000 random `(Δ′_i, Δ′_j)` with `Δ′ ∈ [−val, 4]` over five valence pairs): the **reference form is non-finite in FP32 for 30 % of the samples (6065/20000)**;
  the stable form is finite for all 20 000, equals the reference in FP64 to 4.4e-16, and its FP32 value agrees with the FP64 reference to 1.8e-7. (Values only; no derivative claim.)
* **Sigmoids** `1/(1+exp(±k·x))` (lone pair `k=75`, `exp_ovun2`, `exp_ovun8`, `f9`, `f11`, `f4/f5`): standard two-branch stable sigmoid (`x<0`: `e^{x}/(1+e^{x})`).
* **`log(0.5(e^{-p₂Δ′_i}+e^{-p₂Δ′_j}))` in `f3`**: log-sum-exp with the same factoring as above.
* **Taper**: scaled-variable form (§4.1 item 3), FP32 error to be measured.
The reference semantics inside the safe domain is unchanged; any intentional behaviour change outside it (e.g. saturation) is listed in ENGINE_SPEC §9 as a deviation with its justification.

## 5. Acceptance criteria for FP32 (revised in M0.5 — owner modification of the original "3× twin envelope" rule)

### 5.1 Three separate comparisons (never merged)
| ID | Comparison | Role |
|---|---|---|
| C1 | CPU-64 ↔ pinned LAMMPS (in-LAMMPS A/B, plus hashed fixtures) | validates the reference implementation (§3) |
| C2 | CPU-32 twin ↔ CPU-64 | **characterises** floating-point behaviour: error *distribution* (max, 99.9 %, 99 %, median; per atom/bond/term class) — a feasibility measurement, not an acceptance test |
| C3 | Metal-32 ↔ CPU-64 (and end-to-end Metal-32 ↔ stock LAMMPS via A/B) | **acceptance** of the GPU backend |

### 5.2 Independent acceptance criteria — fixed *before* any Metal result is evaluated
They are derived from **physical requirements** (the energy scale of the dynamics, the accuracy of the force field, conservation), **not** from the twin or GPU output.
Numbers below are **proposals for the owner's confirmation**; they are frozen together with §3 at the start of M1 (hashed file), and changed afterwards only with an
owner-approved log entry stating a *physical* reason.

| Class | Metric | Proposed bound | Rationale |
|---|---|---|---|
| Energy | per-category and total, `\|ΔE\|/N` (max over fixtures); for categories with `\|E_cat\|/N ≥ 1 kcal/mol` also relative `\|ΔE\|/\|E\|` | `≤ 1e-4 kcal/mol/atom` and `≤ 1e-6` relative | 1e-4 is 0.02 % of kT at 300 K (0.6 kcal/mol); keeps the NVE drift budget (§5.2 MD) reachable |
| Forces | RMS of `\|ΔF_i\|` over atoms; max component | `RMS ≤ 5e-4`, `max ≤ 5e-3 kcal/mol/Å` on fixtures with `F_rms ≥ 5 kcal/mol/Å` | force noise ≪ thermal force scale; dominates MD error accumulation |
| Charges | `max\|Δq\|`, RMS, `\|Σq\|` | `max ≤ 2e-6 e`, `RMS ≤ 5e-7 e`, `\|Σq\| ≤ 1e-5 e` | derived: `ΔE_ele ≈ φ·Δq` with `φ ≈ 50 kcal/mol/e` ⇒ `Δq ≲ 1e-4/50`; **to be re-derived from the measured potentials on the fixtures at M1** (the energy bound is the anchor) |
| Derivatives | (a) analytic vs central FD of the *same* precision's energy, FP64 only, best-`h` | per-component relative `≤ 1e-6` |
| | (b) intermediate coefficients (`Cdbo`, `Cdbopi`, `Cdbopi2`, `CdDelta`, `dBOp`) Metal-32 vs CPU-64 | L2 relative per class `≤ 1e-4`; no element worse than `1e-3` of the class scale |
| | (c) stable-formulation derivatives (§4.4) | FP64 analytic vs FD `≤ 1e-7` relative over the full physical Δ′ range |
| MD stability | LAMMPS-hosted NVE, ≥ 20 ps (8·10⁴ steps at 0.25 fs), two fixtures (organic, oxide) | linear energy-drift slope `≤ max(2 × slope of stock LAMMPS-FP64 under the identical protocol, 2e-4 kcal/mol/atom/ps)`; total-energy RMS fluctuation within 10 % of stock; mean `T` within 3σ (block averages); **no NaN/Inf**; run-to-run bitwise reproducibility of the default path |
| MD statistics | NVT run (LAMMPS thermostat) | `T` and potential-energy distributions equal to stock within statistical error; trajectory divergence is **not** a criterion (chaos) |
| Decisions | hard-threshold crossings (§4.1 item 5) | counted and listed; each mismatch's energy jump bounded by the analytic jump size; fraction of interactions within one FP32 rounding of a threshold reported |

### 5.3 How the CPU-32 twin is used
1. **Feasibility check**: if the twin itself violates a physical criterion of §5.2, that criterion is *not* relaxed silently; the response is (in order) a better formulation (§4.4), mixed
   precision for the offending accumulation/term (e.g. FP64 reductions, compensated sums), or an explicit owner decision recorded in the log.
2. **Consistency check** for Metal: results must satisfy §5.2 **and** be consistent with the twin's error distribution (proposal: within `3×` the twin's 99.9 % envelope per class);
   a Metal result that passes §5.2 but sits far outside the twin envelope is an *anomaly to be explained* (different `exp`/`pow` implementation, association order, race, buffer bug), not a pass by default.
3. Neither check can be satisfied by editing a tolerance after seeing Metal output.

### 5.4 Freeze rule
`tolerances/` (hashed) contains §3 and §5.2 numbers; a test fails if the hash recorded in VALIDATION.md differs. Evaluating a Metal run before the hash is recorded is a protocol violation.
Reference LAMMPS-Kokkos `mixed`/`single` is **not** used as an accuracy reference (Q-25).

## 6. Finite-difference and conservation checks (numerical meaning)

* FD forces use central differences with a step sweep (e.g. h ∈ {1e-5,…,1e-2} Å) and report the best-h error and
  the h-dependence; FP64 only for the reference; FP32 FD is meaningless below ≈1e-3 Å.
* **Two FD modes are mandatory** because of Q-03 (the unit conversions inside the reference make QEq charges not
  an exact stationary point of the reported energy): (i) **frozen charges** — analytical force must equal FD of
  the energy at fixed `q` to the FD truncation floor, term by term; (ii) **relaxed charges** — FD with QEq re-solved at
  every displaced geometry; the deviation from the analytical force is *expected* to be non-zero, is measured, and is
  compared with the same deviation computed from pinned LAMMPS forces on the same configurations. We do not "fix"
  the mismatch; we reproduce the reference's.
* NVE drift is compared **against pinned LAMMPS run with the identical protocol** (same dt, same QEq tolerance,
  same initial state, same ffield), because the reference has intentional discontinuities (ENGINE_SPEC §8) and a
  finite QEq tolerance. Criterion: our drift slope ≤ 1.5 × LAMMPS' (provisional, to be confirmed with the owner).
