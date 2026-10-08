# Validation Plan and Results Register

**Rule:** a feature is "works" only when a test listed here has been run and its *actual* result recorded below.
At M0 the only executed tests are the build-skeleton tests (§1). Everything else is a **plan, status NOT RUN**.
References and tolerances are never edited to conceal a mismatch (NUMERICAL_POLICY §3).

## 1. Executed (M0 and M0.5; actual results)

Environment: Linux x86-64 sandbox (the Metal backend cannot be built or run here — no Apple platform),
CMake 3.28.3, g++ 13.3.0, clang++ 18.1.3, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`,
`-ffp-contract=off -fno-fast-math` on the core library.

| Test (CTest name) | What it checks | Result |
|---|---|---|
| `pins` | compiled-in pins == `third_party/lammps/PIN.txt`; full-length SHAs; license hash present in manifest | **PASS** (g++ 13.3.0, clang++ 18.1.3; 2026-10-08) |
| `capabilities` | table integrity; Planned→`NotImplementedError`; Rejected→`UnsupportedFeatureError`; mandated rejections present; every energy term has a `term.*` row | **PASS** (both compilers) |
| `energy_terms` | 13 terms ↔ 14 LAMMPS pvector slots, each term mapped exactly once, over+under merge, conservation of sum | **PASS** (both compilers) |
| `docs_sync` | 7 mandated docs exist; every code feature appears in `FEATURE_MATRIX.md` with identical status & milestone | **PASS** (both compilers) |
| `license_headers` (M0.5) | every authored file has SPDX; `Adapted-from:` files keep an upstream notice and are listed in `THIRD_PARTY_NOTICES.md`; `LICENSE` equals `LICENSES/GPL-2.0-only.txt` and upstream `COPYING` | **PASS** (g++, clang++; mutation-checked: removed SPDX, altered LICENSE, de-listed adapted file) |
| `compat_flags` (M0.5) | exact LAMMPS element/mass predicates, boundaries and exceptions | **PASS** (g++, clang++; 12 mutations caught, one initial "survivor" was a no-op sed — verified) |
| `tools/fetch_lammps.sh` (not a CTest) | fresh sparse clone reproduces pinned commit, tag object and all manifest hashes (72 at M0; 142 after M0.5) | **PASS** (M0: 72 files; re-run in M0.5 for both sparse and `--full` checkouts: "OK: stable_30Sep2026 @ 8de817dd79bfe4525d5d39246a212d833e6dee07 verified; 142 files match manifest") |

These tests validate **bookkeeping, not physics**. No energy, force or charge has been computed.
Their sensitivity was checked by mutation (9 deliberate corruptions, all detected; DEVELOPMENT_LOG M0).

## 2. Fixture protocol (M1, to be executed)

* Generator: pinned LAMMPS built from the pinned tree, `pair_style reaxff` (CPU) as primary oracle and
  `reaxff/kk` (double) as secondary; instrumented build (patch recorded + hashed) for BO/Δ/nlp dumps (Q1 in the M0 report).
* Every fixture directory holds: input geometry, ffield (hash), control settings, LAMMPS input, log, raw dumps,
  `MANIFEST.sha256`, generator commit, LAMMPS commit, compiler/flags, noise-floor runs (NUMERICAL_POLICY §3.2).
* Outputs recorded per fixture: 14 energy slots, per-atom forces (total), charges, (instrumented) BO tables.
* Fixture hashes are pinned in this repository; a fixture is regenerated only with a log entry.

## 3. Test matrix (unless a row says PASS/PARTIAL it is NOT RUN)

Legend: P = periodic, FD = finite difference, `q` = frozen charges.

| ID | Case | Purpose | Backend(s) | Milestone | Status |
|---|---|---|---|---|---|
| ELEM-1…n | one fixture per bundled ffield (`cho`, `rdx`, `lg`, `AB`, `FC`, `Fe_O_C_H`, `V_O_C_H`, `ZnOH`, `AuO`, `mattsson`, `budzien`) with element mixes drawn from each file | multiple element sets & parameterisations; vdw types 1 and 3; `gp[37]=2`; lg; compact torsion; metals | CPU-64 vs LAMMPS | M4–M6 | NOT RUN |
| PARSE-1 | parser reproduces LAMMPS-parsed tables (dump of `tbp/thbp/fbp/hbp` from the instrumented build) for all 11 files | M2 correctness | CPU | M2 | NOT RUN |
| PARSE-2 | crafted bad files: missing bond pair (Q-12), >5 angle sets (Q-09), vdw conflict, lg on type-1 | explicit rejection | CPU | M2 | NOT RUN |
| NBR-1 | cell-list neighbor lists with explicit image shifts equal brute-force enumeration (CPU) and equal CPU lists (Metal) | M3 | CPU, Metal | M3 | NOT RUN |
| PBC-1 | 7.54 Å cubic cell (smaller than 10 Å cutoff; the LAMMPS-test geometry) | multi-image correctness | all | M3–M6 | NOT RUN |
| PBC-2 | triclinic cell; non-periodic and mixed boundaries | M3 | all | M3–M6 | NOT RUN |
| REF-GHOST | LAMMPS energy/forces vs ghost cutoff (`comm_modify cutoff`) sweep | establishes the sufficiency condition for ENGINE_SPEC D-1; defines the reference-validity rule used by every fixture | LAMMPS | M1 | NOT RUN |
| ISO-1 | isolated atoms, isolated pairs near `bond_cut`; `enobonds yes/no` | `exp` ranges, atom terms with 0 bonds, FP32 NaN hazards (NUMERICAL_POLICY 4.1) | all | M4 | NOT RUN |
| OVER-1 | over-coordinated atoms (e.g. C with 5–6 neighbours at short distance) | over/under terms, `Δ` extremes, SBO piecewise regions | all | M4 | NOT RUN |
| THR-1…n | scans across each hard threshold: `BO'=bo_cut`, `r=bond_cut`, `BO=thb_cut`, `BO·BO=thb_cutsq`, `BO=0.01` (H-bond), `r=hbond_cut`, `nonb_cut` taper end, `Δe/2` integer crossing (`trunc`), SBO region boundaries | reproduce reference jumps; classify decision mismatches | CPU-64, CPU-32, Metal | M4–M6 | NOT RUN |
| QEQ-1 | converged-charge comparison, tolerance far below comparison tolerance | charges | CPU-64 vs LAMMPS | M5 | NOT RUN |
| QEQ-2 | LAMMPS production tolerance; iteration-count and stopping-rule match | solver semantics (Q-22) | CPU-64 | M5 | NOT RUN |
| QEQ-3 | **QEq failure**: `maxiter` too small / ill-conditioned (collapsed ions at very short distance) / net-charged system | non-convergence status surfaced (D-5); LAMMPS warns and continues | CPU, Metal | M5 | NOT RUN |
| QEQ-4 | history extrapolation (Q-23) over an MD trajectory; zero-history first step | M5/M7 | CPU | M5–M7 | NOT RUN |
| FD-q | FD vs analytical forces with **frozen charges**, per term and total, step sweep | proves analytical derivatives incl. BO chain rule | CPU-64 | M4–M6 | NOT RUN |
| FD-QEQ | FD with QEq re-solved vs analytical, and vs LAMMPS forces on the same frames | expected Q-03 deviation measured, not hidden | CPU-64, LAMMPS | M5–M6 | NOT RUN |
| FORCE-1 | total force sum = 0 (translation invariance) and torque/virial identities where applicable (P and non-P) | sanity | all | M6 | NOT RUN |
| FORCE-2 | deterministic reductions: run twice → bitwise identical forces/energies | rule 8 | CPU-64, Metal | M3+ | NOT RUN |
| FP32-1 | CPU-32 twin vs CPU-64 error envelope η₃₂ | derives Metal tolerance (NUMERICAL_POLICY 5) | CPU-32 | M4–M6 | NOT RUN |
| GPU-1 | Metal-32 vs CPU-64 against the **independent criteria of NUMERICAL_POLICY §5.2** (energy, force, charge, derivative, MD) **and** consistency with the CPU-32 envelope (§5.3); decision mismatches listed | production backend | Metal | M3–M6 | NOT RUN |
| GPU-2 | atomic-accumulation option vs deterministic path: accuracy + speed | rule 9 | Metal | M8 | NOT RUN |
| NVE-1 | **LAMMPS-hosted** NVE drift (`fix nve`) of `reaxff/metal` vs stock `reaxff` under the identical protocol; two fixtures; criteria NUMERICAL_POLICY §5.2 | M7 | CPU-64, Metal | M7 | NOT RUN |
| MIN-1 | **LAMMPS-hosted** `minimize` with `reaxff/metal` converges to the same stationary point as stock `reaxff` | M7 | CPU-64, Metal | M7 | NOT RUN |
| LINT-1 | rejected/deferred-variant inputs raise `UnsupportedFeatureError` (acks2, qtpie, tabulate, efield, non-neutral group, multi-rank…) | rule 4 | CPU | M2 | NOT RUN (the capability gate itself is tested: `capabilities`) |
| COMPAT-1 | element/mass compatibility predicates: boundaries, exceptions, exact equality (`tests/test_compat_flags.cpp`) | owner approval #2 | CPU | M0.5 | **PASS** (12 mutations caught) |
| INT-1 | **host-contract probe** (`tests/lammps/run_probe.py`): plugin load; ghost = owner + lattice shift; fold-back conservation; topology epoch; neighbor-request modes; fix qeq/reaxff & qeq/shielded vs stock; pvector; minimize; explicit errors | LAMMPS_INTEGRATION §10 | stock LAMMPS, probe plugin | M0.5 | **PASS** (summary failures=0; see register) |
| INT-2 | **in-LAMMPS A/B**: `pair_style reaxff` vs `reaxff/metal backend cpu64`, same input/ghosts/skin/QEq: 14 energy slots, total forces, virial | primary oracle (ADR-013) | CPU-64 | M4–M6 | NOT RUN |
| INT-3 | multi-rank run is refused with an explicit error (needs an MPI-enabled LAMMPS; the serial build used so far cannot test it) | C8 | adapter | M2 | NOT RUN — the check exists in the probe but was never executed with >1 rank |
| INT-4 | global virial/pressure: `reaxff/metal` vs stock `reaxff` (fdotr) on fixtures; then NPT stability via LAMMPS barostat | `lammps.virial_fdotr` | CPU-64, Metal | M6 | NOT RUN |
| INT-5 | adapter on macOS: plugin builds with Apple clang, loads in a macOS LAMMPS (`-undefined dynamic_lookup`), A0 probe passes | owner's M5 Max | probe | M2/M3 | NOT RUN (cannot be run in the Linux sandbox) |
| EEM-1 | strict convergence: non-convergence (small `maxiter`, ill-conditioned) → error/status by default; compat mode warns and continues; never silent; derived-fix residual verification | ADR-014/015 | CPU, Metal | M5 | NOT RUN |
| EEM-2 | `qeq/shielded` ≡ `qeq/reaxff` on production fixtures (probe system: 3.3e-15) incl. different taper/tolerances | Q-26 | stock LAMMPS | M1/M5 | PARTIAL — 1 system executed in INT-1 |
| EEM-3 | `extract(chi/eta/gamma)` equals stock `pair_style reaxff` extraction for all 11 bundled ffields and type maps incl. NULL | Q-29 | adapter | M2 | NOT RUN |
| REF-QEQ-CELL | stock `fix qeq/reaxff` vs dense explicit-image EEM: cubic cells (4 cases) executed; **triclinic and degenerate geometries** open | Q-27 | stock LAMMPS | M1 | PARTIAL — 4 cubic cases agree to ≤2.7e-14 |
| AMS-1 | agreement with AMS-ReaxFF EEM charges on a fixture | LAMMPS_INTEGRATION §7.2 | — | needs owner-supplied AMS reference | NOT RUN (no AMS data) |
| STAB-1 | stable `f1` (overcoordination): equivalence ≤1e-12 (FP64), FD derivative test over the full Δ′ range, FP32 finite at extremes; **value check** executed in M0.5 (`tools/stab_f1_check.py`) | NUMERICAL_POLICY 4.4 | CPU-64/32 | M4 | PARTIAL — values only (FP64 4.4e-16; FP32 finite 20000/20000 vs reference non-finite 6065/20000); derivatives NOT RUN |
| STAB-2…n | stable sigmoids (lone pair, over/under, f9, f11, f4/f5), log-sum-exp in `f3`, taper form: same four tests each | NUMERICAL_POLICY 4.4 | CPU-64/32 | M4–M6 | NOT RUN |
| MET-1 | first Metal run on the M5 Max: runtime shader compile without Xcode; device query; trivial kernel; buffer round trip | ADR-008 | Metal | M3 | NOT RUN |
| TOL-1 | tolerance file (§3 + NUMERICAL_POLICY §5.2) hashed and recorded **before** the first Metal evaluation | NUMERICAL_POLICY 5.4 | — | M1 | NOT RUN |

## 4. Known expected non-agreements (declared up front so they cannot later be mistaken for bugs)

1. Relaxed-charge FD vs analytical force differ by the Q-03 constant inconsistency (0.17 % between `14.4·23.02` and `C_ele`)
   plus the finite QEq tolerance — also present in LAMMPS.
2. NVE total energy is not conserved at the level of FP64 round-off in the reference itself (hard thresholds, QEq tolerance).
3. CPU-64 vs LAMMPS is *not* bitwise (NUMERICAL_POLICY §2.3).
4. Where an interaction sits within rounding of a hard threshold, FP32 and FP64 may disagree on inclusion; such pairs are
   reported as decision mismatches with the size of the associated jump.

## 5. Results register (append-only; one row per executed validation)

| Date | ID | Backend | Reference commit | Fixture hash | Result (numbers) | Tolerance (file hash) | Pass/Fail | Log ref |
|---|---|---|---|---|---|---|---|---|
| 2026-10-08 | INT-1 (a) fold-back | stock LAMMPS + probe | `8de817dd` | geometry of LAMMPS `atomic-pair-reaxff` (64 atoms, 7.54 Å) | Σfx = 4605 = nall; fx[1] = 64 = copies(tag 1), all 6 neighbor modes | exact | PASS | DEVELOPMENT_LOG M0.5 |
| 2026-10-08 | INT-1 (b) topology epoch (61 MD steps, `atom_modify sort 5`) | stock + probe | `8de817dd` | same | 0 order/nghost changes while `ago>0`; 12 reorders at rebuilds; ghost = owner+lattice shift on all 61 steps | exact | PASS | M0.5 |
| 2026-10-08 | INT-1 (c) charges: stock `reaxff` vs probe (`neigh` none/halfoff/halfoffghost/full) with `fix qeq/reaxff` | stock + probe | `8de817dd` | same, tol 1e-20 | max\|Δq\| = 0.0 (all 4); Σq = 1.2e-16 | bitwise | PASS | M0.5 |
| 2026-10-08 | INT-1 (d) `fix qeq/shielded` + probe vs stock `fix qeq/reaxff` | stock + probe | `8de817dd` | same | max\|Δq\| = 3.3e-15 | informational | PASS | M0.5 |
| 2026-10-08 | INT-1 (e) pvector / minimize / explicit errors | stock + probe | `8de817dd` | same | pvector[0]=1, [13]=14 via `compute pair`; `minimize` calls the style; 4 negative paths give explicit LAMMPS errors | exact | PASS | M0.5 |
| 2026-10-08 | REF-QEQ-CELL (cubic part) | stock `fix qeq/reaxff` vs dense explicit-image EEM | `8de817dd` | 2×2×2 and 3×3×3 cells, displaced and perfect | max\|Δq\| 1.9e-14 / 2.0e-14 / 2.7e-14 / 2.2e-14; minimum-image-only model off by 7.76 / 0.065 / 0.55 / 0.079 e | informational | PARTIAL | M0.5 |
| 2026-10-08 | STAB-1 (values only) | numpy, `tools/stab_f1_check.py` | n/a | 20 000 random Δ′ | see test matrix | — | PARTIAL | M0.5 |
