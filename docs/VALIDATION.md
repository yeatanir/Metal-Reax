# Validation Plan and Results Register

**Rule:** a feature is "works" only when a test listed here has been run and its *actual* result recorded below.
At M0 the only executed tests are the build-skeleton tests (§1). Everything else is a **plan, status NOT RUN**.
References and tolerances are never edited to conceal a mismatch (NUMERICAL_POLICY §3).

## 1. Executed in M0 (actual results)

Environment: Linux x86-64 sandbox (the Metal backend cannot be built or run here — no Apple platform),
CMake 3.28.3, g++ 13.3.0, clang++ 18.1.3, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`,
`-ffp-contract=off -fno-fast-math` on the core library.

| Test (CTest name) | What it checks | Result |
|---|---|---|
| `pins` | compiled-in pins == `third_party/lammps/PIN.txt`; full-length SHAs; license hash present in manifest | **PASS** (g++ 13.3.0, clang++ 18.1.3; 2026-10-08) |
| `capabilities` | table integrity; Planned→`NotImplementedError`; Rejected→`UnsupportedFeatureError`; mandated rejections present; every energy term has a `term.*` row | **PASS** (both compilers) |
| `energy_terms` | 13 terms ↔ 14 LAMMPS pvector slots, each term mapped exactly once, over+under merge, conservation of sum | **PASS** (both compilers) |
| `docs_sync` | 7 mandated docs exist; every code feature appears in `FEATURE_MATRIX.md` with identical status & milestone | **PASS** (both compilers) |
| `tools/fetch_lammps.sh` (not a CTest) | fresh sparse clone reproduces pinned commit, tag object and 72 file hashes | **PASS** (11 s; "OK: stable_30Sep2026 @ 8de817dd… verified; 72 files match manifest") |

These tests validate **bookkeeping, not physics**. No energy, force or charge has been computed.
Their sensitivity was checked by mutation (9 deliberate corruptions, all detected; DEVELOPMENT_LOG M0).

## 2. Fixture protocol (M1, to be executed)

* Generator: pinned LAMMPS built from the pinned tree, `pair_style reaxff` (CPU) as primary oracle and
  `reaxff/kk` (double) as secondary; instrumented build (patch recorded + hashed) for BO/Δ/nlp dumps (Q1 in the M0 report).
* Every fixture directory holds: input geometry, ffield (hash), control settings, LAMMPS input, log, raw dumps,
  `MANIFEST.sha256`, generator commit, LAMMPS commit, compiler/flags, noise-floor runs (NUMERICAL_POLICY §3.2).
* Outputs recorded per fixture: 14 energy slots, per-atom forces (total), charges, (instrumented) BO tables.
* Fixture hashes are pinned in this repository; a fixture is regenerated only with a log entry.

## 3. Test matrix (all NOT RUN)

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
| GPU-1 | Metal-32 vs CPU-64 within `k·η₃₂`; decision mismatches listed | production backend | Metal | M3–M6 | NOT RUN |
| GPU-2 | atomic-accumulation option vs deterministic path: accuracy + speed | rule 9 | Metal | M8 | NOT RUN |
| NVE-1 | NVE drift vs pinned LAMMPS under identical protocol; two fixtures (organic, oxide) | M7 | CPU-64, Metal | M7 | NOT RUN |
| MIN-1 | minimiser convergence to stationary point; compare with LAMMPS `minimize` | M7 | CPU-64 | M7 | NOT RUN |
| LINT-1 | rejected-variant inputs raise `UnsupportedFeatureError` (acks2, qtpie, tabulate, efield…) | rule 4 | CPU | M2 | NOT RUN |

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
| — | none executed beyond §1 | | | | | | | |
