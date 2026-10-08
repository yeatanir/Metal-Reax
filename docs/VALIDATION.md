# Validation Plan and Results Register

**Rule:** a feature is "works" only when a test listed here has been run and its *actual* result recorded below.
M0/M0.5 executed bookkeeping and host-contract tests (§1); **M1 executed the reference-oracle measurements (§6, *M1 results*)**. Everything not marked PASS/PARTIAL below is a **plan, status NOT RUN**.
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

## 2. Fixture protocol (executed in M1)

* Oracle: pinned LAMMPS `reaxff` (serial CPU) built several ways; instrumented twin via the hashed patch `0001` (ADR-019). `reaxff/kk` (Kokkos-Serial, double) and MPI (1/2/4 ranks) were run as **noise-floor probes only**.
* A case = JSON (force field name + SHA-256, elements, cell, periodicity, atoms, charge model/settings, pair options); 58 cases in `tests/fixtures/cases/`, hashed in `FIXTURES.sha256`. Force-field files are **not** committed (fetched from the pinned tree and verified against `ffield_manifest.tsv`).
* A result is **valid** only if: no unexpected LAMMPS warning, QEq recursive residual ≤ requested tolerance (1e-12 / 1e-11 / 1e-10, `maxiter` 500–1000), the **independent** equalization residual ≤ 1e-8 eV, |Σq| ≤ 1e-9 e, exactly one force evaluation, and the parameter tables LAMMPS stored equal an independent parse of the file. Invalid results never enter `tests/fixtures/reference/` (58 golden JSONs, 380 kB, provenance + hashes).
* Recorded per fixture: total energy, the 14 pvector slots, the 13 `energy_data` fields, charges, forces, positions, QEq iteration counts and residuals, interaction tallies by ownership class, conditioning class.

## 3. Test matrix (unless a row says PASS/PARTIAL it is NOT RUN)

Legend: P = periodic, FD = finite difference, `q` = frozen charges.

| ID | Case | Purpose | Backend(s) | Milestone | Status |
|---|---|---|---|---|---|
| ELEM-1…n | one fixture per bundled ffield (`cho`, `rdx`, `lg`, `AB`, `FC`, `Fe_O_C_H`, `V_O_C_H`, `ZnOH`, `AuO`, `mattsson`, `budzien`) with element mixes drawn from each file | multiple element sets & parameterisations; vdw types 1 and 3; `gp[37]=2`; lg; compact torsion; metals | CPU-64 vs LAMMPS | M4–M6 | **Reference side PASS (M1)** — all 11 force fields, 10 elements, 58 fixtures; engine comparison NOT RUN |
| PARSE-1 | parser reproduces LAMMPS-parsed tables (canonical `%.17g` dump of every table, from the instrumented LAMMPS `params.txt`) for all 11 bundled files, `lgvdw` on/off as LAMMPS requires | M2 correctness | CPU | M2 | **PASS (M2)** — 11/11 files bit-identical to LAMMPS' stored tables (full dump and portable dump without the libm-dependent `gamma`); SHA-256 of both dumps in `tests/fixtures/ffield_tables.sha256` (CTest `ffield_tables`, opt-in via `REAXMETAL_FFIELD_DIR`) |
| PARSE-2 | crafted bad files: missing bond pair (Q-12), >5 angle sets (Q-09), non-finite numbers, truncation, vdw conflict, lg on type-1 | explicit rejection | CPU | M2 | **PASS for what is implemented (M2)** — unit tests (`ffield_parser`): Q-09 overrun, `inf`/`nan`, truncation, <38 general parameters, p_hbond truncation, duplicate/ordered 3- and 4-body entries, wildcard torsion order; Q-12 is rejected per *used* element pair at `pair_coeff` (`run_a1.py` E2: 10 of 66 type maps). **vdw-conflict and lg-on-type-1 diagnostics: NOT IMPLEMENTED** (LAMMPS' own checks are mirrored only as far as the parser reaches; not separately tested) |
| PARSE-3 | differential fuzz: seeded mutants (16 operators) of 6 bundled files through instrumented LAMMPS and `reaxmetal_ffield_dump`; both-accept ⇒ identical tables; ours-accepts-what-LAMMPS-rejects and unexplained rejections are failures | parser fidelity | CPU vs LAMMPS | M2 | **PASS (M2)** — 3 seeds × 400 = 1200 mutants: 471 accepted by both with identical tables; 601 rejected by both; 17 on which LAMMPS crashed and ours rejected; 90 accepted by LAMMPS but rejected by our strict mode (all in the documented classes: truncation, Q-09 overrun, non-finite number, <38 general parameters); 21 that LAMMPS rejects only after parsing (e.g. non-existent element) where ours accepts the file and rejects at `pair_coeff`; **0 problems** (DISAGREE / BAD-ACCEPT / BAD-REJECT) |
| INT-6 | adapter A1 host checks and refusal: exactly one qeq fix, newton on, `q` present, deferred fixes/keywords, missing control file, `compute()` raises an explicit error (never zero energy), plugin registers in a C++17 LAMMPS from g++ and clang++ builds | rule 4 | adapter | M2 | **PASS (M2)** — `run_a1.py` E3/E4; mutations caught: halving `eta` in `extract` (56 failures), removing the Q-12 check (10 failures, after strengthening the harness — it first survived) |
| NBR-1 | cell-list neighbor lists with explicit image shifts equal brute-force enumeration (CPU) and equal CPU lists (Metal) | M3 | CPU, Metal | M3 | NOT RUN |
| PBC-1 | 7.54 Å cubic cell (smaller than 10 Å cutoff; the LAMMPS-test geometry) | multi-image correctness | all | M3–M6 | **PASS for the reference (M1)** — supercell/translation invariance, see E-PBC; engine side NOT RUN |
| PBC-2 | triclinic cell; non-periodic and mixed boundaries | M3 | all | M3–M6 | **PASS for the reference (M1)** — triclinic, 2-D periodic, 1-D chains; engine side NOT RUN |
| REF-GHOST | LAMMPS energy/forces vs ghost cutoff (`comm_modify cutoff`) sweep | establishes the sufficiency condition for ENGINE_SPEC D-1; defines the reference-validity rule used by every fixture | LAMMPS | M1 | **PARTIAL (M1)** — shell 7.5…14 Å gives identical results (≤1e-12) for nonb_cut 5…10 on 3 systems; QEq `swb` beyond the shell silently truncates (Q-35). Long-bond force fields untested |
| ISO-1 | isolated atoms, isolated pairs near `bond_cut`; `enobonds yes/no` | `exp` ranges, atom terms with 0 bonds, FP32 NaN hazards (NUMERICAL_POLICY 4.1) | all | M4 | NOT RUN |
| OVER-1 | over-coordinated atoms (e.g. C with 5–6 neighbours at short distance) | over/under terms, `Δ` extremes, SBO piecewise regions | all | M4 | NOT RUN |
| THR-1…n | scans across each hard threshold: `BO'=bo_cut`, `r=bond_cut`, `BO=thb_cut`, `BO·BO=thb_cutsq`, `BO=0.01` (H-bond), `r=hbond_cut`, `nonb_cut` taper end, `Δe/2` integer crossing (`trunc`), SBO region boundaries | reproduce reference jumps; classify decision mismatches | CPU-64, CPU-32, Metal | M4–M6 | NOT RUN |
| QEQ-1 | converged-charge comparison, tolerance far below comparison tolerance | charges | CPU-64 vs LAMMPS | M5 | **Reference side PASS (M1)** — independent explicit-image EEM residual ≤ 8.6e-10 eV on 58 fixtures |
| QEQ-2 | LAMMPS production tolerance; iteration-count and stopping-rule match | solver semantics (Q-22) | CPU-64 | M5 | NOT RUN |
| QEQ-3 | **QEq failure**: `maxiter` too small / ill-conditioned (collapsed ions at very short distance) / net-charged system | non-convergence status surfaced (D-5); LAMMPS warns and continues | CPU, Metal | M5 | NOT RUN |
| QEQ-4 | history extrapolation (Q-23) over an MD trajectory; zero-history first step | M5/M7 | CPU | M5–M7 | NOT RUN |
| FD-q | FD vs analytical forces with **frozen charges**, per term and total, step sweep | proves analytical derivatives incl. BO chain rule | CPU-64 | M4–M6 | **Reference side PASS (M1)** — fixed-q central FD of the pinned LAMMPS: 1.9e-5 max (42 well-conditioned non-quirk fixtures); exceptions are Q-33/Q-34 |
| FD-QEQ | FD with QEq re-solved vs analytical, and vs LAMMPS forces on the same frames | expected Q-03 deviation measured, not hidden | CPU-64, LAMMPS | M5–M6 | **Reference side measured (M1)** — relaxed-q deviation: median 0.18 % of F_rms, max 4.6 % (0.65 kcal/mol/Å) |
| FORCE-1 | total force sum = 0 (translation invariance) and torque/virial identities where applicable (P and non-P) | sanity | all | M6 | NOT RUN |
| FORCE-2 | deterministic reductions: run twice → bitwise identical forces/energies | rule 8 | CPU-64, Metal | M3+ | NOT RUN |
| FP32-1 | CPU-32 twin vs CPU-64 error envelope η₃₂ | derives Metal tolerance (NUMERICAL_POLICY 5) | CPU-32 | M4–M6 | NOT RUN |
| GPU-1 | Metal-32 vs CPU-64 against the **independent criteria of NUMERICAL_POLICY §5.2** (energy, force, charge, derivative, MD) **and** consistency with the CPU-32 envelope (§5.3); decision mismatches listed | production backend | Metal | M3–M6 | NOT RUN |
| GPU-2 | atomic-accumulation option vs deterministic path: accuracy + speed | rule 9 | Metal | M8 | NOT RUN |
| NVE-1 | **LAMMPS-hosted** NVE drift (`fix nve`) of `reaxff/metal` vs stock `reaxff` under the identical protocol; two fixtures; criteria NUMERICAL_POLICY §5.2 | M7 | CPU-64, Metal | M7 | NOT RUN |
| MIN-1 | **LAMMPS-hosted** `minimize` with `reaxff/metal` converges to the same stationary point as stock `reaxff` | M7 | CPU-64, Metal | M7 | NOT RUN |
| LINT-1 | rejected/deferred-variant inputs raise `UnsupportedFeatureError` (acks2, qtpie, tabulate, efield, non-neutral group, multi-rank…) | rule 4 | CPU | M2 | **PARTIAL (M2)** — `pair_settings` test: `tabulate N>0` (keyword and control file), unknown keywords, malformed values; `run_a1.py` E3: qtpie fix, newton off, no/two charge fixes, no `q`, multi-rank. NOT RUN: acks2, qeq/rel, efield, non-neutral group (the features exist only as fix-side checks that need physics) |
| COMPAT-1 | element/mass compatibility predicates: boundaries, exceptions, exact equality (`tests/test_compat_flags.cpp`) | owner approval #2 | CPU | M0.5 | **PASS** (12 mutations caught) |
| INT-1 | **host-contract probe** (`tests/lammps/run_probe.py`): plugin load; ghost = owner + lattice shift; fold-back conservation; topology epoch; neighbor-request modes; fix qeq/reaxff & qeq/shielded vs stock; pvector; minimize; explicit errors | LAMMPS_INTEGRATION §10 | stock LAMMPS, probe plugin | M0.5 | **PASS** (summary failures=0; see register) |
| INT-2 | **in-LAMMPS A/B**: `pair_style reaxff` vs `reaxff/metal backend cpu64`, same input/ghosts/skin/QEq: 14 energy slots, total forces, virial | primary oracle (ADR-013) | CPU-64 | M4–M6 | NOT RUN |
| INT-3 | multi-rank run is refused with an explicit error (needs an MPI-enabled LAMMPS) | C8 | adapter | M2 | **PASS for the probe (M1) and for adapter A1 (M2)** — OpenMPI build: np=1 reaches `compute()`, np=2 is refused in `init_style` (`run_a1.py` E5; np=4 only for the probe) |
| INT-4 | global virial/pressure: `reaxff/metal` vs stock `reaxff` (fdotr) on fixtures; then NPT stability via LAMMPS barostat | `lammps.virial_fdotr` | CPU-64, Metal | M6 | NOT RUN |
| INT-5 | adapter on macOS: plugin builds with Apple clang, loads in a macOS LAMMPS (`-undefined dynamic_lookup`), A0 probe passes | owner's M5 Max | probe | M2/M3 | NOT RUN (cannot be run in the Linux sandbox) |
| EEM-1 | strict convergence: non-convergence (small `maxiter`, ill-conditioned) → error/status by default; compat mode warns and continues; never silent; derived-fix residual verification | ADR-014/015 | CPU, Metal | M5 | NOT RUN |
| EEM-2 | `qeq/shielded` ≡ `qeq/reaxff` on production fixtures (probe system: 3.3e-15) incl. different taper/tolerances | Q-26 | stock LAMMPS | M1/M5 | PARTIAL — 1 system executed in INT-1 |
| EEM-3 | `extract(chi/eta/gamma)` equals stock `pair_style reaxff` extraction for all 11 bundled ffields and type maps incl. NULL | Q-29 | adapter | M2 | **PASS (M2)** — `run_a1.py` E1/E2 via the LAMMPS C library: 11 ffields × 7 type maps = 77 combinations → 56 bit-identical extracts, 11 rejected by both, 10 strict Q-12 rejections (all explained: the type map uses an element pair without a bond block), 0 mismatches (`lammps_a1` CTest) |
| REF-QEQ-CELL | stock `fix qeq/reaxff` vs dense explicit-image EEM: cubic cells (4 cases) executed; **triclinic and degenerate geometries** open | Q-27 | stock LAMMPS | M1 | **PASS (M1)** — triclinic, 2-D periodic, 1-atom cells and the cubic cells agree with the dense explicit-image EEM (≤8.6e-10 eV residual); `swb` > shell fails (Q-35) |
| AMS-1 | agreement with AMS-ReaxFF EEM charges on a fixture | LAMMPS_INTEGRATION §7.2 | — | needs owner-supplied AMS reference | NOT RUN (no AMS data) |
| STAB-1 | stable `f1` (overcoordination): equivalence ≤1e-12 (FP64), FD derivative test over the full Δ′ range, FP32 finite at extremes; **value check** executed in M0.5 (`tools/stab_f1_check.py`) | NUMERICAL_POLICY 4.4 | CPU-64/32 | M4 | PARTIAL — values only (FP64 4.4e-16; FP32 finite 20000/20000 vs reference non-finite 6065/20000); derivatives NOT RUN |
| STAB-2…n | stable sigmoids (lone pair, over/under, f9, f11, f4/f5), log-sum-exp in `f3`, taper form: same four tests each | NUMERICAL_POLICY 4.4 | CPU-64/32 | M4–M6 | NOT RUN |
| MET-1 | first Metal run on the M5 Max: runtime shader compile without Xcode; device query; trivial kernel; buffer round trip | ADR-008 | Metal | M3 | NOT RUN |
| TOL-1 | tolerance file (§3 + NUMERICAL_POLICY §5.2) hashed and recorded **before** the first Metal evaluation | NUMERICAL_POLICY 5.4 | — | M1 | **PASS (M1)** — `tolerances/tolerances.json` sha256 recorded below; `tolerances_frozen` CTest |

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

| 2026-10-08 | TOL-1 | n/a | `8de817dd` | `FIXTURES.sha256` | `tolerances/tolerances.json` frozen (hash in `TOLERANCES.sha256`) | C1 = ⌈10×floor⌉; C3 owner values | RECORDED | DEVELOPMENT_LOG M1 |

## 6. M1 results (all executed on Linux x86-64, 4 cores; g++ 13.3, clang++ 18.1, OpenMPI 4.1.6, Kokkos-Serial; CPU only — Metal/macOS **not run**)

**Gate record (written before the results, `tools/reaxref/m1_gate.py`, G1 split into G1a/G1b after seeing E-EQUIV — see DEVELOPMENT_LOG):**
G1a pristine tree + patch hash + instrumented ≡ stock **bitwise** (strict-IEEE builds); G1b FMA builds reported; G2 every fixture valid; G3 coverage; G4 noise floor ≥3 builds + frozen file; G5 periodic accounting; G6 Q-09/Q-12; G7 docs + pushed.

**E-EQUIV (instrumentation).** gcc and clang (no FMA): instrumented = stock on **58/58 fixtures bitwise**, diagnostics on and off (energy, 14 slots, charges, forces). gcc vs clang stock builds are themselves bit-identical (58/58). `-march=native -ffp-contract=fast` builds: instrumented ≠ stock bitwise in 12 (gcc) / 11 (clang) fixtures, max |dF| 1.1e-13 / 1.3e-13, energies bitwise equal — compiler fusion decisions depend on surrounding code; such builds are noise-floor probes only.

**E-NOISE (reference noise floor).** 10 stock builds (gcc, clang, gcc-native, clang-native, gcc -O0, gcc -O2 `-ffp-contract=off`, Kokkos-Serial `reaxff/kk`, OpenMPI np=1/2/4; `cho_diamond_1x1x1` excluded because LAMMPS itself warns for np≥2), all pairs, 43 fixtures well-conditioned under the pre-registered definition **plus** proposed X1/X2 (set B):
max |ΔE_slot| 9.0e-10 kcal/mol (relative to max(1,|E|) 1.8e-10), |ΔE|/N total 3.4e-12, max |Δq| 8.8e-12 e, max |ΔF| component 9.1e-11, RMS 4.2e-11 kcal/mol/Å. Under the pre-registered definition alone (51 fixtures) force noise is **1.5e-2** (exactly collinear water dimers, Q-33) — hence the proposed amendment (ADR-020). Frozen C1 thresholds (⌈10×⌉): energy slot 2e-9 (rel to max(1,|E|)), total 4e-11 /atom, charge 9e-11, force component 1e-9, RMS 5e-10. Valid for ≤216 atoms.

**E-NB (independent non-bonded reference).** vdW (incl. inner wall, lg), Coulomb and E_pol from an independent explicit-image, unique-pair numpy implementation: worst relative difference **8.3e-15**, pair counts equal in **58/58** fixtures (160 835 pairs in total, including self-image pairs).

**E-QEQ.** All 57 QEq fixtures (58 minus the fixed-charge one): independent equalization residual ≤ 8.6e-10 eV, max |Σq| 2.0e-15, CG 2–40 iterations. LAMMPS' own regression system at tolerance 1e-20 reaches 4.2e-21 in 57/53 iterations with no warning (Q-36). Charged *periodic* cells (water box, graphene, triclinic diamond, polyethylene chain, CH₄ in 6 Å) all agree; single-element crystals have q ≈ 1e-15 and are vacuous for QEq.

**E-PBC (periodic-image accounting; ENGINE_SPEC §3.1).** Supercell invariance of energy per cell / charges / forces: diamond (1³…3³) 1.5e-10 / 1.4e-12 / 7.7e-12; graphene (to 4×4) 2.7e-12; Au fcc 3.6e-12; one C per 1.30 Å period (to 8 cells) 2.3e-13; zig-zag chain 5.4e-13; CH₄ in 6 Å 1.9e-11 / 8e-12 / 5e-11; polyethylene chain 3e-12; water dimer in 9 Å 4e-12. **Exceptions (both explained by experiment):** one H₂O in 6.2 Å (0.308 kcal/mol) and a dimer in 7.0 Å (0.211): the H-bond with an image of the donor is dropped (Q-32); with EXP-0001 (`i != k`) both agree to 1e-12. Rigid translation by 5 fractions of the cell: ≤1.8e-11 energy, 8e-13 charge, 4.7e-11 force.

**E-GHOST.** Skin 0…2 Å: identical results. Shell 7.5…14 Å (nonb_cut 5, 6, 8, 10): ≤3.3e-11 on 3 systems. QEq taper radius beyond the shell (water box): residual 1.2e-4 eV at 12.5 Å, 2.4e-2 eV at 14 Å, **no warning** (Q-35).

**E-FD (CPU reference forces vs central FD of the pinned energy, fixed charges, h = 1e-5 and 2e-5 Å).** 42 well-conditioned non-quirk fixtures: max |ΔF| 1.9e-5 kcal/mol/Å (F_rms ≥ 4); 1 step-inconsistent component (diamond 2×2×2). Failures, all explained: exactly collinear dimers 9.0e-3 / 1.3e-2 (Q-33); SO₂/mattsson **16.5** (Q-34, confirmed by EXP-0002 → 2.9e-7). **E-FD-2:** perturbed ethane showed FD error 1.6 kcal/mol/Å at h = 1e-4 Å (0.19 at 1e-3, 2e-6 at 1e-5): an energy jump ≈3.3e-4 kcal/mol where a BO product crosses `thb_cutsq` (Q-37) — the reason for the two-step-size consistency test. Relaxed-charge deviation from LAMMPS forces: median 0.18 % of F_rms (the Q-03 prediction was 0.17 %), max 4.6 %.

**E-QUIRK.** Q-09: raw `cnt` doubles per set; the zero slot is inert; 3 sets ⇒ `cnt` = 6 and the angle energy is 200.5 instead of 11.9 kcal/mol (read past `prm[4]`). Q-12: a removed bond block gives BO′ = (1,1,1) (C–O, C–C) or (1,0,0) (H–O) at every r ≤ 4.9 Å. Q-32/Q-34: see above.

**E-MPI.** OpenMPI build, probe plugin: np=1 runs; np=2 and np=4 abort with "supports a single MPI rank only". Stock `reaxff` np=1/2/4 and `reaxff/kk` are inside the noise floor above. (Ubuntu's MPICH is PMIx-only and starts singleton ranks in this sandbox — an MPICH attempt first gave a false "pass" that I discarded.)

**Not run / unverified:** anything on macOS/Metal; the engine itself (no physics exists yet); Kokkos OpenMP/GPU; AMS comparison; `qtpie`/`qeq/rel`; multi-rank *physics* beyond the noise-floor probes; force fields with long bonds for the ghost-shell rule; fixtures >216 atoms for the floor.
