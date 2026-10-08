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
