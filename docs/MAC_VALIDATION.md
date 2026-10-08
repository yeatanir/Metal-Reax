# Apple-machine validation checklist (milestone M3)

Status taxonomy (ADR-016): *written → compiled → executed → validated*, per backend. As of this document: **CPU side of M3 is validated on Linux; the Metal side is written only.**
Nothing below has been run. Every step records its own environment, so a failure is diagnosable from the report alone.

## Prerequisites (owner-reported machine, `LAMMPS_INTEGRATION.md` §11)
Apple Silicon Mac, Command Line Tools (no Xcode needed), CMake ≥ 3.24, `clang++` (Apple clang), this repository checked out on branch `claude/friendly-ride-rwz7xu`. No network access is needed.

## Steps (run in order; stop at the first failure and send me the report)
| Step | Command (repo root) | Answers | Expected on success |
|---|---|---|---|
| 1 | `tools/mac/step1_bringup.sh` | Does the Objective-C++ layer compile against the CLT SDK? Does `newLibraryWithSource:` work without Xcode? Are trivial kernels and buffer round trips exact? Is safe-math in effect? | `STEP 1 RESULT: PASS`; `MET-1a` lists the device/families; `MET-1e PASS` (NaN detected); `MET-1f` is informational (0 = no FMA contraction, 1.49e-08 = fused multiply-add contraction) |
| 2 | `tools/mac/step2_neighbor.sh` | Do device neighbor rows over owned+ghost atoms satisfy the superset/bounded contract against the CPU-64 list on 7 geometries (up to ~20 000 atoms)? Identical bytes across launches? Does grow-and-retry work? | `NBR-1 PASS` ×8, `FORCE-2 PASS` ×7, `STEP 2 RESULT: PASS`; GPU times are printed (informational) |
| 3 | `tools/mac/step3_reduce.sh` | Are the fixed-order reductions bitwise equal to the CPU twin for 45 (n, chunk) combinations, over 50 repeated launches, and on real row data? | `FORCE-2 PASS` ×4, `MET-4 PASS`, `STEP 3 RESULT: PASS` |

`--quick` (steps 2 and 3) skips the largest inputs. Reports go to `mac-reports/` (git-ignored). Exit codes: 0 pass, 1 fail, 3 "built without Metal" (never reported as a pass).

## What a failure most likely means (my guesses, to be replaced by facts)
* **Step 1 cmake/build fails:** the Objective-C++ was only ever checked against hand-written stub headers; expect typos or API mismatches. Send the first error.
* **`runtime compilation ... failed`:** the runtime compiler may be unavailable without Xcode, or my MSL has an error (the CPU emulation cannot see Metal-specific rules). The message will say which.
* **`MET-1e FAIL` (NaN not detected):** the safe-math option did not take effect; the reductions' bitwise claims then do not hold and I must change how the options are set.
* **Step 2 `missing > 0`:** a real kernel/host disagreement (the margin contract); **`band` pairs are not failures** (legal margin-band pairs, see ADR-024).
* **Step 3 mismatch:** the GPU does not execute the float adds in the order written (fast-math/reassociation) or flushes denormals; this would be a design problem for deterministic force reductions and is exactly what step 3 is for.

## Not yet prepared (will follow the results of steps 1-3)
* **Step 4 — INT-5:** build the pinned LAMMPS (serial, shared) and the plugin with Apple clang on macOS and run the probe, A1 and A2 tests. The existing Linux scripts use `sha256sum` and other GNU tools; I will adapt them once the Metal layer builds.
* GPU performance numbers are collected but are not acceptance criteria in M3.
