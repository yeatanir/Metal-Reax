# Patches against the pinned LAMMPS tree
* `0001-reaxmetal-diagnostics.patch` — observation-only diagnostics (see ADR-019). Hash: `PATCHES.sha256`. Applied by `tools/build_lammps_instrumented.sh` to a private clone; diagnostics are inert unless `REAXMETAL_DIAG_DIR` is set.
* `experiments/EXP-000x-*.patch` — throwaway behaviour experiments, applied **on top of 0001**; builds from them are never used for fixtures or tolerances.
License: derivative of GPL-2.0 LAMMPS sources (GPL-2.0). `git apply --check` against the pristine pinned commit is part of the M1 gate.
