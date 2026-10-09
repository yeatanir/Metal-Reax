# ReaxMetal

A native Apple **Metal** GPU backend for **ReaxFF** molecular dynamics, delivered as a **LAMMPS plugin**
(`pair_style reaxff/metal` and `fix qeq/reaxff/metal`) and validated against a pinned stock LAMMPS release.
Element-agnostic: ordinary ReaxFF `ffield` files are read at runtime and no chemical system is built in.
LAMMPS provides the integrators, thermostats, barostats, minimisers and neighbor/ghost machinery; there is no standalone MD engine.

## Status (Apple M5 Max, macOS, serial LAMMPS)

The complete ReaxFF force field (all 13 energy terms, forces, global virial) runs on the Metal GPU inside LAMMPS, together with a GPU charge-equilibration fix.

| Area | State |
|---|---|
| CPU-64 engine (double precision reference) | matches pinned LAMMPS to ~3e-15 relative in energy and ~3e-12 in force on 58 fixtures (frozen limits 2e-9 / 1e-9) |
| Metal-32 engine (FP32 kernels, no atomics, deterministic) | in-LAMMPS A/B vs stock on 58 fixtures passes the owner-set C3 criteria (PE ≤ 1e-4 kcal/mol/atom, force max 4e-3 / RMS 9e-4 kcal/mol/Å); two launches are bitwise identical |
| GPU QEq (`fix qeq/reaxff/metal`) | EEM matrix and matvec on the GPU, stock double-precision CG; charges within 1.6e-5 e of stock |
| Minimisation (MIN-1) | same minimum as stock on 9 perturbed fixtures |
| NVT (water 648 atoms, VO oxide 512 atoms, 40 000 steps) | ⟨T⟩, ⟨PE⟩, ⟨P⟩ agree with stock within block σ |
| NPT (water, 40 000 steps) | ⟨T⟩, ⟨V⟩, ⟨P⟩ agree with stock within block σ |
| NVE (20 ps) | water passes; **oxide shows more energy noise than stock** (FP32 bonded terms, drift +4e-4 vs +4e-6 kcal/mol/atom/ps). Owner decision: NVE drift is reported, not gated; prefer NVT |
| Per-atom energy / virial (`compute pe/atom`, `stress/atom`) | equal stock on 58 fixtures (≤ 5e-10 relative); produced by the CPU-64 engine's tallies, so a step that requests them runs on CPU-64 even with `backend metal` |
| Multi-rank MPI (2 and 4 ranks) | A/B vs stock on the same rank count passes on 58 fixtures (cpu64 under C1, metal under C3); the GPU QEq matrix is single-rank only (stock CPU matrix on several ranks) |
| CPU-32 twin / FP32 envelope | same CPU source compiled with float: E/atom 6e-5, force 4.7e-3 max / 2.2e-3 RMS vs CPU-64; satisfies C3 and Metal sits inside it |
| Speed vs stock `reaxff` (serial, CHO water, NVT) | 0.9× at 648 atoms, 3.5× at 5k, 7.2× at 24k, 10.5× at 66k atoms |

**Not implemented / not validated**: `fix acks2/qtpie/qeq/rel` and `efield`+QEq (rejected explicitly), vdW type 2 and `enobonds no` tests, a decision-mismatch census
for FP32 threshold crossings, a GPU charge matrix on several ranks, and speed measurements beyond serial CHO water. Metal NVE energy conservation is noisier than stock
on the oxide test (see above). The honest record of every run, including failures and the open decisions, is in `docs/VALIDATION.md` (results register)
and `docs/DEVELOPMENT_LOG.md`.

Reference: LAMMPS `stable_30Sep2026`, commit `8de817dd79bfe4525d5d39246a212d833e6dee07`, GPL-2.0.

## Use

```
plugin load /path/to/reaxmetaladapterplugin.so
pair_style reaxff/metal NULL backend metal          # backend cpu64 (default, double precision on the host) | metal
pair_coeff * * ffield.reax.cho C H O
fix q all qeq/reaxff/metal 1 0.0 10.0 1e-6 reaxff   # or the stock qeq/reaxff (CPU)
fix integ all nvt temp 300 300 25.0                 # any LAMMPS integrator
```
`fix qeq/reaxff/metal` uses the GPU only with `backend metal`, fix group `all` and a taper radius ≤ `nonb_cut`; otherwise it falls back to the stock CPU matrix with a warning.
`REAXMETAL_PROFILE=1` prints per-phase wall times at exit.

## Build and test (macOS, Apple Silicon)

Stock pinned LAMMPS is built once, outside the repository (shared, serial, `LAMMPS_EXCEPTIONS`):
```
tools/fetch_lammps.sh --full /path/lammps && tools/build_lammps_reference.sh /path/lammps /path/build /path/install
```
Then:
```
cmake -S . -B build -DREAXMETAL_ENABLE_METAL=ON -DREAXMETAL_BUILD_LAMMPS_PLUGIN=ON \
      -DREAXMETAL_LAMMPS_SOURCE_DIR=/path/lammps/src -DREAXMETAL_LAMMPS_PREFIX=/path/install \
      -DREAXMETAL_FFIELD_DIR=/path/lammps/potentials
cmake --build build -j && ctest --test-dir build -j4 --output-on-failure      # 38 tests (41+ with the MPI tree)
```
Shaders are compiled at run time from source (no Xcode needed; Command Line Tools suffice). Without `-DREAXMETAL_ENABLE_METAL=ON` (Linux, or macOS CPU-only) the CPU-64 engine, parser, neighbor code and reference tooling still build and test.

Multi-rank: build the pinned LAMMPS with MPI (`tools/build_lammps_reference.sh <src> <build> <install> -DBUILD_MPI=on`), configure a second tree with
`-DREAXMETAL_LAMMPS_MPI=ON` and that install prefix; `ctest` then adds `lammps_int2_mpi{2,4}_{cpu64,metal}` (needs `mpirun`).

Longer protocol runs (not in CTest):
```
python3 tests/lammps/run_nve.py --lmp <lmp> --plugin <so> --ffield-dir <ff> --case cho_water_box_8 --replicate 3 --steps 80000 --gpu-qeq          # NVE-1
python3 tests/lammps/run_nve.py ... --nvt --steps 40000      # NVT stability   (--npt for NPT)
python3 tests/lammps/run_min.py --lmp <lmp> --plugin <so> --ffield-dir <ff> --backend metal --gpu-qeq    # MIN-1
```

## Layout

| Where to look | What it is |
|---|---|
| `include/reaxmetal/terms.hpp` | pure ReaxFF term functions shared by CPU-64 and the Metal shaders |
| `src/cpu`, `src/neighbor` | CPU-64 engine, neighbor/ghost machinery, packing of device inputs |
| `src/metal` | Objective-C++ host and the MSL kernels (`reaxmetal_m3/m4/m6.metal`) |
| `plugin/adapter` | the LAMMPS plugin: `pair_style reaxff/metal`, `fix qeq/reaxff/metal` |
| `tests/` | unit tests, fixtures (58 cases, frozen hashes), in-LAMMPS A/B and dynamics harnesses |
| `tools/` | reference-oracle tooling, LAMMPS fetch/build scripts, Mac check scripts (`tools/mac`) |
| `docs/ENGINE_SPEC.md` | the functional form as implemented by the pinned reference, plus its quirks |
| `docs/NUMERICAL_POLICY.md` | precision tiers and tolerance protocol |
| `docs/VALIDATION.md` | test matrix and results register |
| `docs/ARCHITECTURE_DECISIONS.md`, `docs/DEVELOPMENT_LOG.md` | ADR log; what was actually done, including mistakes |
| `docs/FEATURE_MATRIX.md` | implemented / planned / rejected features (mirrors `src/core/capabilities.cpp`) |

## License
GPL-2.0-only. See `LICENSE`, `REUSE.toml`, `THIRD_PARTY_NOTICES.md` and the per-file upstream audit `third_party/lammps/LICENSE_AUDIT.tsv`.
