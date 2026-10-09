# ReaxMetal user manual

ReaxMetal is a LAMMPS plugin that evaluates the **ReaxFF** reactive force field on the Apple **Metal** GPU, in the same way and with the same numbers as the
`pair_style reaxff` of LAMMPS (pinned release `stable_30Sep2026`, commit `8de817dd…`). It adds two things to a LAMMPS installation and changes nothing else:

| Name | Kind | What it is |
|---|---|---|
| `reaxff/metal` | `pair_style` | the ReaxFF force field: all 13 energy terms, forces, virial, per-atom energy and virial |
| `qeq/reaxff/metal` | `fix` | `fix qeq/reaxff` with the charge matrix and (optionally) the whole solve on the GPU, plus a strict mode |
| `qtpie/reaxff/metal`, `qeq/rel/reaxff/metal` | `fix` | the stock QTPIE and QEq-R charge models for use with `reaxff/metal` |

Everything is validated against the stock LAMMPS style on the same input (see section 12).

---------------------------------------------------------------------------------------------------------------------------------------------------

## 1. Requirements

* An Apple Silicon Mac (tested: M5 Max, macOS 26) and the Xcode **Command Line Tools** (no full Xcode: the GPU kernels are compiled at run time).
* CMake ≥ 3.20, a C++20 compiler (Apple clang).
* The pinned LAMMPS source (`tools/fetch_lammps.sh --full <dir>`) built as a **shared library with `LAMMPS_EXCEPTIONS`** and the `REAXFF`, `QEQ` and `PLUGIN`
  packages; `tools/build_lammps_reference.sh` does exactly this. The plugin must be compiled like the LAMMPS it is loaded into (serial vs MPI).
* Units: `units real` (as the stock style: it does not convert).

## 2. Building

```
tools/fetch_lammps.sh --full /path/lammps
tools/build_lammps_reference.sh /path/lammps /path/build-lmp /path/install            # serial; add -DBUILD_MPI=on for an MPI LAMMPS
cmake -S . -B build -DREAXMETAL_ENABLE_METAL=ON -DREAXMETAL_BUILD_LAMMPS_PLUGIN=ON \
      -DREAXMETAL_LAMMPS_SOURCE_DIR=/path/lammps/src -DREAXMETAL_LAMMPS_PREFIX=/path/install \
      -DREAXMETAL_FFIELD_DIR=/path/lammps/potentials [-DREAXMETAL_LAMMPS_MPI=ON]
cmake --build build -j && ctest --test-dir build -j4        # the plugin is build/plugin/reaxmetaladapterplugin.so
```

Without `-DREAXMETAL_ENABLE_METAL=ON` (or off macOS) the plugin still builds and runs the double-precision CPU engine (`backend cpu64`).

## 3. Quick start

```
units           real
atom_style      charge
read_data       water.data
plugin load     /path/to/reaxmetaladapterplugin.so

pair_style      reaxff/metal NULL backend metal
pair_coeff      * * ffield.reax.cho O H
fix             q all qeq/reaxff/metal 1 0.0 10.0 1.0e-6 reaxff
fix             nvt all nvt temp 300 300 25.0
timestep        0.25
run             1000
```

`pair_style reaxff/metal` is used exactly like `pair_style reaxff`; `fix qeq/reaxff/metal` exactly like `fix qeq/reaxff`. An existing input needs three edits:
`plugin load …`, `reaxff` → `reaxff/metal` in `pair_style`, and `compute pair reaxff` → `compute pair reaxff/metal` if you use it.

## 4. `pair_style reaxff/metal`

```
pair_style reaxff/metal cfile keyword value ...
pair_coeff * * ffield El1 El2 ...            # one element (or NULL) per atom type
```

`cfile` is a ReaxFF control file or `NULL` (all defaults). Keywords (all optional):

| keyword | values | meaning |
|---|---|---|
| `backend` | `cpu64` (default) \| `metal` | `cpu64`: the double-precision reference engine on the CPU. `metal`: the GPU engine (FP32, deterministic). |
| `bonded` | `metal` (default) \| `cpu64` | only with `backend metal`: where the **bond-order terms** (bonds, lone pair, over/under, valence, torsion, H-bond) run. `bonded cpu64` runs them in double precision on the host while the non-bonded terms and the charges stay on the GPU ("mixed mode", section 7). |
| `checkqeq` | `yes` (default) \| `no` | `yes`: exactly one charge fix is required. `no`: charges are taken as they are (fixed, or zero). |
| `enobonds` | `yes` (default) \| `no` | lone-pair / under-coordination energy of atoms without bonds, as in the stock style. |
| `lgvdw` | `no` (default) \| `yes` | Grimme-type low-gradient vdW correction (needs a force field that carries its parameters). |
| `shellcheck` | `yes` (default) \| `no` | the plugin raises an error when the ghost shell (`comm_modify cutoff`) is narrower than `max(nonb_cut, hbond_cut, 2·bond_cut)`, because 3- and 4-body terms of owned atoms would then be incomplete. The stock style never checks. `no` turns the error into one warning. |
| `tabulate` | N ≥ 0 | accepted; **no table is built**, the non-bonded terms are evaluated analytically (a notice says so). A tabulated stock run differs from this by the table's interpolation error. |
| `safezone`, `mincap`, `minhbonds`, `list/blocking` | | accepted and ignored (memory-allocation heuristics / Kokkos option); a notice is printed. |

Control-file keywords (`nbrhood_cutoff`, `hbond_cutoff`, `thb_cutoff`, `thb_cutoff_sq`, `bond_graph_cutoff`, `tabulate_long_range`, `write_freq`, …) are read like the stock style.
Force fields: every ReaxFF `ffield` file the stock style reads (the 11 bundled in `potentials/` and the 39-general-parameter format with the ACKS2 columns).

**What the stock style does that `reaxff/metal` does the same way:** `compute pair` with 14 energy slots, `pe/atom` and `stress/atom` (virial, with atom-ID tallies of the reference),
the global virial and pressure, `fix npt`/`nph`, `minimize`, `hybrid` and `hybrid/overlay` (the example in `examples/reaxff/ci-reaxFF`), triclinic and shrink-wrapped boundaries, MPI domain decomposition, groups for the charge fix.

## 5. Charge models

Exactly one charge fix must be present (unless `checkqeq no`). All of the following reproduce the stock results (section 12):

| model | command | notes |
|---|---|---|
| EEM / QEq | `fix ID group qeq/reaxff/metal Nevery swa swb tol reaxff [keywords]` or the stock `qeq/reaxff` | The plugin fix builds the charge matrix on the GPU. Also accepts a parameter file instead of `reaxff`. |
| shielded QEq | stock `fix qeq/shielded` | |
| QTPIE | `fix ID group qtpie/reaxff/metal …` | use the `/metal` name: the stock fix borrows a neighbor list from `PairReaxFF` that a plugin cannot provide |
| QEq-R | `fix ID group qeq/rel/reaxff/metal …` | same reason |
| ACKS2 | stock `fix acks2/reaxff` | the plugin adds the polarization coupling and the bond-softness Coulomb terms to energy and forces |
| external field | `fix efield` together with any of the above | |

Keywords of `fix qeq/reaxff/metal` (after the stock arguments, e.g. `maxiter 400`):

* `strict` — CG non-convergence is an **error** (the stock fix warns and continues) and a taper radius larger than the ghost cutoff is an error (the stock fix silently truncates the matrix).
* `verify <eV>` — after every solve the equalisation residual `max_i |(H q)_i + χ_i − μ|` is evaluated and must not exceed `<eV>`.
* `resident` — the solve runs on the GPU (single rank): residual and matrix in double-single arithmetic, correction solves by an FP32 CG that stays on the device; charges equal the stock fix's to 1e-10 e. Not faster than the default at 24k atoms; use it when charge accuracy matters (e.g. with `verify`).

The GPU matrix is used with `backend metal`, fix group `all` and a taper radius ≤ `nonb_cut`; otherwise the stock CPU matrix is used with one warning.
A system with **one element** has exactly zero EEM charges; run it with `checkqeq no` and q = 0 (no charge fix, no charge solve).

## 6. Output

* `compute pp all pair reaxff/metal` gives the 14 slots of `compute pair reaxff`: `eb ea elp emol ev epen ecoa ehb et eco ew ep efi eqeq`.
* `compute pe/atom`, `compute stress/atom … virial`: produced by the double-precision engine; **on a step where such a compute is evaluated, that step runs on the CPU engine even with `backend metal`**. Evaluate them only every N steps.
* `thermo` pressure, `fix npt/nph/box/relax`: from the global virial (forces on owned + ghost atoms).
* `reaxmetal_selfcheck yes` (development) verifies the ghost view and the neighbor rows against LAMMPS' own list and stops.

## 7. Precision modes — what to use

| mode | how | accuracy against stock `reaxff` | use for |
|---|---|---|---|
| reference | `backend cpu64` | energy 3e-15 relative, forces 3e-12 kcal/mol/Å | validation, small systems, per-atom analysis |
| GPU | `backend metal` | energy ≤ 1e-4 kcal/mol/atom, force max 4e-3, RMS 9e-4 kcal/mol/Å; deterministic | **NVT, NPT, minimisation**, large systems |
| mixed | `backend metal bonded cpu64` | force max 2.7e-4, RMS 1.2e-4; energy conservation of stock | **NVE** where energy drift matters |

In NVE the all-FP32 GPU mode shows more total-energy noise than stock on a crystal (oxide test: fluctuation 4e-4 vs 1.1e-4 kcal/mol/atom, drift 4e-4 vs 4e-6 kcal/mol/atom/ps over 20 ps): the rounding of the bond-order derivatives in single precision is random, not conservative.
Thermostatted runs (NVT, NPT) agree with stock within statistical noise (tested on water and the oxide). The mixed mode passes the 20 ps NVE test (drift 3.4e-6 vs 3.7e-6) and is about 2× faster than stock on 24,000 atoms.
Nearly linear atom triples (angle within 0.06° of 180° or 0°) carry an intrinsic FP32 error growing like 1/sin²θ (observed 9e-2 kcal/mol/Å at sin θ = 1e-4); everywhere else the force error is ≤ 8e-3.

## 8. Dynamics

NVE, NVT (Nose–Hoover, Berendsen), NPT, `minimize` (cg, fire, …), `hybrid/overlay`, `fix efield`, `fix wall/reflect` and shrink-wrapped boundaries work. Time steps are the ones you would use with stock ReaxFF (0.25–0.5 fs).
Every discrete decision of the force field (bond-order cut-offs, three- and four-body gates, hydrogen-bond gate, SBO branches, the `trunc(Δe/2)` lone-pair crossing, the non-bonded cut-off) is taken identically by the GPU and the reference except within about 1e-7 Å of a threshold (section 12, CENSUS).

## 9. MPI

Build the plugin against an MPI LAMMPS (`-DREAXMETAL_LAMMPS_MPI=ON`) and run `mpirun -np N lmp -in …`. Ghost atoms whose owner is on another rank are used as LAMMPS delivers them. All ranks share the one GPU, so more ranks do **not** make a single-GPU run faster
(5,184 atoms: 0.75 s / 0.72 s / 0.84 s per 100 steps on 1 / 2 / 4 ranks). One process per GPU is the efficient setup.

## 10. Performance and memory

Measured on an M5 Max, serial LAMMPS, CHO water, NVT, 100 steps (idle machine), stock `reaxff` + `qeq/reaxff` versus `reaxff/metal` + `qeq/reaxff/metal`: 0.9× at 648 atoms, 3.5× at 5k, 7.2× at 24k, 10.5× at 66k atoms. Below about 1,000 atoms the GPU does not help.
Pure carbon (no charge solve), 30,000 atoms: about 25 ms per step. Memory: about 75 KB per atom (a 30,000-atom run uses 2.2 GB).
`REAXMETAL_PROFILE=1` prints per-phase wall times of the pair style and the charge fix at exit.

## 11. Limitations and differences from `pair_style reaxff`

1. FP32 rounding in `backend metal` (section 7). Use `bonded cpu64` or `backend cpu64` when you need stock-like NVE conservation or bit-level reproducibility against stock.
2. `pe/atom` / `stress/atom` steps run on the CPU engine (correct, slower).
3. The ghost shell is checked (`shellcheck`, section 4); the stock style runs silently with an incomplete shell.
4. `tabulate` evaluates analytically (differs from a tabulated stock run by the table error).
5. `newton pair on`, atom IDs and the `q` attribute are required (as stock). `units real` is assumed. No Kokkos/OpenMP suffix styles; `compute pair reaxff/metal` instead of `compute pair reaxff`.
6. The pair style does not write restart information (as stock): re-specify `pair_style`/`pair_coeff` after `read_restart`.
7. Analysis commands that read the stock style's internal bond lists — `fix reaxff/bonds`, `fix reaxff/species`, `compute reaxff/atom`, `compute spec/atom` — do **not** work with `reaxff/metal` unless the plugin-provided equivalents are loaded (section 13). Bond orders can always be obtained by running the same configuration with stock `pair_style reaxff` for analysis.
8. A system larger than the unified memory (≈ 800,000 atoms on a 64 GB Mac) cannot run; atom IDs must fit 32 bits for the GPU path.
9. Multi-rank runs share one GPU (section 9).

## 12. Validation

All results are in `docs/VALIDATION.md` (register) and `docs/DEVELOPMENT_LOG.md` (including the failures). The suite (`ctest`) contains, among others:

* `lammps_int2*` — 58 reference configurations: energy slots, forces, charges, pressure against stock (cpu64 under frozen limits 2e-9/1e-9; metal under 1e-3 kcal/mol/atom, 0.05 / 5e-3 kcal/mol/Å), also with 2 and 4 MPI ranks.
* `lammps_charge_models*` — every charge model, with and without electric field, on 3,000 atoms of water.
* `lammps_hybrid*`, `lammps_dense_carbon` (random dense carbon: the nearly-collinear-angle regression), `lammps_peratom*`, `lammps_min1*`, `lammps_eem`, `lammps_variants` (vdW type 2, `enobonds no`).
* `decision_census`, `threshold_scan`, `fp32_envelope`, `metal_no_leak`, emulation tests of the kernels.
* Long protocols (not in `ctest`): `tests/lammps/run_nve.py` (NVE 20 ps; `--nvt`, `--npt`).

## 13. Troubleshooting

| message | meaning / remedy |
|---|---|
| `ghost shell too narrow … use comm_modify cutoff` | add `comm_modify cutoff 10.5` (≥ `2·bond_cut`), or `shellcheck no` |
| `requires use of exactly one of the fix qeq/reaxff …` | a charge fix is missing; or add `checkqeq no` for fixed charges |
| `Pair style reaxff/metal: backend metal requested but this plugin was built without Metal` | rebuild with `-DREAXMETAL_ENABLE_METAL=ON` or use `backend cpu64` |
| `Fix qeq/reaxff/metal strict: CG did not converge …` | raise `maxiter` or loosen the tolerance, or drop `strict` |
| `Fix qeq/reaxff/metal verify: equalisation residual …` | the FP32 matrix cannot meet that residual: use `resident`, or a larger bound |
| `Unrecognized … reaxff/atom` / `Cannot use fix reaxff/bonds without pair_style reaxff` | the stock analysis styles need the stock pair style (limitation 7) |
| shaders fail to compile | the Metal compiler of the installed macOS rejected a kernel: report the message (it is printed in full) |

## 14. Environment variables

`REAXMETAL_PROFILE=1` phase timers; `REAXMETAL_STAGE_PROFILE=1` stage timers of the CPU-64 bonded engine; `REAXMETAL_DEBUG_CPU_NB` / `REAXMETAL_DEBUG_CPU_BONDED` evaluate the non-bonded / bonded terms on the CPU (diagnostics).

## 15. Worked example: random carbon at 1 g/cm³, 4000 K

`examples/graphitization/` packs 30,000 carbon atoms at random into a 84.265 Å cubic box (1.000 g/cm³), relaxes the worst overlaps by a short minimisation and runs NVT at 4000 K for 0.1 ns with the "2013 C" ReaxFF of Srinivasan, van Duin and Ganesh
(J. Phys. Chem. A 119, 571 (2015)); the force field is extracted from the paper's Supporting Information (`make_ffield.sh`, parameters are not redistributed):

```
examples/graphitization/make_ffield.sh jp510274e_si_001.pdf ffield.reax.C2013
python3 examples/graphitization/make_box.py c30000.data --n 30000 --density 1.0
lmp -in examples/graphitization/in.graphitization -var data c30000.data -var ffield ffield.reax.C2013 -var plugin $PWD/build/plugin/reaxmetaladapterplugin.so
```
