# Metal-Reax, eReaxFF branch

This branch (`ereaxff-lammps-merged`) adds **eReaxFF** to the `reaxff/metal` LAMMPS pair style: ReaxFF with explicit, pseudo-electron particles (atom type `El`) that carry a Gaussian charge and interact with the atom cores and with each other.
It exists so that eReaxFF can run inside LAMMPS, and therefore under [i-PI](https://github.com/i-pi/i-pi) for path-integral MD. The project that uses it is [PI-eReaxFF](https://github.com/yeatanir/PI-eReaxFF).

Everything else in the repository (the ReaxFF engine, the Metal GPU backend, the stock charge models) is unchanged from Metal-Reax. Its README is kept as [`README_REAXMETAL.md`](README_REAXMETAL.md).

## What was added

| | |
|---|---|
| Electron pair term | core–El and El–El terms with the 7th-order electron taper, re-implemented from the formulas of the legacy `electron` routine (no legacy code copied). `include/reaxmetal/electron.hpp`, `src/cpu/electron.cpp` |
| Force-field columns | the eReaxFF atom columns (`pxel2`, `vincr`, `alphag`, `betael`) and the bond column `pxel1` of the legacy `ffield` |
| Fixed charges | charges come from the data file (legacy `icharg 8`, the `MOLCHARGE` lines of a geo file). No charge fix, no polarization energy |
| 15th pvector slot | `eelect`, the electron energy, next to the 14 stock ReaxFF terms |
| Species table | `eparams` file: per-element core charge Z, sharing sign, and a `tested` flag |
| Safety checks | element allow-list, periodic-cell rule, ghost-shell check (below) |

## Use

```
plugin load /path/to/reaxmetaladapterplugin.so
pair_style reaxff/metal NULL ereaxff yes eparams species.eparams electron_taper 5.7
pair_coeff * * ffield Av El          # type 1 = Av (the Ag-like core), type 2 = El (the electron)
```
(No charge fix: the charges you read in with the data file are used as they are.)

| Keyword | Meaning |
|---|---|
| `ereaxff yes` | switch on eReaxFF mode. Needs `backend cpu64` (the default); `lgvdw` is not supported |
| `electron_taper <Å>` | electron-pair taper radius, must be > 0. Needs `ereaxff yes` |
| `eparams <file>` | per-element table, needs `ereaxff yes`. Without it the built-in legacy tables are used |

### The `eparams` file
One line per element: `SYMBOL Z SHARE_SIGN tested|untested`. Lines starting with `#` are comments. The built-in legacy tables are written out as an example in `tests/data/legacy_table.eparams`:
```
C   6 -1 tested
AV  3  1 tested
EL  0  0 tested
O   8 -1 untested     # refused
```
**An element is marked `tested` only after it passes the exact-reference frames and the finite-difference force checks.** An `untested` element is refused. Today the allow-list is **C, El and Av**. The silver-like core of the tunneling work is the species `Av` of the `Ag/Av` force field (`inputs/ag_el/ffield_Ag_El` in PI-eReaxFF).

## Rules the code enforces

* **Periodic cells** must have every edge at least `2*bond_cut + 1` Å. Below that the legacy single-minimum-image convention is not reproduced (we measured 4e-2 kcal/mol error at an edge of 10 Å and 5 kcal/mol at 8 Å against the exact reference), so the pair style stops with an error. Periodic triclinic cells are not supported.
* **Ghost shell** must be wide enough for the electron sharing range, checked at run time.
* **Hole particles** (`Ho`) are rejected, because the legacy code has extra branches for them that are not reproduced.
* **Two electron types** must carry identical `alphag` and `betael`, so the result does not depend on the order of a pair.

## What is validated

Reference values come from the rebuilt legacy `ereac` executable, evaluated on the same geometry.

| Check | Result |
|---|---|
| 60 C / 60 El periodic, one frame | total energy 9.6e-6 kcal/mol, terms 1.1e-9 relative, torsion 6.5e-7, forces 9.1e-8 relative |
| 51 `ereac` MD frames of the same system | all pass the gates (terms 1e-7, torsion 1e-6, force 2e-6); worst energy 1.3e-5 kcal/mol, worst term 7e-8 |
| Ag / El reference frames | run through LAMMPS with `tests/lammps/run_ereaxff.py --taper 5.7` |
| Unit tests | electron term, ffield columns, bonded terms, eparams, pair settings, `test_ereaxff_*` and `test_electron*` in `tests/` |

Run a case yourself:
```
python3 tests/lammps/run_ereaxff.py --lmp LMP --plugin reaxmetaladapterplugin.so --ffield F --geo GEO --frames frames.json --periodic
```
Short MD checks: `tests/lammps/run_ereaxff_md.py`; eparams: `tests/lammps/run_eparams.py`.

## Known limits

* **Torsion** carries a small stock-vs-legacy residual (largest in the early frames of the test set; about 1e-6 relative).
* **Energy jumps with moving heavy atoms.** A bond leaving the bond list at `bond_cut`, or the 3-/4-body threshold, makes the energy jump by 0.01 to 0.02 kcal/mol per crossing in the legacy code too. With the cores held fixed (as in the tunneling work) it does not occur. See `DISCONTINUITY.md` and `SMOOTH_ADOPTION.md` in PI-eReaxFF (`validation/phase3_engine/`).
* **CPU only.** The electron term and eReaxFF mode run on the double-precision CPU engine, not on the Metal GPU.
* **Coverage gaps.** Two electrons near one carbon, and an electron midway between two carbons, are tested only in the small standalone cases, not yet through LAMMPS.
* Each frame is a real eReaxFF reference, but that does not make the force field itself right for real silver or carbon. That is a separate question.

## Build

Same as Metal-Reax; see [`README_REAXMETAL.md`](README_REAXMETAL.md#build-and-test-macos-apple-silicon). A build without Metal (`-DREAXMETAL_ENABLE_METAL=OFF`) is enough for eReaxFF, since it runs on the CPU engine.

## License
GPL-2.0-only, as the rest of Metal-Reax. See `LICENSE`.
