# Metal-Reax, eReaxFF branch

This branch (`ereaxff-lammps-merged`) adds **eReaxFF** to the `reaxff/metal` LAMMPS pair style: ReaxFF with explicit, pseudo-electron particles (atom type `El`) that carry a Gaussian charge and interact with the atom cores and with each other.
It exists so that eReaxFF can run inside LAMMPS, and therefore under [i-PI](https://github.com/i-pi/i-pi) for path-integral MD. The project that uses it is [PI-eReaxFF](https://github.com/yeatanir/PI-eReaxFF).

Everything else in the repository (the ReaxFF engine, the Metal GPU backend, the stock charge models) is unchanged from Metal-Reax. Its README is kept as [`README_REAXMETAL.md`](README_REAXMETAL.md).

## What was added

| | |
|---|---|
| Electron pair term | core–El and El–El terms with the 7th-order electron taper, re-implemented from the formulas of the legacy `electron` routine (no legacy code copied). `include/reaxmetal/electron.hpp`, `src/cpu/electron.cpp` |
| Force-field columns | the eReaxFF atom columns (`pxel2`, `vincr`, `alphag`, `betael`) and the bond column `pxel1` of the legacy `ffield` |
| Fixed charges | default: charges come from the data file (legacy `icharg 8`, the `MOLCHARGE` lines of a geo file). No charge fix, no polarization energy |
| ACKS2 charges | `acks2 yes`: the charges of the mobile atoms are solved by ACKS2 inside the pair style (legacy `icharg 7`), electrons keep their data-file charge by default; which species are solved is a per-species setting in the `eparams` file, not an element name. See *ACKS2 charges* below |
| 15th pvector slot | `eelect`, the electron energy, next to the 14 stock ReaxFF terms |
| Species table | `eparams` file: per-element core charge Z, sharing sign, `tested` flag, and the ACKS2 settings `charge`, `polarization`, `electron` |
| Safety checks | missing core charge, periodic-cell rule, ghost-shell check (below); a species not validated against the legacy code only gives a warning |

## Use

```
plugin load /path/to/reaxmetaladapterplugin.so
pair_style reaxff/metal NULL ereaxff yes eparams species.eparams electron_taper 5.7
pair_coeff * * ffield Av El          # type 1 = Av (the Ag-like core), type 2 = El (the electron)
```
(No charge fix: the charges you read in with the data file are used as they are. For solved charges add `acks2 yes`, see below.)

| Keyword | Meaning |
|---|---|
| `ereaxff yes` | switch on eReaxFF mode. Needs `backend cpu64` (the default); `lgvdw` is not supported |
| `electron_taper <Å>` | electron-pair taper radius, must be > 0. Needs `ereaxff yes` |
| `eparams <file>` | per-element table, needs `ereaxff yes`. Without it the built-in legacy tables are used |
| `acks2 yes` | solve the charges with ACKS2 inside the pair style (default `no`: all charges from the data file, bit-for-bit the previous behaviour). Needs `ereaxff yes`, one MPI rank, no charge fix |
| `acks2_constants consistent\|legacy` | `consistent` (default) makes the forces the exact gradient of the energy (finite differences agree to 1e-7 relative). `legacy` reproduces the legacy executable (training energies, comparisons) but is not conservative, **do not use it for dynamics**; see below |

### The `eparams` file
One line per element: `SYMBOL Z SHARE_SIGN tested|untested [electron=yes|no] [charge=fixed|solve] [polarization=on|off]`. Lines starting with `#` are comments. The built-in legacy tables are written out as an example in `tests/data/legacy_table.eparams`:
```
C   6 -1 tested
AV  3  1 tested
AL  5  1 tested
EL  0  0 tested
O   8 -1 untested
```
* **Species data.** There is no element allow-list. A species needs a core charge Z: from its eparams line or from the built-in legacy znucel table (H ... Zn, Ag, Av, Lv, Os). An electron species needs nothing. A non-electron species without a core charge is an error with a message that names the missing data. `SHARE_SIGN 0` (the legacy code has no electron-sharing branch for the element) gives a warning.
* **`tested`.** C, El, Av and Al are validated against the legacy executable. Any other species runs, with a warning, unless its line says `tested`.
* **`electron=yes|no`** marks a species as an explicit electron. Default: the symbol is `El` (the legacy rule).
* **`charge=fixed|solve`** (with `acks2 yes`) keeps the data-file charge of the species or solves it. Default: `fixed` for an electron species, `solve` for everything else. `charge=fixed` on an atom species freezes it (as a test choice); `charge=solve` on the electron species gives the legacy *full* ACKS2.
* **`polarization=on|off`** (with `acks2 yes` and `acks2_constants legacy` only) includes or leaves out the coupling energy `q*s` of a solved species. Default: `off` for an electron species, `on` otherwise. With the default `consistent` constants the field is ignored and every solved species carries `q*s` (the energy must be variational). The legacy charge-energy sum leaves out El, Ho, Lv, Av and Li by name; to reproduce the legacy *full* ACKS2 of a system with Av add `AV 3 1 tested polarization=off`.

## ACKS2 charges (`acks2 yes`)

```
plugin load /path/to/reaxmetaladapterplugin.so
atom_style charge
fix mol all property/atom mol ghost yes            # optional: charge groups (below)
read_data data.lmp fix mol NULL Molecules
pair_style reaxff/metal NULL ereaxff yes eparams species.eparams electron_taper 5.7 acks2 yes
pair_coeff * * ffield Av El
```
* **What is solved.** Every species with `charge=solve` (default: all but the electron species). Their charges `q` and kinetic potentials `s` come from one dense linear system per force call (unknowns q, s, one multiplier for sum s = 0, one per charge group), re-implemented from the equations of the legacy `chargesa`. Fixed-charge particles (electrons by default) keep their data-file charge, stay out of the matrix, and their Coulomb field is folded into the right-hand side of the solved atoms (the legacy *compact* ACKS2). If every species is `solve`, the system is the legacy *full* ACKS2.
* **Charge groups** (the legacy `MOLCHARGE` lines): the solved atoms of each molecule id (`fix property/atom mol`, or `atom_style full` if built) form a group whose total charge is fixed to the sum of their data-file charges at the first force call. Without molecule ids all solved atoms form one group. Give an electron its own molecule id and `charge=solve` to pin it at its data-file charge inside the full system.
* **Energies.** Coulomb with the solved charges, charge energy `chi*q + eta*q^2` of the solved atoms (the 14th energy slot, "eqeq"), coupling `q*s`, and the bond-softness Coulomb term over all pairs (zero kinetic potential on fixed particles). Electron and bonded terms are unchanged.
* **`acks2_constants consistent` (default)**: the forces are the exact gradient of the energy (central finite differences agree to 1e-7 relative). Compared with the legacy executable it differs in three places: the bond-softness pairs between a solved and a fixed particle enter the solve (the legacy energy contains them but its solve does not), the energy carries the `-q_ref*s` term of the ACKS2 functional (zero with one charge group), and one Coulomb constant is used in matrix and energy. Charges and energies therefore differ from the legacy executable (up to 3e-2 e and several kcal/mol in compact Si clusters).
* **`acks2_constants legacy`** uses the numeric constants of the legacy executable and its exact pairing of terms, so charges, kinetic potentials, every energy term and the forces equal the legacy executable (validation table, run with `acks2_constants legacy`). Use it to reproduce training energies or to compare with the legacy code. The legacy forces are **not** the gradient of the legacy energy (3 % of the largest force in the Ag test cluster, 340 % in a small Si cluster), so do not run dynamics with it; a start-up notice says so.
* **Cost** (one force call, M-series CPU, one core): Ag/Av + El, 11 atoms: 0.012 ms with fixed charges, 0.047 ms with `acks2 yes`; Si cluster of 16 atoms: 0.015 / 0.055 ms; 432-atom periodic Si supercell: 30.6 / 34.4 ms (the dense solve of 434 unknowns takes 3.8 ms).
* **Limits.** One MPI rank (run one LAMMPS per bead). The system is solved directly (O(n^3), a few ms up to a few hundred solved atoms; see cost below). CPU `cpu64` only; the Metal GPU path is untouched. No per-atom energy/virial in eReaxFF mode.

## Rules the code enforces

* **Periodic cells** must have every edge at least `2*bond_cut + 1` Å. Below that the legacy single-minimum-image convention is not reproduced (we measured 4e-2 kcal/mol error at an edge of 10 Å and 5 kcal/mol at 8 Å against the exact reference), so the pair style stops with an error. Periodic triclinic cells are not supported.
* **Ghost shell** must be wide enough for the electron sharing range, checked at run time.
* **Hole particles** (`Ho`) are rejected, because the legacy code has extra branches for them that are not reproduced.
* **Core charge.** A non-electron species without a core charge Z (eparams or znucel table) is rejected.
* **Two electron types** must carry identical `alphag` and `betael`, so the result does not depend on the order of a pair.

## What is validated

Reference values come from the rebuilt legacy `ereac` executable, evaluated on the same geometry.

| Check | Result |
|---|---|
| 60 C / 60 El periodic, one frame | total energy 9.6e-6 kcal/mol, terms 1.1e-9 relative, torsion 6.5e-7, forces 9.1e-8 relative |
| 51 `ereac` MD frames of the same system | all pass the gates (terms 1e-7, torsion 1e-6, force 2e-6); worst energy 1.3e-5 kcal/mol, worst term 7e-8 |
| Ag / El reference frames | run through LAMMPS with `tests/lammps/run_ereaxff.py --taper 5.7` |
| ACKS2 charges (`acks2 yes`) | against the rebuilt legacy executables (compact and full `chargesa`): Ag/Av + El, C + El, Si (Al + El) cluster and 3x3x3 supercells. Charges 1e-12 to 3e-9 e, kinetic potentials 1e-12 to 4e-8 eV, every energy term 1e-12 to 4e-9 relative, forces 1e-10 to 6e-7 relative (`tests/lammps/run_acks2_ereaxff.py`, reference data in `tests/data/acks2`) |
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
