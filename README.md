# ReaxMetal

A native Apple **Metal** GPU backend for a general-purpose **ReaxFF** molecular-dynamics engine, validated
against a pinned LAMMPS release. Element-agnostic: ordinary ReaxFF `ffield` files are read at runtime and no
chemical system is built in.

**Current status: M0, M0.5, M1 (LAMMPS reference oracle) and M2 (ffield/control parser, tables with compat flags, adapter A1: parse + `extract()` + host checks, no force backend). No ReaxFF physics is implemented.**
Nothing in this repository computes an energy, force or charge yet, and nothing has been run on a GPU. The target is a **LAMMPS plugin `pair_style reaxff/metal`** (LAMMPS provides integrators, thermostats, minimisers, ghosts); there is no standalone MD engine.

| Where to look | What it is |
|---|---|
| `docs/LAMMPS_INTEGRATION.md` | **M0.5**: plugin mechanism, pair-style/ghost/neighbor contract (measured), host/device split, memory & sync, EEM, strictness, smoke test |
| `docs/ENGINE_SPEC.md` | the functional form *as implemented by the pinned reference*, plus a catalogue of reference quirks |
| `docs/SOURCE_MAP.md` | pinned LAMMPS commit, license provenance, upstream-file → module map, papers |
| `docs/FEATURE_MATRIX.md` | what is planned, rejected, ignored (mirrored in `include/reaxmetal/capabilities.hpp`) |
| `docs/NUMERICAL_POLICY.md` | precision tiers, tolerance protocol, FP32 hazards found in M0 |
| `docs/VALIDATION.md` | test matrix (all NOT RUN except M0 bookkeeping tests) and results register |
| `docs/ARCHITECTURE_DECISIONS.md` | ADR log, repository layout |
| `docs/DEVELOPMENT_LOG.md` | what was actually done and run, including mistakes |

Reference: LAMMPS `stable_30Sep2026`, commit `8de817dd79bfe4525d5d39246a212d833e6dee07`, GPL-2.0.

## Build and test (Linux/macOS, CPU skeleton only)

```
cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
```
The Metal backend (`-DREAXMETAL_ENABLE_METAL=ON`) does not exist before M3 and is refused on purpose.

## LAMMPS plugin probe (Linux; opt-in, needs a stock pinned LAMMPS build)
```
tools/fetch_lammps.sh --full /path/lammps && tools/build_lammps_reference.sh /path/lammps /path/build /path/install
cmake -S . -B build -DREAXMETAL_BUILD_LAMMPS_PLUGIN=ON -DREAXMETAL_LAMMPS_SOURCE_DIR=/path/lammps/src -DREAXMETAL_LAMMPS_PREFIX=/path/install
cmake --build build -j && ctest --test-dir build --output-on-failure      # includes lammps_probe
```
macOS/Metal: **not yet attempted**; see `docs/LAMMPS_INTEGRATION.md` §11.

## M1 — LAMMPS reference oracle and fixtures (Linux; needs the pinned LAMMPS tree, python3 + numpy)
```
tools/m1_reproduce.sh <work-root>          # fetch pinned LAMMPS, build the compiler matrix, run 58 fixtures on every build,
                                           # equivalence / noise floor / conditioning / experiments / FD, then the M1 gate
python3 tools/reaxref/runner.py tests/fixtures/cases/cho_water.json --lmp <lmp> --ffield-dir <ff> --out <dir>   # one case
ctest --test-dir build                      # includes tolerances_frozen, fixtures_frozen, patch_frozen, reaxref_selftest
```
Results are recorded in `docs/VALIDATION.md` (M1 results) and `docs/DEVELOPMENT_LOG.md`; frozen thresholds in `tolerances/`.

## Reproduce the audit input
```
tools/fetch_lammps.sh /path/to/new/empty/dir     # shallow sparse clone, verifies commit, tag and all 142 audited-file hashes
```

## License
GPL-2.0-only (owner-approved). See `LICENSE`, `REUSE.toml`, `docs/ARCHITECTURE_DECISIONS.md` ADR-010, `THIRD_PARTY_NOTICES.md`
and the per-file upstream audit `third_party/lammps/LICENSE_AUDIT.tsv`.

## M2 — parser and adapter A1 (Linux; `ffield_tables` and `lammps_a1` are opt-in)
```
# the plugin must be built like the target LAMMPS (shared lib, LAMMPS_EXCEPTIONS; see docs/LAMMPS_INTEGRATION.md M2 addenda)
cmake -S . -B build -DREAXMETAL_FFIELD_DIR=<dir with the 11 bundled ffield.reax.* files> \
      -DREAXMETAL_BUILD_LAMMPS_PLUGIN=ON -DREAXMETAL_LAMMPS_SOURCE_DIR=/path/lammps/src -DREAXMETAL_LAMMPS_PREFIX=/path/install
cmake --build build -j && ctest --test-dir build --output-on-failure     # adds ffield_tables, lammps_probe, lammps_a1
build/reaxmetal_ffield_dump --sha256 <ffield>                              # canonical table hash
python3 tools/reaxref/parse_diff.py --lmp <instrumented lmp> --dump-tool build/reaxmetal_ffield_dump --ffield-dir <ff> --workdir <w>   # differential fuzz
```
`pair_style reaxff/metal` loads and parses, but **refuses to compute** until M4.
