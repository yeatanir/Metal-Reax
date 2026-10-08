# ReaxMetal

A native Apple **Metal** GPU backend for a general-purpose **ReaxFF** molecular-dynamics engine, validated
against a pinned LAMMPS release. Element-agnostic: ordinary ReaxFF `ffield` files are read at runtime and no
chemical system is built in.

**Current status: Milestone M0 (audit, specification, decisions, skeleton). No physics is implemented.**
Nothing in this repository computes an energy, force or charge yet, and nothing has been run on a GPU.

| Where to look | What it is |
|---|---|
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

## Reproduce the audit input
```
tools/fetch_lammps.sh /path/to/new/empty/dir     # shallow sparse clone, verifies commit, tag and 72 file hashes
```

## License
Provisional; see `docs/ARCHITECTURE_DECISIONS.md` ADR-010 and `THIRD_PARTY_NOTICES.md`. A top-level `LICENSE`
file is intentionally absent until the owner decides.
