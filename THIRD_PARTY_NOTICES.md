# Third-party notices

No third-party **source code** is included in this repository except one structural adaptation (a plugin class skeleton, listed under "Adapted files" below). What is included or referenced:

| Item | Where | License / notice |
|---|---|---|
| GNU General Public License v2 text (verbatim copy of LAMMPS `LICENSE`, sha256 `be38e38d9482c2beae35…`) | `third_party/lammps/COPYING` | GPL-2.0 |
| LAMMPS `stable_30Sep2026` @ `8de817dd79bfe4525d5d39246a212d833e6dee07` | **referenced only** (fetched by `tools/fetch_lammps.sh`, hashes in `third_party/lammps/SOURCE_HASHES.sha256`) | GPL-2.0 (Sandia Corporation notice; U.S. Government retains certain rights under DE-AC04-94AL85000) |
| PuReMD-derived ReaxFF physics sources inside LAMMPS `src/REAXFF/reaxff_*.cpp` | referenced only | © 2010 Purdue University; H. M. Aktulga, J. Fogarty, S. Pandit, A. Grama; GPL v2 "or (at your option) any later version" |
| Papers studied: Zheng, Li, Guo, J. Mol. Graph. Model. 41 (2013) 1–11 (GMD-Reax); Kylasa, Aktulga, Grama, *PuReMD-GPU*, Purdue CS TR 13-005 (2012) | `docs/SOURCE_MAP.md` | cited; no text or code copied |

## Obligations when code is adapted (from M2 on)
1. Keep the upstream copyright and license block verbatim at the top of the adapted file (template: `docs/NOTICE_TEMPLATE.txt`).
2. Add `Adapted-from: <upstream path> @ 8de817dd79bfe4525d5d39246a212d833e6dee07` and a summary of changes.
3. List the file in this document under "Adapted files".
4. Keep the SPDX identifier consistent with the upstream terms (see `docs/ARCHITECTURE_DECISIONS.md` ADR-010).

## Adapted files
| File | Adapted from (pinned commit 8de817dd…) | Upstream notice retained |
|---|---|---|
| `plugin/probe/pair_reaxff_metal_probe.h` | `examples/plugins/pair_morse2.h` (class skeleton only; GPL-2.0, Sandia Corporation notice) | yes (header comment) |

Other plugin files (`probe_plugin.cpp`, `pair_reaxff_metal_probe.cpp`, `plugin/CMakeLists.txt`) follow the documented plugin API
(`doc/src/Developer_plugins.rst`) and the structure of `examples/plugins/{morse2plugin.cpp,CMakeLists.txt}`; they link against and
derive from the GPL-2.0 LAMMPS `Pair` interface and are therefore GPL-2.0 (see `third_party/lammps/LICENSE_AUDIT.tsv`).

## Copyright of original files
Original ReaxMetal files: `SPDX-FileCopyrightText: 2026 Anirban Phukan`, `GPL-2.0-only`. Upstream notices in adapted files are kept verbatim and are **not** reassigned (ADR-018).

## M1 patches against LAMMPS
`third_party/lammps/patches/*.patch` are modifications of GPL-2.0 LAMMPS / PuReMD-derived source files (instrumentation hooks; two throwaway behaviour experiments). The patch text is a derivative work of those files and is distributed under their terms (GPL-2.0); the new file `reaxff_diag.h` in patch 0001 is original (SPDX header inside the patch). The patches are applied to a private clone at build time; no upstream source is vendored in this repository.
