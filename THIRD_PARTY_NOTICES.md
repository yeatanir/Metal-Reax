# Third-party notices

No third-party **source code** is included in this repository at M0. What is included or referenced:

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
*(none yet)*
