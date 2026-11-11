// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Electron pair term against an exact external reference (plan task 3.2, step 4). Usage: test_electron_reference <ffield> <geo> <electron taper> <expected kcal/mol>
// ffield and geo are not stored in this repository (parameter/geometry files of the Jamil-target system); the expected value is the
// 'electron' energy printed at full precision by the rebuilt legacy executable (non-periodic, taper 10 A, step 0).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "reaxmetal/electron.hpp"
#include "reaxmetal/forcefield.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: %s ffield geo taper expected\n", argv[0]); return 2; }
  FfieldOptions fo; fo.ereaxff = true; fo.electron_taper = std::atof(argv[3]);
  const ForceField ff = read_force_field_file(argv[1], fo);
  std::ifstream in(argv[2]);
  std::vector<double> x; std::vector<int> type; std::vector<std::int64_t> tag;
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("HETATM", 0) != 0) continue;
    std::istringstream ls(line); std::string rec, name, sym; int idx; double px, py, pz;
    ls >> rec >> idx >> name >> sym >> px >> py >> pz;   // the file repeats the symbol: name, residue-type column
    const auto m = ff.match_element(name);
    RM_CHECK_MSG(m.size() == 1, "element " + name);
    if (m.size() != 1) return 1;
    type.push_back(m[0]); tag.push_back(idx); x.push_back(px); x.push_back(py); x.push_back(pz);
  }
  const Box box = Box::orthogonal({-100, -100, -100}, {200, 200, 200}, {false, false, false});  // non-periodic: no images
  ExpandOptions eo; eo.shell = 12.0; eo.wrap_owned = false;
  const AtomSet atoms = expand_images(box, x, type, tag, eo);
  NeighborCutoffs cut;
  const FarList far = build_far_list(atoms, cut);
  const ElectronResult r = compute_electron_core(ff, cut, atoms, far);
  const double want = std::atof(argv[4]);
  std::printf("atoms %zu pairs %zu  E_core_el %.12f  E_el_el %.12f  E_electron %.12f  expected %.12f  diff %.3e (rel %.3e)\n", type.size(), r.pairs,
              r.e_core_el, r.e_el_el, r.e_electron, want, r.e_electron - want, (r.e_electron - want) / want);
  RM_CHECK(std::fabs(r.e_electron - want) < 1e-12 * std::fabs(want));
  return rmtest::finish("electron_reference");
}
