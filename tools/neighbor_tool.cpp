// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Command-line driver for the M3 neighbor code (used by tests/python/test_neighbor_fixtures.py):
//   reaxmetal_neighbor_tool [--ffield FILE | --nonb R] [--bond R] [--hbond R] [--shell R] [--lgvdw] [--ghosts] CASE.txt
// CASE.txt: lines "origin x y z", "a x y z", "b x y z", "c x y z", "periodic p0 p1 p2", "atoms N" then N lines "tag type x y z".
// Output (stdout, one item per line): nghost, entries, "vdw oo og self", optionally "ghost tag owner s0 s1 s2 x y z".
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/neighbor.hpp"

using namespace reaxmetal;

int main(int argc, char** argv) {
  std::string ffield, casefile;
  NeighborCutoffs cut;
  double shell = -1.0;
  bool ghosts = false, lgvdw = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "missing value after %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
    if (a == "--ffield") ffield = val();
    else if (a == "--nonb") cut.nonb = std::stod(val());
    else if (a == "--bond") cut.bond = std::stod(val());
    else if (a == "--hbond") cut.hbond = std::stod(val());
    else if (a == "--shell") shell = std::stod(val());
    else if (a == "--ghosts") ghosts = true;
    else if (a == "--lgvdw") lgvdw = true;
    else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    else casefile = a;
  }
  try {
    if (!ffield.empty()) { FfieldOptions fo; fo.lgvdw = lgvdw; cut.nonb = read_force_field_file(ffield, fo).file_control().nonb_cut; }
    if (casefile.empty()) { std::fprintf(stderr, "usage: reaxmetal_neighbor_tool [--ffield F|--nonb R] [--bond R] [--hbond R] [--shell R] [--lgvdw] [--ghosts] CASE.txt\n"); return 2; }
    std::ifstream in(casefile);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", casefile.c_str()); return 2; }
    Box box;
    std::vector<double> x;
    std::vector<int> type;
    std::vector<std::int64_t> tag;
    std::string line;
    std::size_t natoms = 0;
    while (std::getline(in, line)) {
      std::istringstream ss(line);
      std::string key;
      if (!(ss >> key)) continue;
      if (key == "origin") ss >> box.origin[0] >> box.origin[1] >> box.origin[2];
      else if (key == "a" || key == "b" || key == "c") { auto& v = box.vectors[static_cast<std::size_t>(key[0] - 'a')]; ss >> v[0] >> v[1] >> v[2]; }
      else if (key == "periodic") { int p[3]; ss >> p[0] >> p[1] >> p[2]; for (std::size_t d = 0; d < 3; ++d) box.periodic[d] = p[d] != 0; }
      else if (key == "atoms") {
        ss >> natoms;
        for (std::size_t k = 0; k < natoms; ++k) {
          std::int64_t t; int ty; double r[3];
          if (!(in >> t >> ty >> r[0] >> r[1] >> r[2])) { std::fprintf(stderr, "truncated atom list\n"); return 2; }
          tag.push_back(t); type.push_back(ty); x.insert(x.end(), r, r + 3);
        }
        break;
      }
    }
    ExpandOptions opt;
    opt.shell = shell > 0 ? shell : cut.required_shell();
    const AtomSet a = expand_images(box, x, type, tag, opt);
    a.validate(box);
    const FarList f = build_far_list(a, cut);
    const PairCounts c = count_nonbonded_pairs(a, f, cut);
    std::printf("nlocal %zu\nnghost %zu\nshell %.17g\nnonb %.17g\nentries %zu\nvdw %zu %zu %zu\n", a.nlocal, a.nghost(), opt.shell, cut.nonb, f.entries(), c.oo, c.og, c.self);
    if (ghosts)
      for (std::size_t g = a.nlocal; g < a.nall(); ++g)
        std::printf("ghost %lld %d %d %d %d %.17g %.17g %.17g\n", static_cast<long long>(a.tag[g]), a.owner[g], a.shift[g][0], a.shift[g][1], a.shift[g][2], a.x[3 * g], a.x[3 * g + 1], a.x[3 * g + 2]);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
