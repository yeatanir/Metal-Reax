// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// PARSE-1: for each bundled ReaxFF force field the tables built by reaxmetal's parser must hash to the value computed from the
// tables pinned LAMMPS actually stores (instrumented build, tools/reaxref/parse_tables_check.py). Only hashes are committed
// (ADR-017). usage: test_ffield_tables <dir with ffield.reax.*> <tests/fixtures/ffield_tables.sha256>
// The "portable" dump (everything IEEE-exact) must match on every platform; the "full" dump additionally contains
// pow(gamma_i*gamma_j,-1.5) and is compared only on Linux/glibc, where the goldens were produced; elsewhere that value is
// checked against the same expression instead.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/sha256.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: test_ffield_tables <ffield dir> <hash file>\n"); return 2; }
  std::ifstream in(argv[2]);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
  std::string line;
  int n = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string name, want_portable, want_full;
    ss >> name >> want_portable >> want_full;
    FfieldOptions opt;
    opt.lgvdw = name.size() > 3 && name.compare(name.size() - 3, 3, ".lg") == 0;
    try {
      const ForceField ff = read_force_field_file(std::string(argv[1]) + "/" + name, opt);
      RM_CHECK_MSG(sha256_hex(canonical_table_dump(ff, true)) == want_portable, name + ": portable table hash differs from LAMMPS");
#if defined(__linux__)
      RM_CHECK_MSG(sha256_hex(canonical_table_dump(ff, false)) == want_full, name + ": full table hash differs from LAMMPS");
#endif
      for (int i = 0; i < ff.num_types(); ++i)
        for (int j = 0; j < ff.num_types(); ++j)
          RM_CHECK(ff.pair(i, j).gamma == std::pow(ff.single(j).gamma * ff.single(i).gamma, -1.5));
      ++n;
    } catch (const std::exception& e) {
      RM_CHECK_MSG(false, name + ": " + e.what());
    }
  }
  RM_CHECK_MSG(n == 11, "expected 11 bundled force fields, checked " + std::to_string(n));
  return rmtest::finish("ffield_tables");
}
