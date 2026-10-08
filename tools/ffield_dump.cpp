// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// usage: reaxmetal_ffield_dump [--lgvdw] [--portable] [--sha256] [--lenient] <ffield>
// Exit status: 0 = accepted (canonical dump or hash on stdout), 2 = rejected (reason on stderr), 1 = usage error.
#include <cstdio>
#include <cstring>
#include <iostream>

#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/sha256.hpp"

int main(int argc, char** argv) {
  reaxmetal::FfieldOptions opt;
  bool portable = false, hash = false;
  const char* path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--lgvdw")) opt.lgvdw = true;
    else if (!std::strcmp(argv[i], "--portable")) portable = true;
    else if (!std::strcmp(argv[i], "--sha256")) hash = true;
    else if (!std::strcmp(argv[i], "--lenient")) opt.reproduce_lammps_leniency = true;
    else path = argv[i];
  }
  if (!path) { std::fprintf(stderr, "usage: reaxmetal_ffield_dump [--lgvdw] [--portable] [--sha256] [--lenient] <ffield>\n"); return 1; }
  try {
    const auto ff = reaxmetal::read_force_field_file(path, opt);
    const std::string d = reaxmetal::canonical_table_dump(ff, portable);
    if (hash) std::cout << reaxmetal::sha256_hex(d) << "\n"; else std::cout << d;
    for (const auto& w : ff.warnings()) std::fprintf(stderr, "WARNING: %s\n", w.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "REJECTED: %s\n", e.what());
    return 2;
  }
}
