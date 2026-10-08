// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// `pair_style reaxff/metal <control file | NULL> [keyword value]...` -- same syntax as pinned LAMMPS `pair_style reaxff`
// (pair_reaxff.cpp:176-290), parsed here so that it is testable without LAMMPS and gated by the capability table:
//   Rejected / Deferred options raise UnsupportedFeatureError, Ignored options are accepted with a notice, Planned options are
//   accepted (they take effect when their milestone lands; the adapter refuses to compute until M4 anyway).
#include <span>
#include <string>
#include <vector>

#include "reaxmetal/forcefield.hpp"

namespace reaxmetal {

struct PairSettings {
  std::string control_file;          // empty: NULL (defaults)
  bool checkqeq = true;              // a charge fix must be present
  bool lgvdw = false;
  bool enobonds = true;
  double safezone = 1.2;             // allocation heuristics: accepted, ignored (notice)
  int mincap = 50, minhbonds = 25;
  bool list_blocking = false;        // Kokkos-only performance option: accepted, ignored (notice)
  int tabulate = 0;                  // > 0: Deferred (spline tables change the numbers)
  bool selfcheck = false;            // development keyword `reaxmetal_selfcheck yes`: A2 builds the ghost-native view and verifies it against LAMMPS' own neighbor list (then still refuses to compute)
  ControlParams control;             // defaults, or the control file contents
  std::vector<std::string> notices;  // warnings from the control file and "ignored option" notices
};

// Throws FfieldError for illegal syntax / values (message mirrors the LAMMPS "Illegal pair_style" errors) and
// UnsupportedFeatureError (capabilities.hpp) for Deferred / Rejected features (tabulate > 0).
PairSettings parse_pair_style_args(std::span<const std::string> args);

}  // namespace reaxmetal
