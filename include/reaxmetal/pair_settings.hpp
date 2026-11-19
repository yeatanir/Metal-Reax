// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// `pair_style reaxff/metal <control file | NULL> [keyword value]...` -- same syntax as pinned LAMMPS `pair_style reaxff`
// (pair_reaxff.cpp:176-290), parsed here so that it is testable without LAMMPS and gated by the capability table:
//   Rejected / Deferred options raise UnsupportedFeatureError, Ignored options are accepted with a notice, Planned options are
//   accepted (they take effect when their milestone lands; the adapter refuses to compute until M4 anyway).
#include <limits>
#include <memory>
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
  bool shellcheck = true;            // error (yes) or warning (no) when the ghost shell is narrower than max(nonb_cut, hbond_cut, 2*bond_cut); the stock style never checks
  std::string bonded = "metal";   // with backend metal: where the bond-order terms run, metal | cpu64 (double precision on the host: restores stock-like NVE energy conservation, ~1.5x instead of ~7x)
  std::string backend = "cpu64";    // force backend: cpu64 (reference, complete) | metal (not complete yet)
  bool ereaxff = false;              // eReaxFF mode (plan task 3.5): explicit electrons (type El), fixed MOLCHARGE charges from the data file, no charge fix, no polarization energy
  double electron_taper = std::numeric_limits<double>::quiet_NaN();   // Angstrom; NaN = general parameter 38 of the ffield (must then be > 0)
  std::shared_ptr<const SpeciesTable> species;   // `eparams <file>`: per-element Z / sharing sign / tested flag (eReaxFF mode)
  std::string eparams_file;
  bool acks2_legacy_constants = false;  // `acks2_constants consistent|legacy`: consistent (default) = constants and terms that make the forces the exact gradient of the energy; legacy = the numeric constants and pairing of the legacy executable (reproduces its charges and energies, not conservative)
  std::string acks2_solver = "auto";  // `acks2_solver dense|iterative|auto`: auto = dense up to acks2_dense_max solved atoms, preconditioned MINRES above
  double acks2_tol = 1.0e-7;         // `acks2_tol`: iterative stop, preconditioned residual norm <= tol * norm of the right-hand side
  int acks2_dense_max = 64;         // `acks2_dense_max`: largest number of solved atoms solved densely by auto
  bool acks2_cache = true;           // `acks2_cache yes|no`: reuse the assembled matrix / factorisation while the solved atoms, box and groups are unchanged; warm start from the previous call
  bool acks2 = false;                // eReaxFF mode: solve the charges of the species with `charge solve` (default: every non-electron species) with ACKS2 inside the pair style (legacy icharg 7); off = all charges from the data file (legacy icharg 8)
  bool selfcheck = false;            // development keyword `reaxmetal_selfcheck yes`: A2 builds the ghost-native view and verifies it against LAMMPS' own neighbor list (then stops with an error: development aid)
  ControlParams control;             // defaults, or the control file contents
  std::vector<std::string> notices;  // warnings from the control file and "ignored option" notices
};

// Throws FfieldError for illegal syntax / values (message mirrors the LAMMPS "Illegal pair_style" errors) and
// UnsupportedFeatureError (capabilities.hpp) for Deferred / Rejected features (tabulate > 0).
PairSettings parse_pair_style_args(std::span<const std::string> args);

}  // namespace reaxmetal
