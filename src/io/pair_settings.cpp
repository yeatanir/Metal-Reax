// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/pair_settings.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>

#include "reaxmetal/capabilities.hpp"

namespace reaxmetal {
namespace {

bool logical(const std::string& s, const char* key) {
  if (s == "yes" || s == "on" || s == "true" || s == "1") return true;
  if (s == "no" || s == "off" || s == "false" || s == "0") return false;
  throw FfieldError(std::string("Expected boolean parameter instead of '") + s + "' for pair_style reaxff/metal keyword " + key);
}
bool is_integer(const std::string& s) {
  std::size_t i = 0;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
  if (i >= s.size()) return false;
  for (; i < s.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
  return true;
}
int integer(const std::string& s, const char* key) {
  if (!is_integer(s)) throw FfieldError(std::string("Expected integer parameter instead of '") + s + "' for pair_style reaxff/metal keyword " + key);
  const long v = std::strtol(s.c_str(), nullptr, 10);
  if (v > 2147483647L || v < -2147483647L - 1) throw FfieldError("Integer " + s + " is out of range");
  return static_cast<int>(v);
}
double number(const std::string& s, const char* key) {
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (s.empty() || end != s.c_str() + s.size()) throw FfieldError(std::string("Expected floating point parameter instead of '") + s + "' for pair_style reaxff/metal keyword " + key);
  return v;
}

}  // namespace

PairSettings parse_pair_style_args(std::span<const std::string> args) {
  if (args.empty()) throw FfieldError("Illegal pair_style command");
  PairSettings p;
  if (args[0] != "NULL") {
    p.control_file = args[0];
    p.control = read_control_file(args[0], &p.notices);
  }
  for (std::size_t i = 1; i < args.size();) {
    const std::string& key = args[i];
    if (i + 2 > args.size()) throw FfieldError("Illegal pair_style reaxff/metal command: keyword '" + key + "' needs a value");
    const std::string& val = args[i + 1];
    if (key == "checkqeq") p.checkqeq = logical(val, "checkqeq");
    else if (key == "enobonds") p.enobonds = logical(val, "enobonds");
    else if (key == "lgvdw") p.lgvdw = logical(val, "lgvdw");
    else if (key == "shellcheck") p.shellcheck = logical(val, "shellcheck");
    else if (key == "ereaxff") p.ereaxff = logical(val, "ereaxff");
    else if (key == "acks2") p.acks2 = logical(val, "acks2");
    else if (key == "acks2_solver") {
      if (val != "dense" && val != "iterative" && val != "auto") throw FfieldError("Illegal pair_style reaxff/metal command: acks2_solver must be dense, iterative or auto, got '" + val + "'");
      p.acks2_solver = val;
    }
    else if (key == "acks2_tol") {
      p.acks2_tol = number(val, "acks2_tol");
      if (!(p.acks2_tol > 0.0 && p.acks2_tol < 1.0)) throw FfieldError("Illegal pair_style reaxff/metal command: acks2_tol must be in (0, 1)");
    }
    else if (key == "acks2_dense_max") {
      p.acks2_dense_max = integer(val, "acks2_dense_max");
      if (p.acks2_dense_max < 0) throw FfieldError("Illegal pair_style reaxff/metal command: acks2_dense_max must be >= 0");
    }
    else if (key == "acks2_cache") p.acks2_cache = logical(val, "acks2_cache");
    else if (key == "acks2_constants") {
      if (val != "legacy" && val != "consistent") throw FfieldError("Illegal pair_style reaxff/metal command: acks2_constants must be legacy or consistent, got '" + val + "'");
      p.acks2_legacy_constants = val == "legacy";
    }
    else if (key == "eparams") {
      p.eparams_file = val;
      try {
        p.species = std::make_shared<const SpeciesTable>(read_species_table(val));
      } catch (const std::exception& e) {
        throw FfieldError(std::string("pair_style reaxff/metal eparams: ") + e.what());
      }
    }
    else if (key == "electron_taper") {
      p.electron_taper = number(val, "electron_taper");
      if (!(p.electron_taper > 0.0)) throw FfieldError("Illegal pair_style reaxff/metal command: electron_taper must be > 0 (a zero taper silently removes every electron pair in the legacy code)");
    }
    else if (key == "bonded") {
      if (val != "cpu64" && val != "metal") throw FfieldError("Illegal pair_style reaxff/metal command: bonded must be cpu64 or metal, got '" + val + "'");
      p.bonded = val;
    } else if (key == "backend") {
      if (val != "cpu64" && val != "metal") throw FfieldError("Illegal pair_style reaxff/metal command: backend must be cpu64 or metal, got '" + val + "'");
      p.backend = val;
    }
    else if (key == "reaxmetal_selfcheck") p.selfcheck = logical(val, "reaxmetal_selfcheck");
    else if (key == "safezone") {
      p.safezone = number(val, "safezone");
      if (p.safezone < 0.0) throw FfieldError("Illegal pair_style reaxff/metal safezone command");
      p.notices.push_back("pair_style reaxff/metal: 'safezone' sizes the matrices of stock charge fixes; the plugin's own engine does not need it");
    } else if (key == "mincap") {
      p.mincap = integer(val, "mincap");
      if (p.mincap < 0) throw FfieldError("Illegal pair_style reaxff/metal mincap command");
      p.notices.push_back("pair_style reaxff/metal: 'mincap' sizes the matrices of stock charge fixes; the plugin's own engine does not need it");
    } else if (key == "minhbonds") {
      p.minhbonds = integer(val, "minhbonds");
      if (p.minhbonds < 0) throw FfieldError("Illegal pair_style reaxff/metal minhbonds command");
      p.notices.push_back("pair_style reaxff/metal: 'minhbonds' sizes the matrices of stock charge fixes; the plugin's own engine does not need it");
    } else if (key == "list/blocking") {
      p.list_blocking = logical(val, "list/blocking");
      p.notices.push_back("pair_style reaxff/metal: 'list/blocking' is a Kokkos performance option and is ignored");
    } else if (key == "tabulate") {
      p.tabulate = integer(val, "tabulate");
      if (p.tabulate < 0) throw FfieldError("Illegal pair_style reaxff/metal tabulate command");
    } else {
      throw FfieldError("Illegal pair_style reaxff/metal command: unknown keyword '" + key + "'");
    }
    i += 2;
  }
  if (p.ereaxff) {
    if (p.backend != "cpu64") throw FfieldError("pair_style reaxff/metal: eReaxFF mode (ereaxff yes) supports only backend cpu64");
    if (p.lgvdw) throw FfieldError("pair_style reaxff/metal: eReaxFF mode does not support lgvdw");
    p.checkqeq = false;   // charges are fixed (icharg 8) or solved inside the pair style (acks2 yes, icharg 7): a charge fix would overwrite them
    if (p.acks2 && p.acks2_legacy_constants)
      p.notices.push_back("pair_style reaxff/metal: acks2_constants legacy reproduces the legacy executable (training energies, comparisons), but its forces are NOT the gradient of its energy "
                          "(compact: bond-softness pairs with a fixed particle are in the energy but not in the solve; several charge groups: the -q_ref*s term is missing; "
                          "mismatched Coulomb constants). Do not use it for dynamics; the default `acks2_constants consistent` has forces equal to the gradient of the energy.");
    if (p.acks2)
      p.notices.push_back("pair_style reaxff/metal: eReaxFF mode with acks2 yes: the charges of the species marked `charge solve` (default: all but the electron species) are solved by ACKS2 inside the pair style (legacy icharg 7); the others keep their data-file charge");
    else
      p.notices.push_back("pair_style reaxff/metal: eReaxFF mode: charges are taken from the data file (legacy icharg 8, no charge fix, no polarization energy); use `acks2 yes` to solve them");
  } else if (p.acks2) {
    throw FfieldError("pair_style reaxff/metal: acks2 needs ereaxff yes");
  } else if (!std::isnan(p.electron_taper)) {
    throw FfieldError("pair_style reaxff/metal: electron_taper needs ereaxff yes");
  } else if (p.species) {
    throw FfieldError("pair_style reaxff/metal: eparams needs ereaxff yes");
  }
  if (p.tabulate > 0 || p.control.tabulate > 0)
    p.notices.push_back("pair_style reaxff/metal: 'tabulate' is accepted but no table is built: the non-bonded terms are evaluated analytically (the stock spline table "
                        "only approximates them; results differ from a tabulated stock run by its interpolation error)");
  return p;
}

}  // namespace reaxmetal
