// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/pair_settings.hpp"

#include <cctype>
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
      p.notices.push_back("pair_style reaxff/metal: 'safezone' is accepted and ignored (allocation heuristic)");
    } else if (key == "mincap") {
      p.mincap = integer(val, "mincap");
      if (p.mincap < 0) throw FfieldError("Illegal pair_style reaxff/metal mincap command");
      p.notices.push_back("pair_style reaxff/metal: 'mincap' is accepted and ignored (allocation heuristic)");
    } else if (key == "minhbonds") {
      p.minhbonds = integer(val, "minhbonds");
      if (p.minhbonds < 0) throw FfieldError("Illegal pair_style reaxff/metal minhbonds command");
      p.notices.push_back("pair_style reaxff/metal: 'minhbonds' is accepted and ignored (allocation heuristic)");
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
  if (p.tabulate > 0 || p.control.tabulate > 0)
    p.notices.push_back("pair_style reaxff/metal: 'tabulate' is accepted but no table is built: the non-bonded terms are evaluated analytically (the stock spline table "
                        "only approximates them; results differ from a tabulated stock run by its interpolation error)");
  return p;
}

}  // namespace reaxmetal
