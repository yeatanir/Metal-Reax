// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/compat_flags.hpp"

#include <cctype>
#include <stdexcept>

namespace reaxmetal {

std::string normalize_symbol(std::string_view raw) {
  std::string s;
  for (char c : raw) {
    if (s.size() == 3) break;  // strncpy(name, element, 3)
    s.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return s;
}

SpeciesFlags derive_species_flags(std::string_view symbol, double mass) noexcept {
  SpeciesFlags f;
  f.c2_species = (normalize_symbol(symbol) == "C");
  f.heavy_atom_terms = (mass > 21.0);
  f.light_valency_override = (mass < 21);
  f.mass_exactly_12_0000 = (mass == 12.0000);
  f.mass_exactly_15_9990 = (mass == 15.9990);
  return f;
}

bool light_valency_override_applies(const SpeciesFlags& f, double valency_val, double valency_boc) noexcept {
  return f.light_valency_override && (valency_val != valency_boc);
}

bool gp37_forces_stabilisation(std::span<const double> gp) {
  if (gp.size() <= 37)
    throw std::invalid_argument("ffield has only " + std::to_string(gp.size()) +
                                " general parameters; the reference reads gp[37] (out of bounds there)");
  // `double gp37; gp37 = (int) gp.l[37]; ... gp37 == 2`  -> truncation toward zero
  const double g = gp[37];
  if (!(g > -2147483649.0 && g < 2147483648.0))  // NaN or out of int range: (int) cast is undefined upstream
    throw std::invalid_argument("gp[37] is not representable as int");
  return static_cast<double>(static_cast<int>(g)) == 2;
}

bool triple_bond_stabilisation_pair(bool gp37_rule, const SpeciesFlags& a, const SpeciesFlags& b) noexcept {
  return gp37_rule || (a.mass_exactly_12_0000 && b.mass_exactly_15_9990) ||
         (b.mass_exactly_12_0000 && a.mass_exactly_15_9990);
}

bool c2_correction_active(std::span<const double> gp) {
  if (gp.size() <= 5) throw std::invalid_argument("ffield has fewer than 6 general parameters; gp[5] is required");
  return gp[5] > 0.001;
}

}  // namespace reaxmetal
