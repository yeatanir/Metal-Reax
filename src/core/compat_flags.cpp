// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/compat_flags.hpp"

#include <cctype>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace reaxmetal {

std::string normalize_symbol(std::string_view raw) {
  std::string s;
  for (char c : raw) {
    if (s.size() == 3) break;  // strncpy(name, element, 3)
    s.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return s;
}

namespace {
// znucel table, reac.f:4590-4615. The legacy compare is case-sensitive on 2-character names ('Mg'); we compare the normalized
// (upper-case) symbol, a superset that only differs for ffields that spell an element in the wrong case.
double core_charge_of(const std::string& n) {
  static const std::pair<const char*, double> kZ[] = {
      {"H", 1}, {"HE", 2}, {"LI", 1}, {"B", 5}, {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"NA", 1}, {"MG", 2}, {"AL", 5}, {"SI", 4},
      {"P", 15}, {"S", 6}, {"CL", 7}, {"K", 1}, {"CA", 2}, {"FE", 8}, {"NI", 10}, {"CU", 11}, {"ZN", 12}, {"AG", 3}, {"LV", 3},
      {"AV", 3}, {"OS", 5}};
  for (const auto& e : kZ)
    if (n == e.first) return e.second;
  return 0.0;
}
// elecpos sign table, reac.f:8815-8851 (first copy). The later copies at ~8900-8940 and ~8950-9000 use slightly different lists;
// see phase3 notes (task 3.3 must audit which copy is live for each code path).
int share_sign_of(const std::string& n) {
  for (const char* m : {"C", "O", "H", "F", "N", "OS"})
    if (n == m) return -1;
  for (const char* m : {"LI", "LV", "P", "B", "AV", "AL"})
    if (n == m) return +1;
  return 0;
}
}  // namespace

SpeciesTable parse_species_table(std::string_view text, std::string_view name) {
  SpeciesTable t;
  std::istringstream in{std::string(text)};
  std::string line;
  int ln = 0;
  while (std::getline(in, line)) {
    ++ln;
    if (const auto h = line.find('#'); h != std::string::npos) line.erase(h);
    std::istringstream ls(line);
    std::string sym, tag, extra;
    double z = 0.0;
    int sign = 0;
    if (!(ls >> sym)) continue;   // blank line
    const std::string where = std::string(name) + ":" + std::to_string(ln) + ": ";
    if (!(ls >> z >> sign >> tag) || (ls >> extra)) throw std::invalid_argument(where + "expected 'SYMBOL Z SHARE_SIGN tested|untested'");
    if (sign < -1 || sign > 1) throw std::invalid_argument(where + "SHARE_SIGN must be -1, 0 or +1");
    if (!(z >= 0.0 && z < 200.0)) throw std::invalid_argument(where + "nuclear charge Z out of range");
    if (tag != "tested" && tag != "untested") throw std::invalid_argument(where + "last field must be 'tested' or 'untested'");
    const std::string key = normalize_symbol(sym);
    if (t.count(key)) throw std::invalid_argument(where + "element " + key + " listed twice");
    t[key] = SpeciesEntry{z, sign, tag == "tested"};
  }
  return t;
}

SpeciesTable read_species_table(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open eReaxFF species table " + path);
  const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return parse_species_table(text, path);
}

SpeciesFlags derive_species_flags(std::string_view symbol, double mass, const SpeciesTable* table) noexcept {
  SpeciesFlags f;
  f.c2_species = (normalize_symbol(symbol) == "C");
  f.heavy_atom_terms = (mass > 21.0);
  f.light_valency_override = (mass < 21);
  f.mass_exactly_12_0000 = (mass == 12.0000);
  f.mass_exactly_15_9990 = (mass == 15.9990);
  const std::string n = normalize_symbol(symbol);
  f.is_electron = (n == "EL");
  f.core_charge = core_charge_of(n);
  f.share_sign = share_sign_of(n);
  if (table) {
    if (const auto it = table->find(n); it != table->end()) {
      f.core_charge = it->second.core_charge;
      f.share_sign = it->second.share_sign;
    }
  }
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
