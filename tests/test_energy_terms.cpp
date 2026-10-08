// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// The term registry must cover every ReaxFF energy contribution exactly once in the LAMMPS pvector.
#include <set>
#include <string>

#include "reaxmetal/energy_terms.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

int main() {
  const auto terms = energy_term_table();
  const auto slots = pvector_layout();
  RM_CHECK(terms.size() == kEnergyTermCount);
  RM_CHECK(slots.size() == kPvectorSize);

  std::set<std::string> names;
  for (std::size_t i = 0; i < terms.size(); ++i) {
    RM_CHECK_MSG(static_cast<std::size_t>(terms[i].term) == i, "table order must equal enum order");
    RM_CHECK(names.insert(std::string(terms[i].name)).second);
    RM_CHECK(!terms[i].lammps_field.empty());
  }

  // each term belongs to exactly one slot; always-zero slots own no terms
  std::array<int, kEnergyTermCount> owner_count{};
  for (std::size_t s = 0; s < slots.size(); ++s) {
    RM_CHECK(slots[s].index == static_cast<int>(s));
    int members = 0;
    for (std::size_t t = 0; t < kEnergyTermCount; ++t)
      if (slots[s].members[t]) { ++owner_count[t]; ++members; }
    if (slots[s].always_zero) RM_CHECK(members == 0);
    else RM_CHECK(members >= 1);
  }
  for (std::size_t t = 0; t < kEnergyTermCount; ++t)
    RM_CHECK_MSG(owner_count[t] == 1, "term not mapped exactly once: " + std::string(terms[t].name));

  // known LAMMPS layout facts (pair_reaxff.cpp:501-514): ea merges over+under; emol/efi are constant 0
  RM_CHECK(slots[1].lammps_name == "ea" && slots[1].members[static_cast<std::size_t>(EnergyTerm::Over)] &&
           slots[1].members[static_cast<std::size_t>(EnergyTerm::Under)]);
  RM_CHECK(slots[3].always_zero && slots[12].always_zero);
  RM_CHECK(slots[13].lammps_name == "eqeq");

  // conservation: with distinct powers of two as inputs, pvector slots sum to total() exactly
  EnergyBreakdown b;
  for (std::size_t t = 0; t < kEnergyTermCount; ++t) b.e[t] = static_cast<double>(1u << t);
  const auto pv = to_lammps_pvector(b);
  double sum = 0.0;
  for (double v : pv) sum += v;
  RM_CHECK(sum == b.total());
  RM_CHECK(pv[1] == b[EnergyTerm::Over] + b[EnergyTerm::Under]);
  RM_CHECK(pv[3] == 0.0 && pv[12] == 0.0);
  return rmtest::finish("energy_terms");
}
