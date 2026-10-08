// SPDX-License-Identifier: GPL-2.0-only
#include "reaxmetal/energy_terms.hpp"

#include <initializer_list>

namespace reaxmetal {
namespace {

using T = EnergyTerm;

constexpr std::array<EnergyTermInfo, kEnergyTermCount> kTerms{{
    {T::Bond, "bond", "e_bond"},
    {T::Over, "over", "e_ov"},
    {T::Under, "under", "e_un"},
    {T::LonePair, "lone_pair", "e_lp"},
    {T::Valence, "valence", "e_ang"},
    {T::Penalty, "penalty", "e_pen"},
    {T::Coalition, "coalition", "e_coa"},
    {T::HBond, "hbond", "e_hb"},
    {T::Torsion, "torsion", "e_tor"},
    {T::Conjugation, "conjugation", "e_con"},
    {T::VdW, "vdw", "e_vdW"},
    {T::Coulomb, "coulomb", "e_ele"},
    {T::Polarization, "polarization", "e_pol"},
}};

constexpr std::array<bool, kEnergyTermCount> members(std::initializer_list<T> ts) {
  std::array<bool, kEnergyTermCount> m{};
  for (T t : ts) m[static_cast<std::size_t>(t)] = true;
  return m;
}

// Mirrors pair_reaxff.cpp:501-514 (pvector[0..13]) and the name list in doc/src/pair_reaxff.rst.
constexpr std::array<PvectorSlot, kPvectorSize> kPvector{{
    {0, "eb", members({T::Bond}), false},
    {1, "ea", members({T::Over, T::Under}), false},
    {2, "elp", members({T::LonePair}), false},
    {3, "emol", members({}), true},
    {4, "ev", members({T::Valence}), false},
    {5, "epen", members({T::Penalty}), false},
    {6, "ecoa", members({T::Coalition}), false},
    {7, "ehb", members({T::HBond}), false},
    {8, "et", members({T::Torsion}), false},
    {9, "eco", members({T::Conjugation}), false},
    {10, "ew", members({T::VdW}), false},
    {11, "ep", members({T::Coulomb}), false},
    {12, "efi", members({}), true},
    {13, "eqeq", members({T::Polarization}), false},
}};

}  // namespace

std::span<const EnergyTermInfo> energy_term_table() noexcept { return kTerms; }
std::span<const PvectorSlot> pvector_layout() noexcept { return kPvector; }

double EnergyBreakdown::total() const {
  double s = 0.0;
  for (double v : e) s += v;
  return s;
}

std::array<double, kPvectorSize> to_lammps_pvector(const EnergyBreakdown& b) {
  std::array<double, kPvectorSize> out{};
  for (const PvectorSlot& slot : kPvector) {
    double s = 0.0;
    for (std::size_t t = 0; t < kEnergyTermCount; ++t)
      if (slot.members[t]) s += b.e[t];
    out[static_cast<std::size_t>(slot.index)] = s;
  }
  return out;
}

}  // namespace reaxmetal
