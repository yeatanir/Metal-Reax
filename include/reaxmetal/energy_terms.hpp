// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Registry of every ReaxFF energy contribution this engine must compute, and its mapping onto the
// 14-slot `pvector` that pinned LAMMPS exposes through `compute pair reaxff`
// (src/REAXFF/pair_reaxff.cpp:501-514). The mapping is data, not scattered code, so that an omitted
// term is a failing test rather than a silent difference (architectural rule 5).
#include <array>
#include <cstddef>
#include <span>
#include <string_view>

namespace reaxmetal {

enum class EnergyTerm : std::size_t {
  Bond,          // e_bond  (includes terminal-triple-bond stabilisation, see ENGINE_SPEC 5.1)
  Over,          // e_ov
  Under,         // e_un
  LonePair,      // e_lp    (includes the C2 correction term, see ENGINE_SPEC 5.2)
  Valence,       // e_ang
  Penalty,       // e_pen
  Coalition,     // e_coa   (3-body conjugation)
  HBond,         // e_hb
  Torsion,       // e_tor
  Conjugation,   // e_con   (4-body conjugation)
  VdW,           // e_vdW   (incl. inner wall / lg dispersion where active)
  Coulomb,       // e_ele   (taper-shielded Coulomb)
  Polarization,  // e_pol   (QEq self energy, reported by LAMMPS as "eqeq")
  Count
};
inline constexpr std::size_t kEnergyTermCount = static_cast<std::size_t>(EnergyTerm::Count);

struct EnergyTermInfo {
  EnergyTerm term;
  std::string_view name;          // our canonical name
  std::string_view lammps_field;  // ReaxFF::energy_data member in pinned LAMMPS
};

struct PvectorSlot {
  int index;                      // 0-based index into compute-pair pvector (doc lists them 1-based)
  std::string_view lammps_name;   // name used in pair_reaxff.rst
  std::array<bool, kEnergyTermCount> members;  // which EnergyTerm values are summed into the slot
  bool always_zero;               // LAMMPS reports a constant 0.0 here (emol, efi)
};
inline constexpr std::size_t kPvectorSize = 14;

std::span<const EnergyTermInfo> energy_term_table() noexcept;
std::span<const PvectorSlot> pvector_layout() noexcept;

struct EnergyBreakdown {
  std::array<double, kEnergyTermCount> e{};  // kcal/mol
  double& operator[](EnergyTerm t) { return e[static_cast<std::size_t>(t)]; }
  double operator[](EnergyTerm t) const { return e[static_cast<std::size_t>(t)]; }
  double total() const;  // plain left-to-right sum in enum order; deterministic by construction
};

// Collapse our fine-grained breakdown to the 14 LAMMPS slots (e.g. over+under -> "ea").
std::array<double, kPvectorSize> to_lammps_pvector(const EnergyBreakdown& b);

}  // namespace reaxmetal
