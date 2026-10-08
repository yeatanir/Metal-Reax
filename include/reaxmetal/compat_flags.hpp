// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Compatibility flags: the pinned LAMMPS ReaxFF kernels contain element-name and mass-threshold branches (ENGINE_SPEC
// Q-06). Owner decision (M0 approval #2): reproduce them EXACTLY as per-type / per-pair metadata computed once when
// the parameter tables are built, so that no computation kernel ever sees an element name or a mass. These are
// compatibility flags, not new chemical rules; every predicate below mirrors one upstream expression and cites it.
//
//   upstream expression                                              source (stable_30Sep2026 @ 8de817dd)
//   mass < 21  &&  valency_val != valency_boc  (parse-time override) reaxff_ffield.cpp:300-301
//   mass > 21.0                                                       reaxff_bond_orders.cpp:457, reaxff_multi_body.cpp:137
//   strcmp(name,"C") == 0   (C2 correction, central and neighbour)   reaxff_multi_body.cpp:101,107
//   mass_i == 12.0000 && mass_j == 15.9990 (either order)             reaxff_bonds.cpp:109-110
//   (double)(int) gp[37] == 2                                         reaxff_bonds.cpp:54,108
//   p_lp3 = gp[5] > 0.001   (gate for the whole C2 correction)        reaxff_multi_body.cpp:101
#include <span>
#include <string>
#include <string_view>

namespace reaxmetal {

// LAMMPS stores the element symbol upper-cased and truncated to 3 characters (reaxff_ffield.cpp:175-178).
std::string normalize_symbol(std::string_view raw);

struct SpeciesFlags {
  bool c2_species = false;                  // normalized symbol == "C"
  bool heavy_atom_terms = false;            // mass > 21.0  (dfvl = 0, Delta_lp_temp treatment)
  bool light_valency_override = false;      // mass < 21 (strictly): valency_val := valency_boc if they differ
  bool mass_exactly_12_0000 = false;        // mass == 12.0000 (exact double equality, as upstream)
  bool mass_exactly_15_9990 = false;        // mass == 15.9990 (exact double equality, as upstream)
};

// symbol: as read from the ffield (any case, any length); mass: parsed double. NaN compares false everywhere (as upstream).
SpeciesFlags derive_species_flags(std::string_view symbol, double mass) noexcept;

// Upstream applies the override only when the two valencies differ (exact !=).
bool light_valency_override_applies(const SpeciesFlags& f, double valency_val, double valency_boc) noexcept;

// gp[37] is cast to int (truncation toward zero) and compared with 2. gp must have n_global > 37 entries: the upstream
// read of gp.l[37] is out of bounds otherwise; here that is an explicit std::invalid_argument.
bool gp37_forces_stabilisation(std::span<const double> gp);

// Pair flag for the terminal-triple-bond stabilisation: gp37 rule OR (12.0000, 15.9990) mass pair in either order.
bool triple_bond_stabilisation_pair(bool gp37_rule, const SpeciesFlags& a, const SpeciesFlags& b) noexcept;

// Gate for the whole C2 correction: gp[5] > 0.001 (strict).
bool c2_correction_active(std::span<const double> gp);

}  // namespace reaxmetal
