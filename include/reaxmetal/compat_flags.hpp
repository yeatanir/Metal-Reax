// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
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
#include <map>
#include <memory>
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
  // ---- eReaxFF only (computed always, used only when FfieldOptions::ereaxff is set). Legacy source: Margaret/eReaxFF_src.
  bool is_electron = false;                 // normalized symbol == "EL" (reac.f/poten.f compare qa == 'El')
  double core_charge = 0.0;                 // znucel: nuclear charge Z of the element (reac.f:4590-4615); 0 if not in the table
  int share_sign = 0;                       // elecpos (reac.f:8815-8851): -1 -> avalc -= eshare (C,O,H,F,N,Os); +1 -> += (Li,Lv,P,B,Av,Al); 0 none
};

// Per-element eReaxFF data that the legacy source keeps in code (znucel nuclear-charge table, elecpos sharing-sign table). They can be
// supplied from an input file (pair_style keyword `eparams <file>`) so that a new element needs a parameter file, not a code change.
// File format, one element per line ('#' starts a comment):  SYMBOL  Z  SHARE_SIGN  tested|untested
//   Z = nuclear charge used in the electron-core term; SHARE_SIGN in {-1, 0, +1} (elecpos: -1 avalc -= eshare, +1 avalc += eshare).
// An entry overrides the built-in legacy tables for that element. An element is ACCEPTED in eReaxFF mode only if it is C, El, Av
// (validated against the legacy code) or its entry says `tested`; `untested` and absent elements are refused.
struct SpeciesEntry {
  double core_charge = 0.0;
  int share_sign = 0;
  bool tested = false;
};
using SpeciesTable = std::map<std::string, SpeciesEntry>;   // key: normalize_symbol(symbol)
// Throws std::invalid_argument with file:line on any malformed line.
SpeciesTable read_species_table(const std::string& path);
SpeciesTable parse_species_table(std::string_view text, std::string_view name);

// symbol: as read from the ffield (any case, any length); mass: parsed double. NaN compares false everywhere (as upstream).
// table (optional): per-element overrides of core_charge / share_sign (see SpeciesTable).
SpeciesFlags derive_species_flags(std::string_view symbol, double mass, const SpeciesTable* table = nullptr) noexcept;

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
