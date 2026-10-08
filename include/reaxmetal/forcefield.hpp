// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// ReaxFF parameter tables: an independent C++ reader of the ffield format as accepted by pinned LAMMPS
// (stable_30Sep2026 @ 8de817dd, src/REAXFF/reaxff_ffield.cpp) and of its control file (reaxff_control.cpp).
//
// Contract (ENGINE_SPEC section 2, VALIDATION PARSE-1):
//   * For every file pinned LAMMPS parses *correctly*, the tables built here are bit-identical to the ones LAMMPS stores
//     (verified for the 11 bundled files by canonical-dump hashes, and by a differential test against LAMMPS itself).
//   * Where the reference silently corrupts or invents physics, this reader REJECTS with an explicit error
//     (strict mode, the default): out-of-range three-body slots (Q-09), absent bond-pair blocks for *used* element
//     pairs (Q-12), non-finite numbers, truncated files (other than a missing H-bond block, which LAMMPS also accepts).
//     `FfieldOptions::reproduce_lammps_leniency` re-enables the truncation leniency (with a warning), nothing else.
//   * Element knowledge (mass/name branches of the kernels) is expressed as per-type / per-pair compat flags
//     (compat_flags.hpp) computed once here; kernels never see element names or masses (ADR-003).
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "reaxmetal/compat_flags.hpp"

namespace reaxmetal {

class FfieldError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct GlobalParams {
  std::vector<double> l;  // gp.l[0..n_global-1]; layout: reaxff_types.h / ENGINE_SPEC 2.1
  int vdw_type = 0;       // 1 shielding, 2 inner wall, 3 both (derived from the atom blocks, reaxff_ffield.cpp:243-290)
};

struct SingleBody {
  std::string name;  // upper-cased, at most 3 characters
  double r_s = 0, valency = 0, mass = 0, r_vdw = 0, epsilon = 0, gamma = 0, r_pi = 0, valency_e = 0, nlp_opt = 0;
  double alpha = 0, gamma_w = 0, valency_boc = 0, p_ovun5 = 0, chi = 0, eta = 0;  // eta = 2 x file value (as LAMMPS)
  int p_hbond = 0;
  double r_pi_pi = 0, p_lp2 = 0, b_o_131 = 0, b_o_132 = 0, b_o_133 = 0, bcut_acks2 = 0;
  double p_ovun2 = 0, p_val3 = 0, valency_val = 0, p_val5 = 0, rcore2 = 0, ecore2 = 0, acore2 = 0;
  double lgcij = 0, lgre = 0;
  SpeciesFlags flags;  // derived once; kernels use these instead of names / masses
};

struct TwoBody {
  double p_bo1 = 0, p_bo2 = 0, p_bo3 = 0, p_bo4 = 0, p_bo5 = 0, p_bo6 = 0;
  double r_s = 0, r_p = 0, r_pp = 0, p_boc3 = 0, p_boc4 = 0, p_boc5 = 0;
  double p_be1 = 0, p_be2 = 0, De_s = 0, De_p = 0, De_pp = 0, p_ovun1 = 0;
  double D = 0, alpha = 0, r_vdW = 0, gamma_w = 0, rcore = 0, ecore = 0, acore = 0, lgcij = 0, lgre = 0;
  double gamma = 0;  // stored as (gamma_i*gamma_j)^-1.5, as LAMMPS ("note: this parameter is gamma^-3")
  double v13cor = 0, ovc = 0;
  bool listed = false;                     // a bond-parameter block for this pair exists in the file (Q-12)
  bool triple_bond_stabilisation = false;  // compat flag (gp[37]==2 or mass pair 12.0000/15.9990), set after parsing
};

struct ThreeBodySet {
  double theta_00 = 0, p_val1 = 0, p_val2 = 0, p_val4 = 0, p_val7 = 0, p_pen1 = 0, p_coa1 = 0;
  bool operator==(const ThreeBodySet&) const = default;
};

struct FourBody {
  double V1 = 0, V2 = 0, V3 = 0, p_tor1 = 0, p_cot1 = 0;
  bool operator==(const FourBody&) const = default;
};

struct HBondParams {
  double r0_hb = -1.0, p_hb1 = 0, p_hb2 = 0, p_hb3 = 0;  // r0_hb == -1 : no entry
  bool operator==(const HBondParams&) const = default;
};

struct ControlParams {
  double bond_cut = 5.0, bg_cut = 0.3, thb_cut = 0.001, thb_cutsq = 0.00001, hbond_cut = 7.5;
  int tabulate = 0, nthreads = 1;
  // filled from the force field general parameters by ForceField (reaxff_ffield.cpp:620-624)
  double nonb_low = 0, nonb_cut = 0, bo_cut = 0;
};

struct FfieldOptions {
  bool lgvdw = false;                       // pair_style reaxff lgvdw yes: five-line atom blocks, extra off-diagonal column
  bool reproduce_lammps_leniency = false;   // accept truncated files with a warning, as LAMMPS does (default: reject)
};

class ForceField;
ForceField parse_force_field(std::string_view text, std::string_view name, const FfieldOptions& opt);

class ForceField {
 public:
  int num_types() const noexcept { return static_cast<int>(sbp_.size()); }
  const GlobalParams& global() const noexcept { return gp_; }
  const SingleBody& single(int i) const { return sbp_.at(static_cast<std::size_t>(i)); }
  const TwoBody& pair(int i, int j) const { return tbp_.at(static_cast<std::size_t>(i) * sbp_.size() + static_cast<std::size_t>(j)); }
  // effective three-body parameter sets for the ordered triple (j,k,l), in the order the reference sums them.
  // (The reference stores a doubled slot count with inert zero slots for j==l; see ENGINE_SPEC Q-09.)
  std::span<const ThreeBodySet> three_body(int j, int k, int l) const;
  const FourBody* four_body(int j, int k, int l, int m) const;  // nullptr if the reference has no torsion entry
  const HBondParams* hbond(int j, int k, int l) const;           // nullptr if absent
  const ControlParams& file_control() const noexcept { return ctl_; }  // cutoffs that come from the ffield (nonb_low/cut, bo_cut)
  bool gp37_stabilisation() const noexcept { return gp37_; }
  bool c2_correction_active() const noexcept { return c2_active_; }
  const std::vector<std::string>& warnings() const noexcept { return warnings_; }

  // Indices of all file types whose (<=3 character, upper-cased) name equals `arg` compared case-insensitively - the
  // LAMMPS pair_coeff rule (pair_reaxff.cpp:250-258). Normally one entry; more than one is a defective file.
  std::vector<int> match_element(std::string_view arg) const;

  // Q-12: LAMMPS silently zero-fills absent bond-pair blocks, creating a phantom bond (BO' = 1 at every r <= bond_cut).
  // Throws FfieldError naming the first pair among `types` (file indices, both orders implied) without a block.
  void require_bond_blocks(std::span<const int> types) const;

  // internal construction interface (ffield_parser.cpp)
  struct Builder;

 private:
  friend struct Builder;
  friend ForceField parse_force_field(std::string_view, std::string_view, const FfieldOptions&);
  GlobalParams gp_;
  ControlParams ctl_;
  std::vector<SingleBody> sbp_;
  std::vector<TwoBody> tbp_;                                         // n x n, symmetric
  std::map<std::array<int, 3>, std::vector<ThreeBodySet>> thbp_;     // ordered triples
  std::map<std::array<int, 4>, FourBody> fbp_;
  std::map<std::array<int, 3>, HBondParams> hbp_;
  bool gp37_ = false, c2_active_ = false;
  std::vector<std::string> warnings_;
};

// Parse a force field from memory. `name` is used in diagnostics only. Throws FfieldError.
ForceField parse_force_field(std::string_view text, std::string_view name, const FfieldOptions& opt);
inline ForceField parse_force_field(std::string_view text, std::string_view name) { return parse_force_field(text, name, FfieldOptions{}); }
ForceField read_force_field_file(const std::string& path, const FfieldOptions& opt = {});

// Control file (pair_style reaxff <file>): keywords nbrhood_cutoff, bond_graph_cutoff, thb_cutoff, thb_cutoff_sq,
// hbond_cutoff, tabulate_long_range, write_freq; the LAMMPS "inactive" keywords are accepted with a warning;
// anything else is an error. EOF ends the file. Throws FfieldError.
ControlParams parse_control_text(std::string_view text, std::string_view name, std::vector<std::string>* warnings = nullptr);
ControlParams read_control_file(const std::string& path, std::vector<std::string>* warnings = nullptr);

// Canonical, sorted, lossless (%.17g) text dump of every parameter table; the same format is produced from the tables
// LAMMPS stored (tools/reaxref/canonical_tables.py), so equal text <=> equal tables (PARSE-1).
// `portable` omits TwoBody::gamma, the only value that goes through libm pow() and may differ in the last bit between C
// libraries; everything else is IEEE-exact (+,-,*,sqrt) and must match bit for bit on every platform.
std::string canonical_table_dump(const ForceField& ff, bool portable = false);

}  // namespace reaxmetal
