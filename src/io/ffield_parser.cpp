// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
//
// Independent re-implementation of the ReaxFF force-field reader accepted by pinned LAMMPS
// (src/REAXFF/reaxff_ffield.cpp, stable_30Sep2026 @ 8de817dd, PuReMD-derived, GPL-2.0-or-later). Written from the
// documented file layout and from observation of that reader's behaviour (ENGINE_SPEC section 2); no upstream code is
// copied. Every behaviour that reproduces a reference quirk is commented with the upstream line range it mirrors.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>

#include "reaxmetal/forcefield.hpp"
#include "text_reader.hpp"

namespace reaxmetal {

using detail::EofReached;
using detail::Reader;
using detail::Tokens;

namespace {
constexpr int kMaxThreeBodySlots = 5;  // REAX_MAX_3BODY_PARAM: the reference array bound (reaxff_defs.h:56)
}

struct ForceField::Builder {
  ForceField ff;
  int n = 0;
  TwoBody& tb(int i, int j) { return ff.tbp_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)]; }
};

std::span<const ThreeBodySet> ForceField::three_body(int j, int k, int l) const {
  auto it = thbp_.find({j, k, l});
  return it == thbp_.end() ? std::span<const ThreeBodySet>{} : std::span<const ThreeBodySet>(it->second);
}
const FourBody* ForceField::four_body(int j, int k, int l, int m) const {
  auto it = fbp_.find({j, k, l, m});
  return it == fbp_.end() ? nullptr : &it->second;
}
const HBondParams* ForceField::hbond(int j, int k, int l) const {
  auto it = hbp_.find({j, k, l});
  return it == hbp_.end() ? nullptr : &it->second;
}

std::vector<int> ForceField::match_element(std::string_view arg) const {
  auto lower = [](std::string_view s) {
    std::string r;
    for (char c : s) r.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return r;
  };
  std::vector<int> out;
  const std::string a = lower(arg);
  for (int i = 0; i < num_types(); ++i)
    if (a == lower(sbp_[static_cast<std::size_t>(i)].name)) out.push_back(i);
  return out;
}

void ForceField::require_bond_blocks(std::span<const int> types) const {
  for (std::size_t a = 0; a < types.size(); ++a)
    for (std::size_t b = a; b < types.size(); ++b)
      if (!pair(types[a], types[b]).listed)
        throw FfieldError("force field has no bond-parameter block for the element pair " + single(types[a]).name + "-" +
                          single(types[b]).name +
                          "; pinned LAMMPS would silently zero-fill it and create a phantom bond of bond order 1 at every distance "
                          "up to the bond cutoff (ENGINE_SPEC Q-12). Add the block or pick other elements.");
}

namespace {

std::string upper3(std::string_view s) {
  std::string r;
  for (char c : s) {
    if (r.size() == 3) break;  // strncpy(sbp.name, element, 3)
    r.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return r;
}

[[noreturn]] void fail(const Reader& r, const std::string& msg) { throw FfieldError(r.where() + ": " + msg); }

void need_columns(const Reader& r, const Tokens& t, std::size_t want) {
  if (t.count() < want)
    fail(r, "invalid force field file format: expected " + std::to_string(want) + " columns but found " + std::to_string(t.count()));
}

}  // namespace

ForceField parse_force_field(std::string_view text, std::string_view name, const FfieldOptions& opt) {
  ForceField::Builder B;
  ForceField& ff = B.ff;
  Reader rd(text, name);
  const bool lg = opt.lgvdw;

  // Truncation: LAMMPS treats EOF anywhere as a warning and keeps whatever was read (reaxff_ffield.cpp:575-578), leaving the
  // remaining tables zero. Strict mode (default) rejects, except for the absent hydrogen-bond block.
  try {
    // ---- header comment line (reaxff_ffield.cpp:90-96): the first non-blank line is a comment, but must not look like the
    // "N ! Number of general parameters" line.
    {
      auto first = rd.next_line();
      if (!first) throw EofReached{};
      std::string joined;
      // re-create the raw line text for the regex test: ^\s*[0-9]+\s+!.*general parameters.*
      // (the tokens joined by single spaces are equivalent for this test)
      Tokens t = std::move(*first);
      std::vector<std::string> w;
      while (t.has_next()) w.emplace_back(t.next_string());
      for (const auto& x : w) joined += x + " ";
      bool digits = !w.empty() && std::all_of(w[0].begin(), w[0].end(), [](char c) { return c >= '0' && c <= '9'; });
      if (digits && w.size() >= 2 && !w[1].empty() && w[1][0] == '!' && joined.find("general parameters") != std::string::npos)
        fail(rd, "first line of ReaxFF potential file must be a comment or empty");
    }

    // ---- general parameters
    int n = 0;
    {
      Tokens v = rd.next_values();
      n = v.next_int();
      if (n < 1) fail(rd, "invalid number of global parameters");
      if (n < 38) fail(rd, "the force field has only " + std::to_string(n) + " general parameters; the kernels read gp[37] (>= 38 required)");
      ff.gp_.l.resize(static_cast<std::size_t>(n));
      ff.gp_.vdw_type = 0;
      for (int i = 0; i < n; ++i) {
        Tokens g = rd.next_values();
        ff.gp_.l[static_cast<std::size_t>(i)] = g.next_double();
      }
    }

    // ---- number of atom types, then 3 comment lines (raw: they are consumed even if blank)
    int ntypes = 0;
    {
      Tokens v = rd.next_values();
      ntypes = v.next_int();
      if (ntypes < 1) fail(rd, "invalid number of atom types");
      rd.skip_line();
      rd.skip_line();
      rd.skip_line();
    }
    B.n = ntypes;
    ff.sbp_.assign(static_cast<std::size_t>(ntypes), SingleBody{});
    ff.tbp_.assign(static_cast<std::size_t>(ntypes) * static_cast<std::size_t>(ntypes), TwoBody{});
    auto& gp = ff.gp_;

    // ---- atom blocks: four lines (five with lgvdw)
    for (int i = 0; i < ntypes; ++i) {
      SingleBody& s = ff.sbp_[static_cast<std::size_t>(i)];
      Tokens v = rd.next_values();
      if (v.count() < 8 && !lg) fail(rd, "this force field file requires using 'lgvdw yes'");
      need_columns(rd, v, 9);
      s.name = upper3(v.next_string());
      s.r_s = v.next_double();
      s.valency = v.next_double();
      s.mass = v.next_double();
      s.r_vdw = v.next_double();
      s.epsilon = v.next_double();
      s.gamma = v.next_double();
      s.r_pi = v.next_double();
      s.valency_e = v.next_double();
      s.nlp_opt = 0.5 * (s.valency_e - s.valency);

      v = rd.next_values();
      need_columns(rd, v, 8);
      s.alpha = v.next_double();
      s.gamma_w = v.next_double();
      s.valency_boc = v.next_double();
      s.p_ovun5 = v.next_double();
      v.skip();
      s.chi = v.next_double();
      s.eta = 2.0 * v.next_double();
      s.p_hbond = static_cast<int>(v.next_double());  // (int) truncation toward zero, as upstream

      v = rd.next_values();
      need_columns(rd, v, 8);
      s.r_pi_pi = v.next_double();
      s.p_lp2 = v.next_double();
      v.skip();
      s.b_o_131 = v.next_double();
      s.b_o_132 = v.next_double();
      s.b_o_133 = v.next_double();
      s.bcut_acks2 = v.next_double();

      v = rd.next_values();
      need_columns(rd, v, 8);
      s.p_ovun2 = v.next_double();
      s.p_val3 = v.next_double();
      v.skip();
      s.valency_val = v.next_double();
      s.p_val5 = v.next_double();
      s.rcore2 = v.next_double();
      s.ecore2 = v.next_double();
      s.acore2 = v.next_double();

      if (lg) {
        v = rd.next_values();
        // the fifth line must start with a number; otherwise it is the next element and the file has no lg data
        if (!v.first_is_number()) fail(rd, "ReaxFF potential file is not compatible with 'lgvdw yes'");
        need_columns(rd, v, 2);
        s.lgcij = v.next_double();
        s.lgre = v.next_double();
      } else {
        s.lgcij = s.lgre = 0.0;
      }

      // van der Waals method consistency (reaxff_ffield.cpp:243-290); warnings keep the first method, as upstream
      auto warn = [&](const std::string& m) { ff.warnings_.push_back(m); };
      if (s.rcore2 > 0.01 && s.acore2 > 0.01) {
        if (s.gamma_w > 0.5) {
          if (gp.vdw_type != 0 && gp.vdw_type != 3)
            warn("Van der Waals parameters for element " + s.name + " indicate inner wall+shielding, but earlier atoms indicate a different van der Waals method. Keeping van der Waals setting for earlier atoms.");
          else
            gp.vdw_type = 3;
        } else {
          if (gp.vdw_type != 0 && gp.vdw_type != 2)
            warn("Van der Waals parameters for element " + s.name + " indicate inner wall without shielding, but earlier atoms indicate a different van der Waals method. Keeping van der Waals setting for earlier atoms.");
          else
            gp.vdw_type = 2;
        }
      } else {
        if (s.gamma_w > 0.5) {
          if (gp.vdw_type != 0 && gp.vdw_type != 1)
            warn("Van der Waals parameters for element " + s.name + " indicate shielding without inner wall, but earlier elements indicate a different van der Waals method. Keeping van der Waals setting for earlier atoms.");
          else
            gp.vdw_type = 1;
        } else {
          fail(rd, "inconsistent van der Waals parameters: no shielding or inner wall set for element " + s.name);
        }
      }
    }

    // first-row valency fix (reaxff_ffield.cpp:299-305); flags derived from the parsed mass
    for (auto& s : ff.sbp_) {
      s.flags = derive_species_flags(s.name, s.mass);
      if (light_valency_override_applies(s.flags, s.valency_val, s.valency_boc)) {
        ff.warnings_.push_back("Changed valency_val to valency_boc for " + s.name);
        s.valency_val = s.valency_boc;
      }
    }

    // ---- two-body (bond) blocks: count line + one raw comment line; then two lines per pair
    {
      Tokens v = rd.next_values();
      const int nb = v.next_int();
      rd.skip_line();
      for (int i = 0; i < nb; ++i) {
        v = rd.next_values();
        need_columns(rd, v, 10);
        const int j = v.next_int() - 1, k = v.next_int() - 1;
        if (j < 0 || k < 0) fail(rd, "inconsistent force field file");
        const bool in = (j < ntypes) && (k < ntypes);
        auto set2 = [&](auto member, double val) {
          if (in) { B.tb(j, k).*member = val; B.tb(k, j).*member = val; }
        };
        if (in) {
          set2(&TwoBody::De_s, v.next_double());
          set2(&TwoBody::De_p, v.next_double());
          set2(&TwoBody::De_pp, v.next_double());
          set2(&TwoBody::p_be1, v.next_double());
          set2(&TwoBody::p_bo5, v.next_double());
          set2(&TwoBody::v13cor, v.next_double());
          set2(&TwoBody::p_bo6, v.next_double());
          set2(&TwoBody::p_ovun1, v.next_double());
        } else {
          ff.warnings_.push_back(rd.where() + ": bond block for an element index beyond the number of atom types is ignored");
        }
        v = rd.next_values();
        need_columns(rd, v, 7);
        if (in) {
          set2(&TwoBody::p_be2, v.next_double());
          set2(&TwoBody::p_bo3, v.next_double());
          set2(&TwoBody::p_bo4, v.next_double());
          v.skip();
          set2(&TwoBody::p_bo1, v.next_double());
          set2(&TwoBody::p_bo2, v.next_double());
          const double ovc = v.has_next() ? v.next_double() : 0.0;  // 8th value optional (reaxff_ffield.cpp:344-348)
          set2(&TwoBody::ovc, ovc);
          B.tb(j, k).listed = B.tb(k, j).listed = true;
        }
      }
    }

    // ---- combination rules (reaxff_ffield.cpp:352-382); every operation is IEEE-exact except pow() for `gamma`
    for (int i = 0; i < ntypes; ++i)
      for (int j = i; j < ntypes; ++j) {
        const SingleBody& si = ff.sbp_[static_cast<std::size_t>(i)];
        const SingleBody& sj = ff.sbp_[static_cast<std::size_t>(j)];
        auto both = [&](auto member, double val) { B.tb(i, j).*member = val; B.tb(j, i).*member = val; };
        both(&TwoBody::r_s, 0.5 * (sj.r_s + si.r_s));
        both(&TwoBody::r_p, 0.5 * (sj.r_pi + si.r_pi));
        both(&TwoBody::r_pp, 0.5 * (sj.r_pi_pi + si.r_pi_pi));
        both(&TwoBody::p_boc3, std::sqrt(sj.b_o_132 * si.b_o_132));
        both(&TwoBody::p_boc4, std::sqrt(sj.b_o_131 * si.b_o_131));
        both(&TwoBody::p_boc5, std::sqrt(sj.b_o_133 * si.b_o_133));
        both(&TwoBody::D, std::sqrt(sj.epsilon * si.epsilon));
        both(&TwoBody::alpha, std::sqrt(sj.alpha * si.alpha));
        both(&TwoBody::r_vdW, 2.0 * std::sqrt(sj.r_vdw * si.r_vdw));
        both(&TwoBody::gamma_w, std::sqrt(sj.gamma_w * si.gamma_w));
        both(&TwoBody::gamma, std::pow(sj.gamma * si.gamma, -1.5));
        both(&TwoBody::rcore, std::sqrt(si.rcore2 * sj.rcore2));
        both(&TwoBody::ecore, std::sqrt(si.ecore2 * sj.ecore2));
        both(&TwoBody::acore, std::sqrt(si.acore2 * sj.acore2));
        both(&TwoBody::lgcij, std::sqrt(si.lgcij * sj.lgcij));
        both(&TwoBody::lgre, 2.0 * gp.l[35] * std::sqrt(si.lgre * sj.lgre));
      }

    // ---- off-diagonal overrides
    {
      Tokens v = rd.next_values();
      const int no = v.next_int();
      for (int i = 0; i < no; ++i) {
        v = rd.next_values();
        need_columns(rd, v, static_cast<std::size_t>(8 + (lg ? 1 : 0)));
        const int j = v.next_int() - 1, k = v.next_int() - 1;
        if (j < 0 || k < 0) fail(rd, "inconsistent force field file");
        if (j < ntypes && k < ntypes) {
          auto over = [&](auto member, double val, double scale) {
            if (val > 0.0) { B.tb(j, k).*member = scale * val; B.tb(k, j).*member = scale * val; }
          };
          over(&TwoBody::D, v.next_double(), 1.0);
          over(&TwoBody::r_vdW, v.next_double(), 2.0);
          over(&TwoBody::alpha, v.next_double(), 1.0);
          over(&TwoBody::r_s, v.next_double(), 1.0);
          over(&TwoBody::r_p, v.next_double(), 1.0);
          over(&TwoBody::r_pp, v.next_double(), 1.0);
          if (lg) {
            const double val = v.next_double();
            if (val >= 0.0) { B.tb(j, k).lgcij = val; B.tb(k, j).lgcij = val; }
          }
        }
      }
    }

    // ---- three-body (valence angle) sets. The reference stores a slot count that is incremented for BOTH orientations; for
    // j==l that is the same header twice, so each set takes two slots (the second stays zero and is inert), and more than five
    // slots overrun the fixed array (Q-09). We keep only the effective sets, in the order the reference sums them.
    {
      Tokens v = rd.next_values();
      const int n3 = v.next_int();
      std::map<std::array<int, 3>, int> slots;
      for (int i = 0; i < n3; ++i) {
        v = rd.next_values();
        need_columns(rd, v, 10);
        const int j = v.next_int() - 1, k = v.next_int() - 1, l = v.next_int() - 1;
        if (j < 0 || k < 0 || l < 0) fail(rd, "inconsistent force field file");
        if (j < ntypes && k < ntypes && l < ntypes) {
          ThreeBodySet s;
          s.theta_00 = v.next_double();
          s.p_val1 = v.next_double();
          s.p_val2 = v.next_double();
          s.p_coa1 = v.next_double();  // file order: theta, p_val1, p_val2, p_coa1, p_val7, p_pen1, p_val4
          s.p_val7 = v.next_double();
          s.p_pen1 = v.next_double();
          s.p_val4 = v.next_double();
          int& a = slots[{j, k, l}];
          a += 1;
          int& b = slots[{l, k, j}];
          b += 1;  // j==l: the same counter is incremented a second time
          if (slots[{j, k, l}] > kMaxThreeBodySlots || slots[{l, k, j}] > kMaxThreeBodySlots)
            fail(rd, "too many valence-angle parameter sets for the triple " + std::to_string(j + 1) + "-" + std::to_string(k + 1) + "-" +
                         std::to_string(l + 1) + ": the reference stores them in a fixed array of " + std::to_string(kMaxThreeBodySlots) +
                         " slots, counting a j==l set twice, and reads past it (ENGINE_SPEC Q-09); at most 2 sets for j==l, 5 otherwise");
          ff.thbp_[{j, k, l}].push_back(s);
          if (j != l) ff.thbp_[{l, k, j}].push_back(s);
        }
      }
    }

    // ---- four-body (torsion) entries, including the compact 0-X-Y-0 form with its order-dependent overwrite
    {
      Tokens v = rd.next_values();
      const int n4 = v.next_int();
      std::set<std::array<int, 4>> explicit_flag;  // tor_flag
      for (int i = 0; i < n4; ++i) {
        v = rd.next_values();
        need_columns(rd, v, 9);
        const int j = v.next_int() - 1, k = v.next_int() - 1, l = v.next_int() - 1, m = v.next_int() - 1;
        if (j < -1 || k < 0 || l < 0 || m < -1) fail(rd, "inconsistent force field file");
        FourBody fb;
        fb.V1 = v.next_double();
        fb.V2 = v.next_double();
        fb.V3 = v.next_double();
        fb.p_tor1 = v.next_double();
        fb.p_cot1 = v.next_double();
        if (j >= 0 && m >= 0) {
          if (j < ntypes && k < ntypes && l < ntypes && m < ntypes) {
            explicit_flag.insert({j, k, l, m});
            explicit_flag.insert({m, l, k, j});
            ff.fbp_[{j, k, l, m}] = fb;
            ff.fbp_[{m, l, k, j}] = fb;
          }
        } else if (k < ntypes && l < ntypes) {
          for (int p = 0; p < ntypes; ++p)
            for (int o = 0; o < ntypes; ++o) {
              if (!explicit_flag.count({p, k, l, o})) ff.fbp_[{p, k, l, o}] = fb;
              if (!explicit_flag.count({o, l, k, p})) ff.fbp_[{o, l, k, p}] = fb;
            }
        }
      }
    }

    // ---- hydrogen bonds: the whole block may be absent (the only truncation LAMMPS and we both accept)
    {
      auto first = rd.next_line();
      if (!first) {
        ff.warnings_.push_back("ReaxFF parameter file has no hydrogen bond parameters");
      } else {
        Tokens v = std::move(*first);
        const int nh = v.next_int();
        for (int i = 0; i < nh; ++i) {
          v = rd.next_values();
          need_columns(rd, v, 7);
          const int j = v.next_int() - 1, k = v.next_int() - 1, l = v.next_int() - 1;
          if (j < 0 || k < 0 || l < 0) fail(rd, "inconsistent force field file");
          if (j < ntypes && k < ntypes && l < ntypes) {
            HBondParams h;
            h.r0_hb = v.next_double();
            h.p_hb1 = v.next_double();
            h.p_hb2 = v.next_double();
            h.p_hb3 = v.next_double();
            ff.hbp_[{j, k, l}] = h;
          }
        }
      }
    }
  } catch (EofReached&) {
    if (!opt.reproduce_lammps_leniency)
      throw FfieldError(rd.name() + ": unexpected end of file at line " + std::to_string(rd.lineno()) +
                        " (strict mode; pinned LAMMPS only warns and leaves the remaining tables zero)");
    ff.warnings_.push_back(rd.name() + ": unexpected end of file (accepted: reproduce_lammps_leniency)");
  } catch (FfieldError&) {
    throw;
  }

  // ---- derived control values and compat flags
  if (ff.gp_.l.size() < 38 || ff.sbp_.empty())
    throw FfieldError(rd.name() + ": file ends before the general parameters and atom blocks are complete");
  const auto& L = ff.gp_.l;
  ff.ctl_.bo_cut = 0.01 * L[29];
  ff.ctl_.nonb_low = L[11];
  ff.ctl_.nonb_cut = L[12];
  ff.gp37_ = gp37_forces_stabilisation(L);
  ff.c2_active_ = c2_correction_active(L);
  for (int i = 0; i < ff.num_types(); ++i)
    for (int j = 0; j < ff.num_types(); ++j) {
      TwoBody& t = B.tb(i, j);
      t.triple_bond_stabilisation = triple_bond_stabilisation_pair(ff.gp37_, ff.sbp_[static_cast<std::size_t>(i)].flags, ff.sbp_[static_cast<std::size_t>(j)].flags);
    }
  return std::move(ff);
}

ForceField read_force_field_file(const std::string& path, const FfieldOptions& opt) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw FfieldError("The ReaxFF parameter file " + path + " cannot be opened");
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return parse_force_field(text, path, opt);
}

}  // namespace reaxmetal
