// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Unit tests of the ReaxFF force-field / control-file readers with SYNTHETIC files (no upstream data). Expected values are
// derived by hand from the file layout and from the cited behaviour of pinned LAMMPS, not from running this parser.
// Equality with the tables LAMMPS actually stores on the 11 bundled files is test `ffield_tables` / tools/reaxref/parse_diff.py.
#include <cmath>
#include <sstream>
#include <string>
#include <utility>

#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/sha256.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

namespace {

struct Spec {
  // two types: 1 = "C" (mass 12.0), 2 = "H" (mass 1.008); all numbers chosen to be exactly representable where it matters
  std::string atom1_name = "C", atom2_name = "H";
  double mass1 = 12.0, mass2 = 1.008;
  double valency_val1 = 4.0, valency_boc1 = 4.0;  // first-row override test changes valency_val1
  bool lg = false;
  std::string first_line = "synthetic force field";
  std::string bonds_extra;     // extra bond blocks (appended after the standard ones)
  int nbonds = 3;
  std::string angles = "  1  1  1  67.0 22.0 1.5 0.0 1.8 15.0 1.8\n  2  1  2  70.0 25.0 3.4 0.0 0.0 0.0 3.0\n";
  int nangles = 2;
  std::string torsions = "  1  1  1  1  -0.25 11.5 0.19 -4.7 -2.2 0.0 0.0\n";
  int ntorsions = 1;
  bool hbond_block = true;
  std::string hbonds = "  2  1  2  2.0 -3.0 3.0 1.5\n";
  int nhbonds = 1;
  int nglobal = 39;
  std::string tail_comment;    // text appended to the very end
  bool drop_pair_12 = false;   // do not list the C-H bond block
  double gamma_w1 = 2.0;
  double phb1 = 0.0;           // p_hbond column of atom 1 (stored as (int) truncation)
};

std::string num(double v) {
  std::ostringstream o;
  o.precision(17);
  o << v;
  return o.str();
}

std::string make(const Spec& s) {
  std::ostringstream o;
  o << s.first_line << "\n";
  o << s.nglobal << " ! Number of general parameters\n";
  for (int i = 0; i < s.nglobal; ++i) {
    double v = 1.0 + i * 0.5;
    if (i == 5) v = 0.0;      // p_lp3: C2 correction inactive
    if (i == 11) v = 0.0;     // swa
    if (i == 12) v = 10.0;    // swb
    if (i == 29) v = 0.5;     // bo_cut = 0.01 * gp[29]
    if (i == 35) v = 1.0;     // lg scaling
    if (i == 37) v = 0.0;     // gp37
    o << "  " << num(v) << " !p" << i << "\n";
  }
  o << "2 ! Nr of atoms; cov.r; valency;a.m;Rvdw;Evdw;gammaEEM;cov.r2;#\n  alpha;gammavdW;valency;Eunder;Eover;chiEEM;etaEEM;n.u.\n  cov r3;Elp;Heat inc.;n.u.;n.u.;n.u.;n.u.\n  ov/un;val1;n.u.;val3,vval4\n";
  auto atom = [&](const std::string& nm, double mass, double vval, double vboc, double gw, double eta, double phb) {
    o << " " << nm << " 1.5 4.0 " << num(mass) << " 1.75 0.25 0.9 1.25 4.0\n";
    o << "  9.0 " << num(gw) << " " << num(vboc) << " 20.0 0.0 5.5 " << num(eta) << " " << num(phb) << "\n";
    o << "  1.25 0.5 0.0 0.1 0.2 0.3 0.4 0.0\n";
    o << "  0.7 2.5 0.0 " << num(vval) << " 3.0 0.0 0.0 0.0\n";
    if (s.lg) o << "  0.4 1.5\n";
  };
  atom(s.atom1_name, s.mass1, s.valency_val1, s.valency_boc1, s.gamma_w1, 3.5, s.phb1);
  atom(s.atom2_name, s.mass2, 1.0, 1.0, 2.0, 7.0, 1.0);
  const int nb = s.nbonds - (s.drop_pair_12 ? 1 : 0);
  o << nb << " ! Nr of bonds; Edis1;LPpen;n.u.;pbe1;pbo5;13corr;pbo6\n   pbe2;pbo3;pbo4;n.u.;pbo1;pbo2;ovcorr\n";
  auto bond = [&](int a, int b, double de) {
    o << "  " << a << "  " << b << " " << num(de) << " 100.0 80.0 -0.8 -0.4 1.0 37.0 0.4\n";
    o << "    0.45 -0.1 9.2 1.0 -0.075 6.8 1.0 0.0\n";
  };
  bond(1, 1, 150.0);
  if (!s.drop_pair_12) bond(1, 2, 170.0);
  bond(2, 2, 156.0);
  o << s.bonds_extra;
  o << "1 ! Nr of off-diagonal terms; at1;at2;Dij;RvdW;alfa;ro(sigma);ro(pi);ro(pipi)\n  1  2  0.25  1.5  9.5  1.1  -1.0  -1.0" << (s.lg ? "  0.3" : "") << "\n";
  o << s.nangles << " ! Nr of angles\n" << s.angles;
  o << s.ntorsions << " ! Nr of torsions\n" << s.torsions;
  if (s.hbond_block) o << s.nhbonds << " ! Nr of hydrogen bonds\n" << s.hbonds;
  o << s.tail_comment;
  return o.str();
}

bool throws_containing(const std::string& text, const std::string& needle, FfieldOptions opt = {}) {
  try {
    (void)parse_force_field(text, "synthetic", opt);
  } catch (const FfieldError& e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

}  // namespace

static const char* g_block = "start";
#define BLOCK(name) g_block = name

int main() try {
  BLOCK("basic values, derived values, flags");
  // ---------------------------------------------------------------- basic values, derived values, flags
  {
    Spec s;
    s.atom1_name = "carbon";  // longer than 3 characters, mixed case
    const ForceField ff = parse_force_field(make(s), "synthetic");
    RM_CHECK(ff.num_types() == 2);
    RM_CHECK(ff.single(0).name == "CAR" && ff.single(1).name == "H");
    RM_CHECK(ff.single(0).mass == 12.0);
    RM_CHECK(ff.single(0).eta == 7.0);              // eta stored as 2 x file value (3.5)
    RM_CHECK(ff.single(0).nlp_opt == 0.5 * (4.0 - 4.0));
    RM_CHECK(ff.single(0).p_hbond == 0);
    RM_CHECK(ff.single(0).chi == 5.5);
    RM_CHECK(ff.global().l.size() == 39 && ff.global().vdw_type == 1);  // shielding, no inner wall
    RM_CHECK(ff.file_control().nonb_cut == 10.0 && ff.file_control().nonb_low == 0.0 && ff.file_control().bo_cut == 0.01 * 0.5);
    // combination rules: IEEE-exact operations
    const TwoBody& hh = ff.pair(1, 1);
    RM_CHECK(hh.r_s == 0.5 * (1.5 + 1.5) && hh.r_p == 0.5 * (1.25 + 1.25));
    RM_CHECK(hh.D == std::sqrt(0.25 * 0.25) && hh.alpha == std::sqrt(9.0 * 9.0) && hh.r_vdW == 2.0 * std::sqrt(1.75 * 1.75));
    RM_CHECK(hh.p_boc3 == std::sqrt(0.2 * 0.2) && hh.p_boc4 == std::sqrt(0.1 * 0.1) && hh.p_boc5 == std::sqrt(0.3 * 0.3));
    // gamma = (g_i g_j)^-1.5 through libm pow: compare with the same expression, not with a literal
    RM_CHECK(hh.gamma == std::pow(0.9 * 0.9, -1.5));
    RM_CHECK(ff.pair(0, 1).gamma == ff.pair(1, 0).gamma && ff.pair(0, 1).De_s == 170.0 && ff.pair(1, 0).De_s == 170.0);
    // off-diagonal override (1,2): D=0.25, r_vdW = 2*1.5, alpha 9.5, r_s = 1.1; ro(pi) = -1 (not > 0) leaves the combined value
    RM_CHECK(ff.pair(0, 1).D == 0.25 && ff.pair(0, 1).r_vdW == 3.0 && ff.pair(0, 1).alpha == 9.5 && ff.pair(0, 1).r_s == 1.1);
    RM_CHECK(ff.pair(0, 1).r_p == 0.5 * (1.25 + 1.25) && ff.pair(1, 0).r_s == 1.1);
    RM_CHECK(ff.pair(0, 1).listed && ff.pair(1, 1).listed);
    // compat flags
    RM_CHECK(!ff.single(0).flags.c2_species);   // "CAR" != "C"
    RM_CHECK(!ff.c2_correction_active() && !ff.gp37_stabilisation());
  }
  {
    Spec s;  // symbol "c" -> "C": the C2 species flag
    s.atom1_name = "c";
    const ForceField ff = parse_force_field(make(s), "synthetic");
    RM_CHECK(ff.single(0).name == "C" && ff.single(0).flags.c2_species);
    RM_CHECK(!ff.single(1).flags.c2_species);
    RM_CHECK(ff.match_element("c") == std::vector<int>{0} && ff.match_element("h") == std::vector<int>{1});
    RM_CHECK(ff.match_element("Cl").empty() && ff.match_element("CARBON").empty());
  }

  BLOCK("p_hbond truncation");
  for (const auto& [value, expect] : {std::pair{1.9, 1}, std::pair{2.7, 2}, std::pair{-0.9, 0}, std::pair{2.0, 2}}) {
    Spec s;
    s.phb1 = value;
    RM_CHECK(parse_force_field(make(s), "synthetic").single(0).p_hbond == expect);   // (int) cast, not rounding
  }
  BLOCK("first-row valency override (mass < 21 only)");
  // ---------------------------------------------------------------- first-row valency override (mass < 21 only)
  {
    Spec s;
    s.valency_val1 = 3.0;  // != valency_boc1 (4.0), mass 12 < 21
    const ForceField ff = parse_force_field(make(s), "synthetic");
    RM_CHECK(ff.single(0).valency_val == 4.0);
    bool warned = false;
    for (const auto& w : ff.warnings()) warned |= w.find("Changed valency_val to valency_boc for C") != std::string::npos;
    RM_CHECK(warned);
    Spec heavy = s;
    heavy.mass1 = 32.06;  // heavy: no override
    RM_CHECK(parse_force_field(make(heavy), "synthetic").single(0).valency_val == 3.0);
    Spec edge = s;
    edge.mass1 = 21.0;  // exactly 21 is neither light nor heavy
    RM_CHECK(parse_force_field(make(edge), "synthetic").single(0).valency_val == 3.0);
  }

  BLOCK("triple-bond stabilisation pair flag from the masses");
  // ---------------------------------------------------------------- triple-bond stabilisation pair flag from the masses
  {
    Spec s;
    s.mass1 = 12.0;
    s.atom2_name = "O";
    s.mass2 = 15.9990;
    const ForceField ff = parse_force_field(make(s), "synthetic");
    RM_CHECK(ff.pair(0, 1).triple_bond_stabilisation && ff.pair(1, 0).triple_bond_stabilisation);
    RM_CHECK(!ff.pair(0, 0).triple_bond_stabilisation);
    s.mass2 = 15.999000000001;  // exact equality is required
    RM_CHECK(!parse_force_field(make(s), "synthetic").pair(0, 1).triple_bond_stabilisation);
  }

  BLOCK("three-body sets (Q-09)");
  // ---------------------------------------------------------------- three-body sets (Q-09)
  {
    Spec s;  // j==l single set -> exactly one effective set; mirrored explicit lines for j!=l give each orientation both sets in file order
    s.angles =
        "  1  1  1  67.0 22.0 1.5 0.0 1.8 15.0 1.8\n"
        "  1  2  2  70.0 25.0 3.4 0.1 0.2 0.3 3.0\n"
        "  2  2  1  71.0 26.0 3.5 0.4 0.5 0.6 3.1\n";
    s.nangles = 3;
    const ForceField ff = parse_force_field(make(s), "synthetic");
    RM_CHECK(ff.three_body(0, 0, 0).size() == 1);
    RM_CHECK(ff.three_body(0, 0, 0)[0].theta_00 == 67.0 && ff.three_body(0, 0, 0)[0].p_val2 == 1.5 && ff.three_body(0, 0, 0)[0].p_coa1 == 0.0 &&
             ff.three_body(0, 0, 0)[0].p_val7 == 1.8 && ff.three_body(0, 0, 0)[0].p_pen1 == 15.0 && ff.three_body(0, 0, 0)[0].p_val4 == 1.8);
    // (1,2,2) then (2,2,1): both orientations hold [set from line 2, set from line 3]
    RM_CHECK(ff.three_body(0, 1, 1).size() == 2 && ff.three_body(1, 1, 0).size() == 2);
    RM_CHECK(ff.three_body(0, 1, 1)[0].theta_00 == 70.0 && ff.three_body(0, 1, 1)[1].theta_00 == 71.0);
    RM_CHECK(ff.three_body(1, 1, 0)[0].theta_00 == 70.0 && ff.three_body(1, 1, 0)[1].theta_00 == 71.0);
    RM_CHECK(ff.three_body(1, 0, 1).empty());
    // file column order: theta, p_val1, p_val2, p_coa1, p_val7, p_pen1, p_val4
    RM_CHECK(ff.three_body(0, 1, 1)[0].p_val1 == 25.0 && ff.three_body(0, 1, 1)[0].p_coa1 == 0.1 && ff.three_body(0, 1, 1)[0].p_val7 == 0.2 &&
             ff.three_body(0, 1, 1)[0].p_pen1 == 0.3 && ff.three_body(0, 1, 1)[0].p_val4 == 3.0);
  }
  {
    Spec two;  // two sets for a j==l triple: 4 reference slots, accepted
    two.angles = "  1  1  1  67.0 22.0 1.5 0.0 1.8 15.0 1.8\n  1  1  1  68.0 23.0 1.6 0.0 1.9 16.0 1.9\n";
    two.nangles = 2;
    const ForceField ff = parse_force_field(make(two), "synthetic");
    RM_CHECK(ff.three_body(0, 0, 0).size() == 2 && ff.three_body(0, 0, 0)[1].theta_00 == 68.0);
    Spec three = two;  // three sets: the reference counts 6 slots and reads past its 5-slot array
    three.angles += "  1  1  1  69.0 24.0 1.7 0.0 2.0 17.0 2.0\n";
    three.nangles = 3;
    RM_CHECK(throws_containing(make(three), "ENGINE_SPEC Q-09"));
    Spec six;  // j != l: five sets are the maximum
    six.angles.clear();
    for (int i = 0; i < 5; ++i) six.angles += "  1  2  2  70.0 25.0 3.4 0.0 0.0 0.0 3.0\n";
    six.nangles = 5;
    RM_CHECK(!throws_containing(make(six), "Q-09"));
    RM_CHECK(parse_force_field(make(six), "synthetic").three_body(0, 1, 1).size() == 5);
    six.angles += "  1  2  2  70.0 25.0 3.4 0.0 0.0 0.0 3.0\n";
    six.nangles = 6;
    RM_CHECK(throws_containing(make(six), "Q-09"));
  }

  BLOCK("torsions: explicit and compact 0-X-Y-0, order dependence");
  // ---------------------------------------------------------------- torsions: explicit and compact 0-X-Y-0, order dependence
  {
    Spec s;
    s.torsions =
        "  0  1  1  0  -0.10 10.0 0.1 -4.0 -2.0 0.0 0.0\n"   // wildcard first: fills every (p,1,1,o) not yet explicit
        "  1  1  1  1  -0.25 11.5 0.19 -4.7 -2.2 0.0 0.0\n"   // explicit afterwards overwrites that slot
        "  2  1  1  2  -0.50 12.5 0.29 -4.8 -2.3 0.0 0.0\n"   // explicit
        "  0  1  1  0  -0.60 13.5 0.39 -4.9 -2.4 0.0 0.0\n";  // second wildcard: overwrites wildcard slots, not explicit ones
    s.ntorsions = 4;
    const ForceField ff = parse_force_field(make(s), "synthetic");
    RM_CHECK(ff.four_body(0, 0, 0, 0) && ff.four_body(0, 0, 0, 0)->V1 == -0.25);  // explicit survives the later wildcard
    RM_CHECK(ff.four_body(1, 0, 0, 1) && ff.four_body(1, 0, 0, 1)->V1 == -0.5);
    RM_CHECK(ff.four_body(0, 0, 0, 1) && ff.four_body(0, 0, 0, 1)->V1 == -0.6);   // wildcard slot: last wildcard wins
    RM_CHECK(ff.four_body(1, 0, 0, 0) && ff.four_body(1, 0, 0, 0)->V1 == -0.6);
    RM_CHECK(ff.four_body(0, 1, 1, 0) == nullptr);                                 // only the (p,k,l,o) pattern with k=l=type1
    RM_CHECK(ff.four_body(0, 0, 0, 0)->p_tor1 == -4.7 && ff.four_body(0, 0, 0, 0)->p_cot1 == -2.2);
    Spec t;  // symmetric assignment of an explicit entry
    t.torsions = "  1  1  1  2  -0.25 11.5 0.19 -4.7 -2.2 0.0 0.0\n";
    t.ntorsions = 1;
    const ForceField f2 = parse_force_field(make(t), "synthetic");
    RM_CHECK(f2.four_body(0, 0, 0, 1) && f2.four_body(1, 0, 0, 0) && f2.four_body(1, 0, 0, 1) == nullptr);
  }

  BLOCK("hydrogen bonds");
  // ---------------------------------------------------------------- hydrogen bonds
  {
    Spec s;
    const ForceField ff = parse_force_field(make(s), "synthetic");
    RM_CHECK(ff.hbond(1, 0, 1) && ff.hbond(1, 0, 1)->r0_hb == 2.0 && ff.hbond(1, 0, 1)->p_hb1 == -3.0 && ff.hbond(1, 0, 1)->p_hb3 == 1.5);
    RM_CHECK(ff.hbond(0, 1, 0) == nullptr);   // not mirrored
    Spec none = s;
    none.hbond_block = false;  // a missing H-bond block is accepted (as LAMMPS) with a warning
    const ForceField f2 = parse_force_field(make(none), "synthetic");
    bool warned = false;
    for (const auto& w : f2.warnings()) warned |= w.find("no hydrogen bond parameters") != std::string::npos;
    RM_CHECK(warned && f2.hbond(1, 0, 1) == nullptr);
  }

  BLOCK("errors: format, numbers, truncation, lgvdw");
  // ---------------------------------------------------------------- errors: format, numbers, truncation, lgvdw
  {
    Spec s;
    s.first_line = "39 ! Number of general parameters";   // the count line where the comment must be
    RM_CHECK(throws_containing(make(s), "first line of ReaxFF potential file must be a comment"));
    s = Spec{};
    RM_CHECK(throws_containing(make(s).replace(make(s).find("1.5 4.0 12"), 3, "abc"), "not a valid floating point"));
    s = Spec{};
    s.nglobal = 30;
    RM_CHECK(throws_containing(make(s), "general parameters"));
    s = Spec{};
    std::string text = make(s);
    const auto pos = text.find("0.9 1.25 4.0");  // atom 1 line 1, swap a value for inf
    RM_CHECK(pos != std::string::npos);
    text.replace(pos, 3, "inf");
    RM_CHECK(throws_containing(text, "not finite"));
    // a file with lgvdw-style atom blocks read without lgvdw is rejected when line 5 would be consumed as the next element
    Spec lg;
    lg.lg = true;
    FfieldOptions with_lg;
    with_lg.lgvdw = true;
    RM_CHECK(parse_force_field(make(lg), "synthetic", with_lg).single(0).lgcij == 0.4);
    RM_CHECK(parse_force_field(make(lg), "synthetic", with_lg).single(0).lgre == 1.5);
    RM_CHECK(parse_force_field(make(lg), "synthetic", with_lg).pair(0, 1).lgcij == 0.3);   // off-diagonal lg column overrides (>= 0)
    RM_CHECK(parse_force_field(make(lg), "synthetic", with_lg).pair(0, 0).lgcij == std::sqrt(0.4 * 0.4));
    RM_CHECK(parse_force_field(make(lg), "synthetic", with_lg).pair(0, 0).lgre == 2.0 * 1.0 * std::sqrt(1.5 * 1.5));  // gp[35] = 1
    bool lg_without = false;
    try { (void)parse_force_field(make(lg), "synthetic"); lg_without = true; } catch (const FfieldError&) {}
    RM_CHECK(!lg_without);
    // inconsistent vdW: gamma_w <= 0.5 and no inner wall for an element
    Spec bad;
    bad.gamma_w1 = 0.25;
    RM_CHECK(throws_containing(make(bad), "inconsistent van der Waals"));
    // truncated file: strict rejects, lenient accepts with a warning
    Spec full;
    std::string t2 = make(full);
    const std::string cut = t2.substr(0, t2.find("Nr of angles"));
    RM_CHECK(throws_containing(cut, "unexpected end of file"));
    FfieldOptions lenient;
    lenient.reproduce_lammps_leniency = true;
    const ForceField part = parse_force_field(cut, "synthetic", lenient);
    RM_CHECK(!part.warnings().empty() && part.num_types() == 2);
    // CRLF line endings and blank / comment-only lines between records
    std::string crlf;
    for (char c : make(Spec{})) { if (c == '\n') crlf += "\r\n"; else crlf += c; }
    RM_CHECK(parse_force_field(crlf, "synthetic").pair(0, 1).De_s == 170.0);
  }

  BLOCK("Q-12: absent bond-pair block");
  // ---------------------------------------------------------------- Q-12: absent bond-pair block
  {
    Spec s;
    s.drop_pair_12 = true;
    const ForceField ff = parse_force_field(make(s), "synthetic");    // the file itself parses (as in LAMMPS) ...
    RM_CHECK(!ff.pair(0, 1).listed && ff.pair(0, 0).listed);
    RM_CHECK(ff.pair(0, 1).De_s == 0.0 && ff.pair(0, 1).p_bo1 == 0.0);   // ... with zero-filled parameters
    const int both[] = {0, 1};
    RM_EXPECT_THROW(ff.require_bond_blocks(both), FfieldError);          // ... but using the pair is refused
    const int only_c[] = {0};
    ff.require_bond_blocks(only_c);                                      // elements whose blocks exist are fine
  }

  BLOCK("control file");
  // ---------------------------------------------------------------- control file
  {
    std::vector<std::string> w;
    const ControlParams c = parse_control_text("nbrhood_cutoff 4.5\n\nhbond_cutoff 7.0\nthb_cutoff 0.002\nthb_cutoff_sq 2e-5 ! comment words are ignored\n"
                                               "tabulate_long_range 0\nwrite_freq 0\nsimulation_name run1\nbond_graph_cutoff 0.4\n", "ctl", &w);
    RM_CHECK(c.bond_cut == 4.5 && c.hbond_cut == 7.0 && c.thb_cut == 0.002 && c.thb_cutsq == 2e-5 && c.bg_cut == 0.4 && c.tabulate == 0);
    RM_CHECK(w.size() == 1 && w[0].find("simulation_name") != std::string::npos);
    const ControlParams d = parse_control_text("", "empty");
    RM_CHECK(d.bond_cut == 5.0 && d.hbond_cut == 7.5 && d.thb_cut == 0.001 && d.thb_cutsq == 0.00001 && d.bg_cut == 0.3);
    RM_EXPECT_THROW(parse_control_text("bogus_keyword 1\n", "x"), FfieldError);
    RM_EXPECT_THROW(parse_control_text("nbrhood_cutoff\n", "x"), FfieldError);
    RM_EXPECT_THROW(parse_control_text("nbrhood_cutoff abc\n", "x"), FfieldError);
    RM_EXPECT_THROW(parse_control_text("tabulate_long_range 1.5\n", "x"), FfieldError);
  }

  BLOCK("canonical dump and SHA-256");
  // ---------------------------------------------------------------- canonical dump and SHA-256
  {
    const ForceField ff = parse_force_field(make(Spec{}), "synthetic");
    const std::string full = canonical_table_dump(ff), portable = canonical_table_dump(ff, true);
    RM_CHECK(full == canonical_table_dump(parse_force_field(make(Spec{}), "again")));   // deterministic
    RM_CHECK(full != portable && full.size() > portable.size());                          // gamma omitted in the portable form
    RM_CHECK(full.rfind("FFIELD-TABLES 1\nGP 39 1\n", 0) == 0);
    RM_CHECK(full.find("HB 1 0 1 2 -3 3 1.5\n") != std::string::npos);
    // FIPS 180-4 test vectors
    RM_CHECK(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    RM_CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    RM_CHECK(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    RM_CHECK(sha256_hex(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  }
  return rmtest::finish("ffield_parser");
} catch (const std::exception& e) {
  std::fprintf(stderr, "UNEXPECTED EXCEPTION in block '%s': %s\n", g_block, e.what());
  return 1;
}
