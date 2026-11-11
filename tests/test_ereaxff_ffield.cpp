// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// eReaxFF parser additions (plan task 3.1). SYNTHETIC force field (invented numbers, no upstream data); the column positions come from
// reading the legacy reader (Margaret/eReaxFF_src/reac.f:4516-4524 atom lines, 4668-4670 bond lines), not from running this parser.
// With a path argument (or REAXMETAL_EREAXFF_FFIELD) the same checks also run on a real eReaxFF force field: values are then
// compared with the numbers read off the Jamil-target file by hand (see the comments).
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>

#include "reaxmetal/forcefield.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

namespace {

std::string num(double v) {
  std::ostringstream o;
  o.precision(17);
  o << v;
  return o.str();
}

// two types: "C" and "El"; one bond block (C-C only), so every pair involving El is unlisted
std::string make(double gp38) {
  std::ostringstream o;
  o << "synthetic eReaxFF force field\n39 ! Number of general parameters\n";
  for (int i = 0; i < 39; ++i) {
    double v = 1.0 + 0.5 * i;
    if (i == 5) v = 0.0;
    if (i == 11) v = 0.0;
    if (i == 12) v = 10.0;
    if (i == 29) v = 0.5;
    if (i == 35) v = 1.0;
    if (i == 37) v = gp38;
    o << "  " << num(v) << " !p" << i << "\n";
  }
  o << "2 ! Nr of atoms\n c1\n c2\n c3\n";
  o << " C 1.5 4.0 12.0 1.75 0.25 0.9 1.25 4.0\n"
       "  9.0 2.0 4.0 20.0 0.8 5.5 4.5 0.0\n"
       "  1.25 0.5 181.0 0.1 0.2 0.3 0.4 22.0\n"
       "  0.7 2.5 0.0027 4.0 3.0 0.0 0.0 0.0\n";
  o << " El -0.1 0.0 1.0 1.3 0.005 0.25 -0.1 0.0\n"
       "  2.0 2.0 4.0 0.0 0.11 1.0002 1.0009 10.0\n"
       "  -0.1 0.0 -2.37 6.4918 8.5961 0.2368 2.0 3.5\n"
       "  -5.0 3.1873 0.16 6.2998 2.5791 0.0 0.0 0.0\n";
  o << "1 ! Nr of bonds\n comment\n"
       "  1  1  78.4 115.4 68.2 0.58 -0.29 1.0 35.0 0.51\n"
       "  3.856 -0.164 8.2326 1.0 -0.0585 6.7997 1.0 0.1481\n";
  o << "0 ! off-diagonal\n0 ! angles\n0 ! torsions\n0 ! hbonds\n";
  return o.str();
}

FfieldOptions eopt(double taper = std::numeric_limits<double>::quiet_NaN()) {
  FfieldOptions o;
  o.ereaxff = true;
  o.electron_taper = taper;
  return o;
}

void check_electron_columns(const ForceField& ff, int c, int el) {
  // legacy columns: atom line 2 col 5 = pxel2, line 3 col 3 = vincr, line 3 col 8 = alphag, line 4 col 3 = betael
  RM_CHECK(ff.single(c).pxel2 == 0.8);
  RM_CHECK(ff.single(c).vincr == 181.0);
  RM_CHECK(ff.single(c).alphag == 22.0);
  RM_CHECK(ff.single(c).betael == 0.0027);
  RM_CHECK(ff.single(el).pxel2 == 0.11);
  RM_CHECK(ff.single(el).vincr == -2.37);
  RM_CHECK(ff.single(el).alphag == 3.5);
  RM_CHECK(ff.single(el).betael == 0.16);
  // stock columns must be unchanged by the new reads
  RM_CHECK(ff.single(c).chi == 5.5);
  RM_CHECK(ff.single(c).p_ovun5 == 20.0);
  RM_CHECK(ff.single(c).bcut_acks2 == 0.4);
  RM_CHECK(ff.single(c).valency_val == 4.0);
}

}  // namespace

int main(int argc, char** argv) {
  // --- synthetic file, eReaxFF mode with an explicit taper while gp[38] is 0 (the Jamil file as written)
  {
    const ForceField ff = parse_force_field(make(0.0), "syn", eopt(10.0));
    RM_CHECK(ff.ereaxff());
    RM_CHECK(ff.electron_taper() == 10.0);
    check_electron_columns(ff, 0, 1);
    RM_CHECK(ff.pair(0, 0).pxel1 == 0.1481);
    RM_CHECK(ff.pair(0, 1).pxel1 == 0.0 && !ff.pair(0, 1).listed);
    RM_CHECK(!ff.gp37_stabilisation());
    RM_CHECK(ff.single(0).flags.core_charge == 6.0 && ff.single(0).flags.share_sign == -1 && !ff.single(0).flags.is_electron);
    RM_CHECK(ff.single(1).flags.is_electron && ff.single(1).flags.core_charge == 0.0 && ff.single(1).flags.share_sign == 0);
  }
  // --- taper from the file when no override is given; explicit override wins over the file
  {
    RM_CHECK(parse_force_field(make(10.0), "syn", eopt()).electron_taper() == 10.0);
    RM_CHECK(parse_force_field(make(10.0), "syn", eopt(5.7)).electron_taper() == 5.7);
  }
  // --- gp[38] = 0 without an override is rejected in eReaxFF mode (the electron term would silently vanish); stock mode is unaffected
  RM_EXPECT_THROW(parse_force_field(make(0.0), "syn", eopt()), FfieldError);
  RM_EXPECT_THROW(parse_force_field(make(10.0), "syn", eopt(0.0)), FfieldError);
  RM_EXPECT_THROW(parse_force_field(make(10.0), "syn", eopt(-3.0)), FfieldError);
  { const ForceField ff = parse_force_field(make(0.0), "syn"); RM_CHECK(!ff.ereaxff()); check_electron_columns(ff, 0, 1); }
  // --- the gp[37] == 2 triple-bond predicate: active in stock mode, disabled in eReaxFF mode (where the slot is the taper radius)
  {
    const ForceField stock = parse_force_field(make(2.0), "syn");
    RM_CHECK(stock.gp37_stabilisation());
    RM_CHECK(stock.pair(0, 0).triple_bond_stabilisation);
    const ForceField e = parse_force_field(make(2.0), "syn", eopt());
    RM_CHECK(e.electron_taper() == 2.0);
    RM_CHECK(!e.gp37_stabilisation());
    RM_CHECK(!e.pair(0, 0).triple_bond_stabilisation);
  }
  // --- strict missing-pair rule (Q-12): El pairs have no bond block and are exempt only in eReaxFF mode
  {
    const ForceField stock = parse_force_field(make(10.0), "syn");
    const int both[2] = {0, 1};
    const int el_only[1] = {1};
    RM_EXPECT_THROW(stock.require_bond_blocks(both), FfieldError);
    RM_EXPECT_THROW(stock.require_bond_blocks(el_only), FfieldError);
    const ForceField e = parse_force_field(make(10.0), "syn", eopt());
    e.require_bond_blocks(both);     // must not throw: C-C is listed, C-El and El-El are exempt
    e.require_bond_blocks(el_only);
  }
  // --- real eReaxFF force field (optional): numbers read off the Jamil-target file np_export/ffield by hand
  const char* real = argc > 1 ? argv[1] : std::getenv("REAXMETAL_EREAXFF_FFIELD");
  if (real && *real) {
    const ForceField ff = read_force_field_file(real, eopt(10.0));
    const auto c = ff.match_element("C"), el = ff.match_element("El");
    RM_CHECK(c.size() == 1 && el.size() == 1);
    if (c.size() == 1 && el.size() == 1) {
      const SingleBody& C = ff.single(c[0]);
      const SingleBody& E = ff.single(el[0]);
      RM_CHECK(C.pxel2 == 0.8 && C.vincr == 181.0 && C.alphag == 22.0 && C.betael == 0.0027);  // C block, lines 46-49
      RM_CHECK(E.pxel2 == 0.0 && E.vincr == -2.37 && E.alphag == 3.5 && E.betael == 0.16);      // El block, lines 94-97
      RM_CHECK(ff.pair(c[0], c[0]).pxel1 == 0.1481);                                           // C-C bond block, line 109
      RM_CHECK(E.flags.is_electron && C.flags.core_charge == 6.0);
      RM_CHECK(ff.electron_taper() == 10.0);
    }
    // the file as written (gp[38] may be 0) must be rejected without an explicit taper only if gp[38] is 0
    try { (void)read_force_field_file(real, eopt()); } catch (const FfieldError&) {}
    RM_CHECK(read_force_field_file(real, FfieldOptions{}).num_types() == ff.num_types());
    std::printf("real ffield checks done on %s\n", real);
  }
  return rmtest::finish("ereaxff_ffield");
}
