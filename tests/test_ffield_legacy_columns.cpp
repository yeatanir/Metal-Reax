// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Parsed force-field values of Metal-Reax (eReaxFF mode) against the values the LEGACY fixed-column reader takes from the same file.
// The expected values come from tests/python/legacy_ffield_slice.py, an independent re-implementation of the legacy formats
// (reac.f ffinpt: (f10.4), (1x,a2,10f9.4), (3x,10f9.4), (2i3,8f9.4), (6x,8f9.4)). A mismatch means the two codes would run different parameters,
// e.g. a value wider than its 9-character column (the legacy reader clips it silently, a token parser does not).
// usage: test_ffield_legacy_columns <ffield> <expected.txt>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "reaxmetal/forcefield.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

namespace {
bool close(double a, double b) { return std::fabs(a - b) <= 1e-9 * std::fmax(1.0, std::fabs(b)); }
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s ffield expected\n", argv[0]); return 2; }
  FfieldOptions fo; fo.ereaxff = true; fo.electron_taper = 10.0;   // explicit taper: the files carry 0.0 in general parameter 38 as written
  const ForceField ff = read_force_field_file(argv[1], fo);
  std::ifstream in(argv[2]); std::string line;
  int atom = 0, atoms_checked = 0, bonds_checked = 0, gen_checked = 0, bad = 0;
  while (std::getline(in, line)) {
    std::istringstream ls(line); std::string k; ls >> k;
    if (k == "GEN") {
      int n; ls >> n; const auto& l = ff.global().l;
      for (int i = 0; i < n; ++i) { double v; ls >> v;
        // general parameter 38 (index 37) is the legacy electron taper slot; Metal-Reax keeps the file value in gp.l as well
        if (static_cast<std::size_t>(i) < l.size() && !close(l[static_cast<std::size_t>(i)], v)) { std::printf("general[%d]: parsed %.6f legacy %.6f\n", i, l[static_cast<std::size_t>(i)], v); ++bad; }
        ++gen_checked; }
    } else if (k == "ATOM") {
      std::string name; ls >> name; std::vector<double> v; double x; while (ls >> x) v.push_back(x);
      RM_CHECK(v.size() == 32);
      const SingleBody& s = ff.single(atom);
      const double got[32] = {
          s.r_s, s.valency, s.mass, s.r_vdw, s.epsilon, s.gamma, s.r_pi, s.valency_e,
          s.alpha, s.gamma_w, s.valency_boc, s.p_ovun5, s.pxel2, s.chi, s.eta / 2.0, static_cast<double>(s.p_hbond),   // eta is stored as 2 x the file value (as LAMMPS)
          s.r_pi_pi, s.p_lp2, s.vincr, s.b_o_131, s.b_o_132, s.b_o_133, s.bcut_acks2, s.alphag,
          s.p_ovun2, s.p_val3, s.betael, s.valency_val, s.p_val5, s.rcore2, s.ecore2, s.acore2};
      static const char* const nm[32] = {"r_s","valency","mass","r_vdw","epsilon","gamma","r_pi","valency_e","alpha","gamma_w","valency_boc","p_ovun5","pxel2","chi","eta/2","p_hbond",
          "r_pi_pi","p_lp2","vincr","b_o_131","b_o_132","b_o_133","bcut_acks2","alphag","p_ovun2","p_val3","betael","valency_val","p_val5","rcore2","ecore2","acore2"};
      for (int c = 0; c < 32; ++c) {
        // p_hbond is an integer in the file column (int(vnphb) in the legacy reader)
        const double want = c == 15 ? std::trunc(v[15]) : v[static_cast<std::size_t>(c)];
        if (!close(got[c], want)) { std::printf("atom %d (%s) %s: parsed %.6f legacy %.6f\n", atom + 1, name.c_str(), nm[c], got[c], want); ++bad; }
      }
      ++atom; ++atoms_checked;
    } else if (k == "BOND") {
      int a, b; ls >> a >> b; std::vector<double> v; double x; while (ls >> x) v.push_back(x);
      RM_CHECK(v.size() == 16);
      const TwoBody& t = ff.pair(a - 1, b - 1);
      const double got[16] = {t.De_s, t.De_p, t.De_pp, t.p_be1, t.p_bo5, t.v13cor, t.p_bo6, t.p_ovun1,
                              t.p_be2, t.p_bo3, t.p_bo4, v[11] /* legacy `bom`: not kept by Metal-Reax */, t.p_bo1, t.p_bo2, t.ovc, t.pxel1};
      static const char* const nm[16] = {"De_s","De_p","De_pp","p_be1","p_bo5","v13cor","p_bo6","p_ovun1","p_be2","p_bo3","p_bo4","bom","p_bo1","p_bo2","ovc","pxel1"};
      for (int c = 0; c < 16; ++c)
        if (!close(got[c], v[static_cast<std::size_t>(c)])) { std::printf("bond %d-%d %s: parsed %.6f legacy %.6f\n", a, b, nm[c], got[c], v[static_cast<std::size_t>(c)]); ++bad; }
      ++bonds_checked;
    }
  }
  std::printf("checked: %d general values, %d atoms (32 values each), %d bonds (16 values each); mismatches: %d\n", gen_checked, atoms_checked, bonds_checked, bad);
  RM_CHECK(atoms_checked == ff.num_types());
  RM_CHECK(bad == 0);
  return rmtest::finish("ffield_legacy_columns");
}
