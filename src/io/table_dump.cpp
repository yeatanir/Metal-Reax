// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Canonical text dump of the parameter tables. tools/reaxref/canonical_tables.py writes the SAME text from the tables stored
// by the instrumented pinned LAMMPS (params.txt), so `equal text <=> equal tables` is the PARSE-1 test.
#include <cstdio>
#include <string>

#include "reaxmetal/forcefield.hpp"

namespace reaxmetal {
namespace {
void put(std::string& o, double v) {
  char b[40];
  std::snprintf(b, sizeof b, " %.17g", v);
  o += b;
}
void put(std::string& o, int v) { o += " " + std::to_string(v); }
}  // namespace

std::string canonical_table_dump(const ForceField& ff, bool portable) {
  std::string o = "FFIELD-TABLES 1\n";
  const auto& g = ff.global();
  o += "GP" + std::string(" ") + std::to_string(g.l.size()) + " " + std::to_string(g.vdw_type) + "\n";
  for (std::size_t i = 0; i < g.l.size(); ++i) {
    o += "G " + std::to_string(i);
    put(o, g.l[i]);
    o += "\n";
  }
  const int n = ff.num_types();
  for (int i = 0; i < n; ++i) {
    const SingleBody& s = ff.single(i);
    o += "S " + std::to_string(i) + " " + s.name;
    for (double v : {s.r_s, s.valency, s.mass, s.r_vdw, s.epsilon, s.gamma, s.r_pi, s.valency_e, s.nlp_opt, s.alpha, s.gamma_w, s.valency_boc, s.p_ovun5}) put(o, v);
    put(o, s.p_hbond);
    for (double v : {s.chi, s.eta, s.r_pi_pi, s.p_lp2, s.b_o_131, s.b_o_132, s.b_o_133, s.bcut_acks2, s.p_ovun2, s.p_val3, s.valency_val, s.p_val5,
                     s.rcore2, s.ecore2, s.acore2, s.lgcij, s.lgre})
      put(o, v);
    o += "\n";
  }
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j) {
      const TwoBody& t = ff.pair(i, j);
      o += "T " + std::to_string(i) + " " + std::to_string(j);
      for (double v : {t.p_bo1, t.p_bo2, t.p_bo3, t.p_bo4, t.p_bo5, t.p_bo6, t.r_s, t.r_p, t.r_pp, t.p_boc3, t.p_boc4, t.p_boc5, t.p_be1, t.p_be2, t.De_s,
                       t.De_p, t.De_pp, t.p_ovun1, t.D, t.alpha, t.r_vdW, t.gamma_w, t.rcore, t.ecore, t.acore, t.lgcij, t.lgre})
        put(o, v);
      if (!portable) put(o, t.gamma);
      put(o, t.v13cor);
      put(o, t.ovc);
      o += "\n";
    }
  // sparse tables: iterate in key order through the public accessors
  for (int j = 0; j < n; ++j)
    for (int k = 0; k < n; ++k)
      for (int l = 0; l < n; ++l) {
        auto sets = ff.three_body(j, k, l);
        if (sets.empty()) continue;
        o += "H3 " + std::to_string(j) + " " + std::to_string(k) + " " + std::to_string(l) + " " + std::to_string(sets.size());
        for (const auto& s : sets)
          for (double v : {s.theta_00, s.p_val1, s.p_val2, s.p_val4, s.p_val7, s.p_pen1, s.p_coa1}) put(o, v);
        o += "\n";
      }
  for (int j = 0; j < n; ++j)
    for (int k = 0; k < n; ++k)
      for (int l = 0; l < n; ++l)
        for (int m = 0; m < n; ++m)
          if (const FourBody* f = ff.four_body(j, k, l, m)) {
            o += "H4 " + std::to_string(j) + " " + std::to_string(k) + " " + std::to_string(l) + " " + std::to_string(m);
            for (double v : {f->V1, f->V2, f->V3, f->p_tor1, f->p_cot1}) put(o, v);
            o += "\n";
          }
  for (int j = 0; j < n; ++j)
    for (int k = 0; k < n; ++k)
      for (int l = 0; l < n; ++l)
        if (const HBondParams* h = ff.hbond(j, k, l)) {
          if (*h == HBondParams{}) continue;  // an explicit entry equal to the default is indistinguishable from "absent" in the reference
          o += "HB " + std::to_string(j) + " " + std::to_string(k) + " " + std::to_string(l);
          for (double v : {h->r0_hb, h->p_hb1, h->p_hb2, h->p_hb3}) put(o, v);
          o += "\n";
        }
  return o;
}

}  // namespace reaxmetal
