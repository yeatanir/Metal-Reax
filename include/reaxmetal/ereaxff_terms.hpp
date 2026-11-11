// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// eReaxFF atom-energy pieces (plan tasks 3.3 / 3.4), double precision, host only. Re-implemented from the formulas of the legacy routines
// lonpar (Margaret/eReaxFF_src/poten.f:397-619) and ovcor (poten.f:862-1700); only the ENERGY formulas are taken from there, every
// derivative below is derived here and checked by finite differences (tests/test_ereaxff_terms.cpp, test_ereaxff_bonded.cpp).
//
// Per atom i the legacy code works with a "corrected" valency pair (elecpos, reac.f:8779-8851):
//     A = avalc = aval + share_sign * e2,   B = stlpc = stlp + e2,   e2 = eshare2 = sum_El exp(-vincr * r^2)
// and abo = total corrected bond order.
//   lone pair      Delta_e = abo - B, nlp = exp(-p_lp1 (2 + vlpex)^2) - int(Delta_e/2), nlp_opt = (B - A)/2, Delta_lp = nlp_opt - nlp
//                  E_lp = p_lp2 * Delta_lp / (1 + exp(-75 Delta_lp))                         (poten.f:442-457)
//   over / under   X = f * (abo - A),  f = exp(-pxel2 * e2 * S),  S = sum_bonds pxel1 * BO / abo           (poten.f:930-946)
//                  neighbour sum  sum2 = sum_j (BO_pi + BO_pi2)_ij (X_j - dfvl_i * Dlt_j),   Dlt = Delta_lp (non-heavy) or 0 (heavy)
//                  g = X_i - dfvl_i Dlt_i / (1 + p3 exp(p4 sum2))   (the stock "Delta_lpcorr" with Delta -> X, valency -> A)
//                  E_ov = sum1 * g / (g + A + 1e-8) / (1 + exp(p2 g)),   E_un = valp1 (1/(1+exp(-p2 g))) (exp(p6 g) - 1) / (1 + p7 exp(p8 sum2))
//                  with E_un skipped when valp1 < 0 (poten.f:1503)
#include <cmath>

namespace reaxmetal {
namespace ereaxff_terms {

// ---- corrected atom quantities -----------------------------------------------------------------------------------------------------
struct CorrectedAtom {
  double A = 0, B = 0;      // avalc, stlpc
  double Delta = 0;         // abo - A
  double nlp = 0, Delta_lp = 0;
  double Clp = 0;           // -d nlp / d Delta_e = 2 p_lp1 exp(...) (2 + vlpex)
  double Dlt = 0;           // Delta_lp_temp: Delta_lp for ordinary atoms, 0 for heavy (legacy amas > 31) atoms
  bool heavy = false;
};
// d Delta_lp / d abo = Clp, d Delta_lp / d B = 1/2 - Clp, d Delta_lp / d A = -1/2   (same for Dlt unless heavy, where all are 0)
inline CorrectedAtom corrected_atom(double abo, double A, double B, double p_lp1, bool heavy) {
  CorrectedAtom c;
  c.A = A; c.B = B; c.heavy = heavy;
  c.Delta = abo - A;
  const double De = abo - B;
  const double half = static_cast<double>(static_cast<int>(De / 2.0));       // Fortran int(): truncation toward zero (poten.f:431)
  const double vlpex = De - 2.0 * half;
  const double expv = std::exp(-p_lp1 * (2.0 + vlpex) * (2.0 + vlpex));
  c.nlp = expv - half;
  c.Delta_lp = 0.5 * (B - A) - c.nlp;
  c.Clp = 2.0 * p_lp1 * expv * (2.0 + vlpex);
  c.Dlt = heavy ? 0.0 : c.Delta_lp;
  return c;
}

struct LonePairE { double e, dEdDlp; };
inline LonePairE lone_pair(double p_lp2, double Delta_lp) {
  const double ex = std::exp(-75.0 * Delta_lp), inv = 1.0 / (1.0 + ex);
  return {p_lp2 * Delta_lp * inv, p_lp2 * inv + 75.0 * p_lp2 * Delta_lp * ex * inv * inv};
}

// ---- over / under coordination of one atom ---------------------------------------------------------------------------------------------
struct OuParams { double p2, p3, p4, p6, p7, p8, valp1; };   // p2 = vovun (atom), p3 = gp[32], p4 = gp[31], p6 = gp[6], p7 = gp[8], p8 = gp[9], valp1 (atom)
struct OuOut {
  double e_ov, e_un;
  double dEdu1;    // d(e_ov + e_un)/d sum1  (only e_ov depends on sum1)
  double dEdX;     // total derivative wrt X (own effective Delta)
  double dEdT;     // wrt T = dfvl * Dlt_i (own)
  double dEdu2;    // total derivative wrt sum2 (through g and explicitly in e_un)
  double dEdA;     // explicit derivative wrt A (through 1/(g + A + 1e-8) of the over-coordination energy)
};
inline OuOut over_under(const OuParams& p, double sum1, double sum2, double X, double T, double A, bool under_active) {
  const double exp1 = p.p3 * std::exp(p.p4 * sum2);
  const double inv1 = 1.0 / (1.0 + exp1);
  const double g = X - T * inv1;
  const double exp2 = std::exp(p.p2 * g), inv2 = 1.0 / (1.0 + exp2);
  const double Dv = 1.0 / (g + A + 1e-8);
  OuOut o{};
  o.e_ov = sum1 * g * Dv * inv2;
  o.dEdu1 = g * Dv * inv2;
  double dEdg = sum1 * Dv * inv2 * (1.0 - g * (Dv + p.p2 * exp2 * inv2));
  o.dEdA = -sum1 * g * inv2 * Dv * Dv;
  double dEdu2_expl = 0.0;
  if (under_active && p.valp1 >= 0.0) {
    const double exp6 = std::exp(p.p6 * g), exp8 = p.p7 * std::exp(p.p8 * sum2);
    const double hu = 1.0 / (1.0 + std::exp(-p.p2 * g));          // hulpu
    const double hu2 = 1.0 / (1.0 + exp8);                         // hulpu2
    const double hu3 = exp6 - 1.0;                                 // hulpu3
    o.e_un = p.valp1 * hu * hu2 * hu3;
    dEdg += p.valp1 * hu2 * (p.p2 * hu * (1.0 - hu) * hu3 + hu * p.p6 * exp6);
    dEdu2_expl = -o.e_un * p.p8 * exp8 * hu2;
  }
  o.dEdX = dEdg;
  o.dEdT = -inv1 * dEdg;
  o.dEdu2 = dEdg * T * p.p4 * exp1 * inv1 * inv1 + dEdu2_expl;
  return o;
}

// ---- effective Delta of one atom: X = f (abo - A), f = exp(-pxel2 e2 S), S = sum_b pxel1_b BO_b / abo -----------------------------------
struct Effective { double S, f, X; };
inline Effective effective_delta(double abo, double A, double pxel2, double e2, double sum_pxel1_bo) {
  Effective e;
  e.S = abo > 0.0 ? sum_pxel1_bo / abo : 0.0;
  e.f = std::exp(-pxel2 * e2 * e.S);
  e.X = e.f * (abo - A);
  return e;
}

}  // namespace ereaxff_terms
}  // namespace reaxmetal
