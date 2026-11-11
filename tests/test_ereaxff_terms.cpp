// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Pure eReaxFF atom-energy functions (ereaxff_terms.hpp): every analytic partial derivative against central finite differences of the energy.
#include <cmath>
#include <cstdio>
#include <random>

#include "reaxmetal/ereaxff_terms.hpp"
#include "test_util.hpp"

using namespace reaxmetal::ereaxff_terms;

int main() {
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double worst_ou = 0, worst_lp = 0;
  const double h = 1e-6;
  for (int t = 0; t < 400; ++t) {
    // ---- over / under: random but smooth region
    OuParams p{0.5 + 3 * u(rng), 4.0 + 20 * u(rng), -0.5 - 1.5 * u(rng), 0.2 + 2 * u(rng), 4.0 + 10 * u(rng), -1.0 - 3 * u(rng), (t % 5 == 0) ? -1.0 : 10.0 + 30 * u(rng)};
    const double sum1 = 3 * u(rng), sum2 = -1 + 3 * u(rng), X = -1 + 3 * u(rng), T = -0.5 + u(rng), A = 2 + 2 * u(rng);
    const auto o = over_under(p, sum1, sum2, X, T, A, true);
    auto E = [&](double s1, double s2, double x, double tt, double a) { const auto q = over_under(p, s1, s2, x, tt, a, true); return q.e_ov + q.e_un; };
    auto fd = [&](double s1, double s2, double x, double tt, double a, int which) {
      double v[5] = {s1, s2, x, tt, a}, w[5] = {s1, s2, x, tt, a};
      v[which] += h; w[which] -= h;
      return (E(v[0], v[1], v[2], v[3], v[4]) - E(w[0], w[1], w[2], w[3], w[4])) / (2 * h);
    };
    worst_ou = std::fmax(worst_ou, std::fabs(fd(sum1, sum2, X, T, A, 0) - o.dEdu1));
    worst_ou = std::fmax(worst_ou, std::fabs(fd(sum1, sum2, X, T, A, 2) - o.dEdX));
    worst_ou = std::fmax(worst_ou, std::fabs(fd(sum1, sum2, X, T, A, 3) - o.dEdT));
    worst_ou = std::fmax(worst_ou, std::fabs(fd(sum1, sum2, X, T, A, 1) - o.dEdu2));
    // dEdA is the explicit derivative at fixed g: only the over-coordination energy has it
    auto Ea = [&](double a) { return over_under(p, sum1, sum2, X, T, a, true).e_ov; };
    worst_ou = std::fmax(worst_ou, std::fabs((Ea(A + h) - Ea(A - h)) / (2 * h) - o.dEdA));
    // under skipped for valp1 < 0
    if (p.valp1 < 0) RM_CHECK(o.e_un == 0.0);

    // ---- corrected atom quantities and lone pair: derivatives wrt abo, A, B
    const double abo = 0.3 + 4 * u(rng), Ac = 3 + 2 * u(rng), Bc = 3 + 2 * u(rng), p_lp1 = 6.0 + 2 * u(rng), p_lp2 = 5 + 20 * u(rng);
    const auto c = corrected_atom(abo, Ac, Bc, p_lp1, false);
    const auto lp = lone_pair(p_lp2, c.Delta_lp);
    auto Elp = [&](double a, double A_, double B_) { const auto cc = corrected_atom(a, A_, B_, p_lp1, false); return lone_pair(p_lp2, cc.Delta_lp).e; };
    const double fa = (Elp(abo + h, Ac, Bc) - Elp(abo - h, Ac, Bc)) / (2 * h);
    const double fA = (Elp(abo, Ac + h, Bc) - Elp(abo, Ac - h, Bc)) / (2 * h);
    const double fB = (Elp(abo, Ac, Bc + h) - Elp(abo, Ac, Bc - h)) / (2 * h);
    // skip points within the int() truncation kinks of Delta_e / 2
    const double De = abo - Bc, frac = De / 2.0 - std::floor(De / 2.0);
    if (frac > 1e-4 && frac < 1 - 1e-4 && std::fabs(De / 2.0 - std::round(De / 2.0)) > 1e-4) {
      worst_lp = std::fmax(worst_lp, std::fabs(fa - lp.dEdDlp * c.Clp));
      worst_lp = std::fmax(worst_lp, std::fabs(fA - lp.dEdDlp * (-0.5)));
      worst_lp = std::fmax(worst_lp, std::fabs(fB - lp.dEdDlp * (0.5 - c.Clp)));
    }
  }
  std::printf("over/under: max |FD - analytic| = %.3e;  lone pair: %.3e\n", worst_ou, worst_lp);
  RM_CHECK_MSG(worst_ou < 1e-6, "over/under partials");
  RM_CHECK_MSG(worst_lp < 1e-6, "lone-pair partials");
  // heavy atoms: Dlt is zero
  RM_CHECK(corrected_atom(2.0, 4.0, 4.0, 6.5, true).Dlt == 0.0);
  RM_CHECK(corrected_atom(2.0, 4.0, 4.0, 6.5, false).Dlt == corrected_atom(2.0, 4.0, 4.0, 6.5, false).Delta_lp);
  return rmtest::finish("ereaxff_terms");
}
