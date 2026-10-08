// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Shared pure term functions (terms.hpp): the stable taper form equals the reference Horner form, float stays close, and its derivative is consistent.
#include <cmath>

#include "reaxmetal/terms_host.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

int main() {
  for (const double swa : {0.0, 1.5}) {
    const double swb = 10.0;
    const auto c = terms::taper_coeffs(swa, swb);
    double worst = 0, worstd = 0, worstf = 0;
    for (double r = swa + 0.01; r < swb; r += 0.0371) {
      double T1, d1, T2, d2;
      terms::taper_horner<double>(c.c, r, T1, d1);
      terms::taper_stable<double>(swa, swb, r, T2, d2);
      worst = std::fmax(worst, std::fabs(T1 - T2));
      worstd = std::fmax(worstd, std::fabs(d1 - d2) / std::fmax(1e-3, std::fabs(d1)));
      float Tf, df;
      terms::taper_stable<float>(static_cast<float>(swa), static_cast<float>(swb), static_cast<float>(r), Tf, df);
      worstf = std::fmax(worstf, std::fabs(static_cast<double>(Tf) - T2));
      // central difference of Tap equals r * dTap
      const double h = 1e-6;
      double Tp, Tm, dummy;
      terms::taper_stable<double>(swa, swb, r + h, Tp, dummy);
      terms::taper_stable<double>(swa, swb, r - h, Tm, dummy);
      RM_CHECK(std::fabs((Tp - Tm) / (2 * h) - r * d2) < 1e-6);
    }
    RM_CHECK_MSG(worst < 1e-12, "taper value stable vs Horner");
    RM_CHECK_MSG(worstd < 1e-9, "taper derivative stable vs Horner");
    RM_CHECK_MSG(worstf < 1e-5, "taper float vs double");
    std::printf("swa=%g: |Tap_horner - Tap_stable| %.2e, rel dTap %.2e, float %.2e\n", swa, worst, worstd, worstf);
    double t0, d0;
    terms::taper_stable<double>(swa, swb, swa, t0, d0);
    RM_CHECK(std::fabs(t0 - 1.0) < 1e-15);
  }
  return rmtest::failures() == 0 ? 0 : 1;
}
