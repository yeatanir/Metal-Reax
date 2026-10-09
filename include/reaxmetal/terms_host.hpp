// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// Host-side wrapper of terms.hpp: maps the math macros onto the C++ library and adds the host-only pieces (double precision taper
// coefficients, Horner taper of the CPU-64 reference). reaxff_init_md.cpp:72-106.
#include <cmath>

#define RM_POW std::pow
#define RM_EXP std::exp
#define RM_LOG std::log
#include "reaxmetal/terms.hpp"

namespace reaxmetal {
namespace terms {

struct TaperCoeffs { double c[8]; };
inline TaperCoeffs taper_coeffs(double swa, double swb) {
  const double d1 = swb - swa, d7 = std::pow(d1, 7.0);
  const double swa2 = swa * swa, swa3 = swa * swa * swa, swb2 = swb * swb, swb3 = swb * swb * swb;
  TaperCoeffs t{};
  t.c[7] = 20.0 / d7;
  t.c[6] = -70.0 * (swa + swb) / d7;
  t.c[5] = 84.0 * (swa2 + 3.0 * swa * swb + swb2) / d7;
  t.c[4] = -35.0 * (swa3 + 9.0 * swa2 * swb + 9.0 * swa * swb2 + swb3) / d7;
  t.c[3] = 140.0 * (swa3 * swb + 3.0 * swa2 * swb2 + swa * swb3) / d7;
  t.c[2] = -210.0 * (swa3 * swb2 + swa2 * swb3) / d7;
  t.c[1] = 140.0 * swa3 * swb3 / d7;
  t.c[0] = (-35.0 * swa3 * swb2 * swb2 + 21.0 * swa2 * swb3 * swb2 - 7.0 * swa * swb3 * swb3 + swb3 * swb3 * swb) / d7;
  return t;
}
// Tap(r) and dTap(r) = (1/r) dTap/dr, by Horner in the order of the reference (CPU-64 reference form).
template <class T>
inline void taper_horner(const T* c, T r, T& Tap, T& dTap) {
  Tap = c[7] * r + c[6];
  Tap = Tap * r + c[5]; Tap = Tap * r + c[4]; Tap = Tap * r + c[3]; Tap = Tap * r + c[2]; Tap = Tap * r + c[1]; Tap = Tap * r + c[0];
  dTap = 7 * c[7] * r + 6 * c[6];
  dTap = dTap * r + 5 * c[5]; dTap = dTap * r + 4 * c[4]; dTap = dTap * r + 3 * c[3]; dTap = dTap * r + 2 * c[2];
  dTap += c[1] / r;
}

}  // namespace terms
}  // namespace reaxmetal
