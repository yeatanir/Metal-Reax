// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// eReaxFF explicit-electron pair term (plan task 3.2), CPU-64. Re-implemented from the formulas of the legacy routine `electron`
// (Margaret/eReaxFF_src/poten.f:3924-4160); no legacy code is copied.
//
//   core--El :  E = -C * betael(core) * Z(core) * erf( sqrt(2*alphag(core)) * r ) / r  * Tap(r)          (poten.f:3987-3992)
//   El--El   :  E = +C * betael(El_i) * erf( sqrt(alphag(El_i)) * r ) / r              * Tap(r)          (poten.f:4119-4127)
//   C = 332.0638 (kcal A / mol e^2), Z = znucel table (flags.core_charge), Tap = 7th-order taper from (gp[11], electron taper),
//   pairs with r > electron taper or r < 1e-10 skipped (poten.f:3972); every pair counted once, self-images excluded.
// The legacy El--El branch uses the parameters of the first atom of the pair only; here both El types must carry identical
// (alphag, betael) so the result does not depend on pair order (checked, SystemError otherwise).
// Hole particles ("Ho") are not supported (the legacy code has extra branches for them) and are rejected.
#include <cstddef>
#include <vector>

#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/neighbor.hpp"
#include "reaxmetal/system.hpp"

namespace reaxmetal {

struct ElectronResult {
  double e_electron = 0;          // total electron-pair energy, kcal/mol (the legacy `eelect`)
  double e_core_el = 0, e_el_el = 0;
  std::vector<double> grad;       // 3 * nall, dE/dx (physical force = -grad), ghosts included
  std::size_t pairs = 0;          // counted pairs
};

// The far list must reach the electron taper (taper <= cut.nonb), otherwise SystemError.
ElectronResult compute_electron_core(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far);

namespace electron_terms {
// Value and dE/dr of one pair including the taper. `dTap_dr` = dTap/dr (NOT divided by r).
struct PairOut { double e, dEdr; };
PairOut core_electron(double betael, double Z, double alphag, double r, double Tap, double dTap_dr) noexcept;
PairOut electron_electron(double betael, double alphag, double r, double Tap, double dTap_dr) noexcept;
extern const double kCoulombKcal;   // the legacy c1c
}  // namespace electron_terms

}  // namespace reaxmetal
