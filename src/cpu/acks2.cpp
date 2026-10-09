// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/acks2.hpp"

#include <cmath>

#include "reaxmetal/terms_host.hpp"

namespace reaxmetal {

void add_acks2_terms(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far, const std::vector<double>& q,
                     const double* s_kin, NonbondedResult& res) {
  (void)cut;
  const std::size_t N = atoms.nall(), n = atoms.nlocal;
  if (res.grad.size() != 3 * N) throw SystemError("add_acks2_terms: result gradient has the wrong size");
  const double softness = ff.global().l.size() > 34 ? ff.global().l[34] : 0.0;
  const bool tally = !res.eatom.empty();
  double e_ele = 0.0, e_pol = 0.0;
  // polarization coupling: KCALpMOL_to_EV * q * s[N + i]
  for (std::size_t i = 0; i < n; ++i) {
    if (atoms.type[i] < 0) continue;
    const double e = RM_KCAL_TO_EV * q[i] * s_kin[i];
    e_pol += e;
    if (tally) res.eatom[i] += e;
  }
  // bond-softness term over the far list (owner-computes counting as the non-bonded pairs)
  for (std::size_t i = 0; i < n; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    for (std::size_t pj = far.row_start[i]; pj < far.row_start[i + 1]; ++pj) {
      const std::size_t j = static_cast<std::size_t>(far.nbr[pj]);
      const int tj = atoms.type[j];
      if (tj < 0) continue;
      const double xcut = 0.5 * (ff.single(ti).bcut_acks2 + ff.single(tj).bcut_acks2);
      const double dist = far.dist[pj];
      if (!(dist <= xcut)) continue;
      const double* dv = &far.dvec[3 * pj];
      if (classify_nonbonded_entry(i, j, n, atoms.tag[i], atoms.tag[j], dv) == PairClass::None) continue;
      const double d = dist / xcut;
      const double bs = softness * std::pow(d, 3.0) * std::pow(1.0 - d, 6.0);
      if (!(bs > 0.0)) continue;
      const double effpot = s_kin[i] - s_kin[j];
      const double e = -0.5 * RM_KCAL_TO_EV * bs * effpot * effpot;
      e_ele += e;
      double dbs = softness * 3.0 / xcut * std::pow(d, 2.0) * std::pow(1.0 - d, 5.0) * (1.0 - 3.0 * d);
      dbs = -0.5 * dbs * effpot * effpot;
      dbs = RM_KCAL_TO_EV * dbs / dist;
      for (std::size_t c = 0; c < 3; ++c) {
        res.grad[3 * i + c] += -dbs * dv[c];
        res.grad[3 * j + c] += +dbs * dv[c];
      }
      if (tally) {
        res.eatom[i] += 0.5 * e;
        res.eatom[j] += 0.5 * e;
        const double fp = -dbs;   // ev_tally(i, j, ..., 0, e_ele, fpair = -d_bond_softness, del = xi - xj)
        const double v[6] = {dv[0] * dv[0] * fp, dv[1] * dv[1] * fp, dv[2] * dv[2] * fp, dv[0] * dv[1] * fp, dv[0] * dv[2] * fp, dv[1] * dv[2] * fp};
        for (std::size_t c = 0; c < 6; ++c) { res.vatom[i][c] += 0.5 * v[c]; res.vatom[j][c] += 0.5 * v[c]; }
      }
    }
  }
  res.e[EnergyTerm::Coulomb] += e_ele;
  res.e[EnergyTerm::Polarization] += e_pol;
}

}  // namespace reaxmetal
