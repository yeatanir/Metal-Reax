// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/nonbonded.hpp"

#include "reaxmetal/terms_host.hpp"

namespace reaxmetal {

NonbondedResult compute_nonbonded_core(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far,
                                       const std::vector<double>& q, const NonbondedOptions& opt) {
  const std::size_t N = atoms.nall(), n = atoms.nlocal;
  if (q.size() != n) throw SystemError("compute_nonbonded_core: need one charge per owned atom");
  const auto& gp = ff.global().l;
  const double p_vdW1 = gp[28];
  const int vdw_type = ff.global().vdw_type;
  const terms::TaperCoeffs tap = terms::taper_coeffs(ff.file_control().nonb_low, ff.file_control().nonb_cut);

  NonbondedResult res;
  res.grad.assign(3 * N, 0.0);
  if (opt.per_atom) { res.eatom.assign(N, 0.0); res.vatom.assign(N, std::array<double, 6>{}); }
  double e_vdw = 0, e_ele = 0, e_pol = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    for (std::size_t pj = far.row_start[i]; pj < far.row_start[i + 1]; ++pj) {
      const std::size_t j = static_cast<std::size_t>(far.nbr[pj]);
      const int tj = atoms.type[j];
      if (tj < 0) continue;
      const double r = far.dist[pj];
      if (!(r <= cut.nonb)) continue;
      const double* dv = &far.dvec[3 * pj];
      if (classify_nonbonded_entry(i, j, n, atoms.tag[i], atoms.tag[j], dv) == PairClass::None) continue;
      const TwoBody& t = ff.pair(ti, tj);
      const terms::NbPair<double> np{t.alpha, t.D, t.r_vdW, t.gamma_w, t.gamma, t.ecore, t.acore, t.rcore, t.lgcij, t.lgre};
      double Tap, dTap;
      terms::taper_horner<double>(tap.c, r, Tap, dTap);
      const double qi = q[i], qj = q[static_cast<std::size_t>(atoms.owner[j])];
      const auto o = terms::nonbonded_pair<double>(np, vdw_type, opt.lgvdw, p_vdW1, qi, qj, r, Tap, dTap);
      e_vdw += o.e_vdW;
      e_ele += o.e_ele;
      if (opt.per_atom) {   // ev_tally(i, j, natoms, 1, pe_vdw, e_ele, fpair = -CE, del = xi - xj): half to each atom
        const double eh = 0.5 * (o.e_vdW + o.e_ele), fp = -o.CE;
        res.eatom[i] += eh; res.eatom[j] += eh;
        const double v[6] = {dv[0] * dv[0] * fp, dv[1] * dv[1] * fp, dv[2] * dv[2] * fp, dv[0] * dv[1] * fp, dv[0] * dv[2] * fp, dv[1] * dv[2] * fp};
        for (std::size_t c = 0; c < 6; ++c) { res.vatom[i][c] += 0.5 * v[c]; res.vatom[j][c] += 0.5 * v[c]; }
      }
      // reference: f[i] += -(CE) * dvec, f[j] += +(CE) * dvec  (workspace f is the gradient)
      for (std::size_t c = 0; c < 3; ++c) {
        res.grad[3 * i + c] += -o.CE * dv[c];
        res.grad[3 * j + c] += +o.CE * dv[c];
      }
      ++res.pairs;
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    const SingleBody& s = ff.single(ti);
    const double ep = terms::polarization<double>(s.chi, s.eta, q[i]);
    e_pol += ep;
    if (opt.per_atom) res.eatom[i] += ep;   // ev_tally(i, i, n, 1, 0, en_tmp)
  }
  res.e[EnergyTerm::VdW] = e_vdw;
  res.e[EnergyTerm::Coulomb] = e_ele;
  res.e[EnergyTerm::Polarization] = e_pol;
  return res;
}

}  // namespace reaxmetal
