// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/nonbonded.hpp"

#include "reaxmetal/terms_host.hpp"

#ifndef RM_REAL
#define RM_REAL double
#endif
#ifndef RM_NONBONDED_NAME
#define RM_NONBONDED_NAME compute_nonbonded_core
#endif
namespace reaxmetal {

using Real = RM_REAL;
inline Real R(double v) { return static_cast<Real>(v); }

NonbondedResult RM_NONBONDED_NAME(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far,
                                       const std::vector<double>& q, const NonbondedOptions& opt) {
  const std::size_t N = atoms.nall(), n = atoms.nlocal;
  if (q.size() != n && q.size() != N) throw SystemError("compute_nonbonded_core: need one charge per owned atom (ghosts use their owner's) or per atom");
  const auto& gp = ff.global().l;
  const Real p_vdW1 = gp[28];
  const int vdw_type = ff.global().vdw_type;
  [[maybe_unused]] const terms::TaperCoeffs tap = terms::taper_coeffs(ff.file_control().nonb_low, ff.file_control().nonb_cut);

  NonbondedResult res;
  res.grad.assign(3 * N, Real(0.0));
  if (opt.per_atom) { res.eatom.assign(N, Real(0.0)); res.vatom.assign(N, std::array<double, 6>{}); }
  double e_vdw = 0, e_ele = 0, e_pol = 0;   // term values are Real; the sums are double (the GPU path sums its row partials in double on the host)
  for (std::size_t i = 0; i < n; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    for (std::size_t pj = far.row_start[i]; pj < far.row_start[i + 1]; ++pj) {
      const std::size_t j = static_cast<std::size_t>(far.nbr[pj]);
      const int tj = atoms.type[j];
      if (tj < 0) continue;
      const Real r = far.dist[pj];
      if (!(r <= cut.nonb)) continue;
      const double* dv = &far.dvec[3 * pj];
      if (classify_nonbonded_entry(i, j, n, atoms.tag[i], atoms.tag[j], dv) == PairClass::None) continue;
      const TwoBody& t = ff.pair(ti, tj);
      const terms::NbPair<Real> np{R(t.alpha), R(t.D), R(t.r_vdW), R(t.gamma_w), R(t.gamma), R(t.ecore), R(t.acore), R(t.rcore), R(t.lgcij), R(t.lgre)};
      Real Tap, dTap;
#ifdef RM_REAL_FLOAT
      terms::taper_stable<Real>(R(ff.file_control().nonb_low), R(ff.file_control().nonb_cut), r, Tap, dTap);   // the form the GPU kernels use
#else
      terms::taper_horner<Real>(tap.c, r, Tap, dTap);
#endif
      const Real qi = q[i], qj = (q.size() == N) ? q[j] : q[static_cast<std::size_t>(atoms.owner[j])];
      const auto o = terms::nonbonded_pair<Real>(np, vdw_type, opt.lgvdw, p_vdW1, qi, qj, r, Tap, dTap);
      e_vdw += o.e_vdW;
      e_ele += o.e_ele;
      if (opt.per_atom) {   // ev_tally(i, j, natoms, 1, pe_vdw, e_ele, fpair = -CE, del = xi - xj): half to each atom
        const Real eh = Real(0.5) * (o.e_vdW + o.e_ele), fp = -o.CE;
        res.eatom[i] += eh; res.eatom[j] += eh;
        const Real v[6] = {R(dv[0] * dv[0]) * fp, R(dv[1] * dv[1]) * fp, R(dv[2] * dv[2]) * fp, R(dv[0] * dv[1]) * fp, R(dv[0] * dv[2]) * fp, R(dv[1] * dv[2]) * fp};
        for (std::size_t c = 0; c < 6; ++c) { res.vatom[i][c] += Real(0.5) * v[c]; res.vatom[j][c] += Real(0.5) * v[c]; }
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
    const Real ep = terms::polarization<Real>(R(s.chi), R(s.eta), R(q[i]));
    e_pol += ep;
    if (opt.per_atom) res.eatom[i] += ep;   // ev_tally(i, i, n, 1, 0, en_tmp)
  }
  res.e[EnergyTerm::VdW] = e_vdw;
  res.e[EnergyTerm::Coulomb] = e_ele;
  res.e[EnergyTerm::Polarization] = e_pol;
  return res;
}

}  // namespace reaxmetal
