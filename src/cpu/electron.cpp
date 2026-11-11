// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/electron.hpp"

#include <cmath>

#include "reaxmetal/terms_host.hpp"

namespace reaxmetal {
namespace electron_terms {

// Compat (reproduce the legacy executable): poten.f:3935 `c1c=332.0638` is a default-REAL literal assigned to a double, i.e. the
// single-precision value 332.0638122558594. Checked against the exact reference: with the double 332.0638 the electron energy of the
// 225-atom Jamil-target case is low by 3.69e-8 relative; with this constant the match is at round-off (phase 3 notes).
const double kCoulombKcal = static_cast<double>(332.0638f);

PairOut core_electron(double betael, double Z, double alphag, double r, double Tap, double dTap_dr) noexcept {
  const double pi = 3.14159265358979323846;
  const double s = std::sqrt(2.0 * alphag * 1.0);
  const double erfv = std::erf(s * r);
  const double e0 = -kCoulombKcal * betael * Z * erfv / r;                                   // poten.f:3987-3988
  const double t1 = 2.0 * std::sqrt((2.0 * alphag) / pi) * r * std::exp(-2.0 * r * r * alphag);  // poten.f:3994-3995
  const double d0 = -kCoulombKcal * betael * Z * (t1 - erfv) / (r * r);                      // poten.f:3997
  return {e0 * Tap, d0 * Tap + e0 * dTap_dr};                                                // poten.f:4007-4008 (sw, sw1)
}

PairOut electron_electron(double betael, double alphag, double r, double Tap, double dTap_dr) noexcept {
  const double pi = 3.14159265358979323846;
  const double erfv = std::erf(std::sqrt(alphag) * r);
  const double e0 = kCoulombKcal * betael * erfv / r;                                         // poten.f:4121
  const double t1 = 2.0 * std::sqrt(alphag / pi) * r * std::exp(-r * r * alphag);            // poten.f:4123
  const double d0 = kCoulombKcal * betael * (t1 - erfv) / (r * r);                            // poten.f:4125
  return {e0 * Tap, d0 * Tap + e0 * dTap_dr};
}

}  // namespace electron_terms

ElectronResult compute_electron_core(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far) {
  if (!ff.ereaxff()) throw SystemError("compute_electron_core: the force field was not parsed in eReaxFF mode");
  const double swb = ff.electron_taper(), swa = ff.file_control().nonb_low;
  if (swb > cut.nonb) throw SystemError("compute_electron_core: electron taper radius exceeds the far-list cutoff nonb_cut");
  // El--El parameters must not depend on which atom of the pair is "first" (the legacy code uses the first one)
  const SingleBody* el_ref = nullptr;
  for (int t = 0; t < ff.num_types(); ++t) {
    const SingleBody& s = ff.single(t);
    if (s.name == "HO") continue;  // type present in the ffield; rejected below only if atoms use it
    if (!s.flags.is_electron) continue;
    if (el_ref && (s.alphag != el_ref->alphag || s.betael != el_ref->betael))
      throw SystemError("compute_electron_core: two electron types with different (alphag, betael) are not supported");
    el_ref = &s;
  }
  const std::size_t N = atoms.nall(), n = atoms.nlocal;
  for (std::size_t i = 0; i < N; ++i)
    if (atoms.type[i] >= 0 && ff.single(atoms.type[i]).name == "HO") throw SystemError("compute_electron_core: hole particles (Ho) are not supported");

  const terms::TaperCoeffs tap = terms::taper_coeffs(swa, swb);
  ElectronResult res;
  res.grad.assign(3 * N, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    const SingleBody& si = ff.single(ti);
    for (std::size_t pj = far.row_start[i]; pj < far.row_start[i + 1]; ++pj) {
      const std::size_t j = static_cast<std::size_t>(far.nbr[pj]);
      const int tj = atoms.type[j];
      if (tj < 0) continue;
      const SingleBody& sj = ff.single(tj);
      if (!si.flags.is_electron && !sj.flags.is_electron) continue;
      const double r = far.dist[pj];
      if (r > swb || r < 1.0e-10) continue;                                   // poten.f:3972
      const double* dv = &far.dvec[3 * pj];
      const PairClass pc = classify_nonbonded_entry(i, j, n, atoms.tag[i], atoms.tag[j], dv);
      if (pc == PairClass::None || pc == PairClass::SelfImage) continue;      // once per pair; legacy skips the self-image block
      double Tap, dTap;
      terms::taper_horner<double>(tap.c, r, Tap, dTap);                       // dTap = (1/r) dTap/dr
      const double dTap_dr = r * dTap;
      electron_terms::PairOut o;
      if (si.flags.is_electron && sj.flags.is_electron) {
        o = electron_terms::electron_electron(si.betael, si.alphag, r, Tap, dTap_dr);
        res.e_el_el += o.e;
      } else {
        const SingleBody& core = si.flags.is_electron ? sj : si;
        o = electron_terms::core_electron(core.betael, core.flags.core_charge, core.alphag, r, Tap, dTap_dr);
        res.e_core_el += o.e;
      }
      // gradient: r = |x_j - x_i|, dvec = x_j - x_i  ->  dE/dx_i = -dEdr * dvec / r, dE/dx_j = +dEdr * dvec / r
      for (std::size_t c = 0; c < 3; ++c) {
        res.grad[3 * i + c] -= o.dEdr * dv[c] / r;
        res.grad[3 * j + c] += o.dEdr * dv[c] / r;
      }
      ++res.pairs;
    }
  }
  res.e_electron = res.e_core_el + res.e_el_el;
  return res;
}

}  // namespace reaxmetal
