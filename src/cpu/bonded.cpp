// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/bonded.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "reaxmetal/terms.hpp"

namespace reaxmetal {
namespace {

using R3 = std::array<double, 3>;

struct Bond {
  int nbr = 0, sym = 0;
  double d = 0;
  R3 dvec{}, dBOp{}, dln_BOp_s{}, dln_BOp_pi{}, dln_BOp_pi2{};
  double BO = 0, BO_s = 0, BO_pi = 0, BO_pi2 = 0;
  double C1dbo = 0, C2dbo = 0, C3dbo = 0, C1dbopi = 0, C2dbopi = 0, C3dbopi = 0, C4dbopi = 0, C1dbopi2 = 0, C2dbopi2 = 0, C3dbopi2 = 0, C4dbopi2 = 0;
  double Cdbo = 0, Cdbopi = 0, Cdbopi2 = 0;
};

terms::PairParams<double> pair_params(const TwoBody& t) {
  return {t.p_bo1, t.p_bo2, t.p_bo3, t.p_bo4, t.p_bo5, t.p_bo6, t.r_s, t.r_p, t.r_pp, t.p_boc3, t.p_boc4, t.p_boc5, t.ovc, t.v13cor,
          t.p_be1, t.p_be2, t.De_s, t.De_p, t.De_pp, t.p_ovun1};
}

inline void add(R3& a, const R3& b) { for (int c = 0; c < 3; ++c) a[static_cast<std::size_t>(c)] += b[static_cast<std::size_t>(c)]; }
inline void scaled_add(R3& a, double s, const R3& b) { for (int c = 0; c < 3; ++c) a[static_cast<std::size_t>(c)] += s * b[static_cast<std::size_t>(c)]; }
inline R3 scaled(double s, const R3& b) { return {s * b[0], s * b[1], s * b[2]}; }

}  // namespace

BondedResult compute_bonded_core(const ForceField& ff, const ControlParams& ctl, const AtomSet& atoms, const FarList& far,
                                 const BondedOptions& opt) {
  const std::size_t N = atoms.nall(), n = atoms.nlocal;
  const auto& gp = ff.global().l;
  const double bo_cut = ff.file_control().bo_cut;
  const double p_boc1 = gp[0], p_boc2 = gp[1], p_lp1 = gp[15], p_lp3 = gp[5];
  const double gp3 = gp[3], gp4 = gp[4], gp7 = gp[7], gp10 = gp[10];
  const double p_ovun3 = gp[32], p_ovun4 = gp[31], p_ovun6 = gp[6], p_ovun7 = gp[8], p_ovun8 = gp[9];
  const bool c2_gate = p_lp3 > 0.001;

  std::vector<std::vector<Bond>> bonds(N);
  std::vector<R3> dDeltap_self(N, R3{0, 0, 0});
  std::vector<double> total_bo(N, 0.0), Deltap(N, 0.0), Deltap_boc(N, 0.0);
  std::vector<int> bond_mark(N, 0);
  for (std::size_t i = n; i < N; ++i) bond_mark[i] = 1000;  // ghosts start "infinitely" far from any owned atom

  // ---- bond list (Init_Forces_noQEq + BOp) ----------------------------------------------------------------------------------
  for (std::size_t i = 0; i < N; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    const SingleBody& si = ff.single(ti);
    for (std::size_t pj = far.row_start[i]; pj < far.row_start[i + 1]; ++pj) {
      const std::size_t j = static_cast<std::size_t>(far.nbr[pj]);
      const double d = far.dist[pj];
      if (!(d <= ctl.bond_cut)) continue;
      const int tj = atoms.type[j];
      if (tj < 0) continue;
      const SingleBody& sj = ff.single(tj);
      const TwoBody& tb = ff.pair(ti, tj);
      const auto pp = pair_params(tb);
      const auto bp = terms::bo_prime<double>(pp, d, si.r_s > 0.0 && sj.r_s > 0.0, si.r_pi > 0.0 && sj.r_pi > 0.0,
                                              si.r_pi_pi > 0.0 && sj.r_pi_pi > 0.0, bo_cut);
      if (!(bp.total() >= bo_cut)) continue;

      Bond bi, bj;
      bi.nbr = static_cast<int>(j); bj.nbr = static_cast<int>(i);
      bi.sym = static_cast<int>(bonds[j].size()); bj.sym = static_cast<int>(bonds[i].size());
      bi.d = bj.d = d;
      const double* dv = &far.dvec[3 * pj];
      bi.dvec = {dv[0], dv[1], dv[2]};
      bj.dvec = scaled(-1.0, bi.dvec);
      bi.BO = bj.BO = bp.total();
      bi.BO_s = bj.BO_s = bp.BO_s; bi.BO_pi = bj.BO_pi = bp.BO_pi; bi.BO_pi2 = bj.BO_pi2 = bp.BO_pi2;
      const double rr2 = 1.0 / (d * d);
      const double Cln_s = tb.p_bo2 * bp.C12 * rr2, Cln_pi = tb.p_bo4 * bp.C34 * rr2, Cln_pi2 = tb.p_bo6 * bp.C56 * rr2;
      bi.dln_BOp_s = scaled(-bi.BO_s * Cln_s, bi.dvec);
      bi.dln_BOp_pi = scaled(-bi.BO_pi * Cln_pi, bi.dvec);
      bi.dln_BOp_pi2 = scaled(-bi.BO_pi2 * Cln_pi2, bi.dvec);
      bj.dln_BOp_s = scaled(-1.0, bi.dln_BOp_s);
      bj.dln_BOp_pi = scaled(-1.0, bi.dln_BOp_pi);
      bj.dln_BOp_pi2 = scaled(-1.0, bi.dln_BOp_pi2);
      bi.dBOp = scaled(-(bi.BO_s * Cln_s + bi.BO_pi * Cln_pi + bi.BO_pi2 * Cln_pi2), bi.dvec);
      bj.dBOp = scaled(-1.0, bi.dBOp);
      add(dDeltap_self[i], bi.dBOp);
      add(dDeltap_self[j], bj.dBOp);
      bi.BO_s -= bo_cut; bi.BO -= bo_cut; bj.BO_s -= bo_cut; bj.BO -= bo_cut;
      total_bo[i] += bi.BO;  // currently total_BOp
      total_bo[j] += bj.BO;
      bonds[i].push_back(bi);
      bonds[j].push_back(bj);

      if (bond_mark[j] > bond_mark[i] + 1) bond_mark[j] = bond_mark[i] + 1;
      else if (bond_mark[i] > bond_mark[j] + 1) bond_mark[i] = bond_mark[j] + 1;
    }
  }

  // ---- BO: corrected bond orders and atom quantities -------------------------------------------------------------------------
  for (std::size_t i = 0; i < N; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    const SingleBody& si = ff.single(ti);
    Deltap[i] = total_bo[i] - si.valency;
    Deltap_boc[i] = total_bo[i] - si.valency_val;
    total_bo[i] = 0;
  }
  for (std::size_t i = 0; i < N; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    const SingleBody& si = ff.single(ti);
    for (Bond& b : bonds[i]) {
      const std::size_t j = static_cast<std::size_t>(b.nbr);
      const int tj = atoms.type[j];
      if (tj < 0) continue;
      if (i < j || bond_mark[j] > 3) {
        const auto pp = pair_params(ff.pair(ti, tj));
        const auto c = terms::bo_correct<double>(pp, p_boc1, p_boc2, si.valency, ff.single(tj).valency, Deltap[i], Deltap[j], Deltap_boc[i],
                                                 Deltap_boc[j], b.BO, b.BO_s, b.BO_pi, b.BO_pi2);
        b.BO = c.BO; b.BO_s = c.BO_s; b.BO_pi = c.BO_pi; b.BO_pi2 = c.BO_pi2;
        b.C1dbo = c.C1dbo; b.C2dbo = c.C2dbo; b.C3dbo = c.C3dbo;
        b.C1dbopi = c.C1dbopi; b.C2dbopi = c.C2dbopi; b.C3dbopi = c.C3dbopi; b.C4dbopi = c.C4dbopi;
        b.C1dbopi2 = c.C1dbopi2; b.C2dbopi2 = c.C2dbopi2; b.C3dbopi2 = c.C3dbopi2; b.C4dbopi2 = c.C4dbopi2;
      } else {
        const Bond& s = bonds[j][static_cast<std::size_t>(b.sym)];
        b.BO = s.BO; b.BO_s = s.BO_s; b.BO_pi = s.BO_pi; b.BO_pi2 = s.BO_pi2;
      }
      total_bo[i] += b.BO;
    }
  }
  std::vector<terms::AtomQuantities<double>> aq(N);
  for (std::size_t j = 0; j < N; ++j) {
    const int tj = atoms.type[j];
    if (tj < 0) continue;
    const SingleBody& sj = ff.single(tj);
    aq[j] = terms::atom_quantities<double>(total_bo[j], sj.valency, sj.valency_e, sj.valency_boc, sj.valency_val, sj.nlp_opt,
                                           sj.flags.heavy_atom_terms, p_lp1);
  }

  BondedResult res;
  std::vector<double> CdDelta(N, 0.0);
  double e_bond = 0, e_lp = 0, e_ov = 0, e_un = 0;

  // ---- Bonds (owned centres, tag order selects one end of each bond) -------------------------------------------------------
  const bool gp37 = ff.gp37_stabilisation();
  for (std::size_t i = 0; i < n; ++i) {
    const int ti = atoms.type[i];
    for (Bond& b : bonds[i]) {
      const std::size_t j = static_cast<std::size_t>(b.nbr);
      if (atoms.tag[i] > atoms.tag[j]) continue;
      if (atoms.tag[i] == atoms.tag[j]) {
        const R3 xi = atoms.position(i), xj = atoms.position(j);
        if (xj[2] < xi[2]) continue;
        if (xj[2] == xi[2] && xj[1] < xi[1]) continue;
        if (xj[2] == xi[2] && xj[1] == xi[1] && xj[0] < xi[0]) continue;
      }
      const int tj = atoms.type[j];
      if (ti < 0 || tj < 0) continue;
      const TwoBody& tb = ff.pair(ti, tj);
      const auto pp = pair_params(tb);
      const auto be = terms::bond_energy<double>(pp, b.BO_s, b.BO_pi, b.BO_pi2);
      e_bond += be.e;
      b.Cdbo += be.CEbo;
      b.Cdbopi -= (be.CEbo + tb.De_p);
      b.Cdbopi2 -= (be.CEbo + tb.De_pp);
      if (b.BO >= 1.00 && tb.triple_bond_stabilisation) {
        const auto ts = terms::triple_bond_stabilisation<double>(gp3, gp4, gp7, gp10, b.BO, total_bo[i], total_bo[j], aq[i].Delta, aq[j].Delta);
        e_bond += ts.e;
        b.Cdbo += ts.decobdbo;
        CdDelta[i] += ts.decobdboua;
        CdDelta[j] += ts.decobdboub;
      }
    }
  }
  (void)gp37;

  // ---- Atom_Energy: lone pair + C2 correction, then over/under ------------------------------------------------------------
  for (std::size_t i = 0; i < n; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    const SingleBody& si = ff.single(ti);
    const bool active = !bonds[i].empty() || opt.enobonds;
    if (active) {
      const auto lp = terms::lone_pair<double>(si.p_lp2, aq[i].Delta_lp, aq[i].dDelta_lp);
      e_lp += lp.e;
      CdDelta[i] += lp.CElp;
    }
    if (c2_gate && si.flags.c2_species) {
      for (Bond& b : bonds[i]) {
        const int tj = atoms.type[static_cast<std::size_t>(b.nbr)];
        if (tj < 0 || !ff.single(tj).flags.c2_species) continue;
        const auto c2 = terms::c2_correction<double>(p_lp3, b.BO, aq[i].Delta);
        if (c2.vov3 > 3.) {
          e_lp += c2.e;
          b.Cdbo += c2.deahu2dbo;
          CdDelta[i] += c2.deahu2dsbo;
        }
      }
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    const SingleBody& si = ff.single(ti);
    const double dfvl = si.flags.heavy_atom_terms ? 0.0 : 1.0;
    double sum1 = 0, sum2 = 0;
    for (const Bond& b : bonds[i]) {
      const std::size_t j = static_cast<std::size_t>(b.nbr);
      const int tj = atoms.type[j];
      if (tj < 0) continue;
      const TwoBody& tb = ff.pair(ti, tj);
      sum1 += tb.p_ovun1 * tb.De_s * b.BO;
      sum2 += (aq[j].Delta - dfvl * aq[j].Delta_lp_temp) * (b.BO_pi + b.BO_pi2);
    }
    const bool under_active = !bonds[i].empty() || opt.enobonds;
    const auto ou = terms::over_under<double>(sum1, sum2, aq[i].Delta, aq[i].Delta_lp_temp, aq[i].dDelta_lp, dfvl, si.valency, si.p_ovun2,
                                              si.p_ovun5, p_ovun3, p_ovun4, p_ovun6, p_ovun7, p_ovun8, under_active);
    e_ov += ou.e_ov;
    if (under_active) e_un += ou.e_un;
    CdDelta[i] += ou.CEover3;
    if (under_active) CdDelta[i] += ou.CEunder3;
    for (Bond& b : bonds[i]) {
      const std::size_t j = static_cast<std::size_t>(b.nbr);
      const int tj = atoms.type[j];
      const TwoBody& tb = ff.pair(ti, tj);
      b.Cdbo += ou.CEover1 * tb.p_ovun1 * tb.De_s;
      const double pis = b.BO_pi + b.BO_pi2;
      const double dj = aq[j].Delta - dfvl * aq[j].Delta_lp_temp;
      CdDelta[j] += ou.CEover4 * (1.0 - dfvl * aq[j].dDelta_lp) * pis;
      b.Cdbopi += ou.CEover4 * dj;
      b.Cdbopi2 += ou.CEover4 * dj;
      CdDelta[j] += ou.CEunder4 * (1.0 - dfvl * aq[j].dDelta_lp) * pis;
      b.Cdbopi += ou.CEunder4 * dj;
      b.Cdbopi2 += ou.CEunder4 * dj;
    }
  }

  // ---- Compute_Total_Force: bond-derivative gather (Add_dBond_to_Forces) -------------------------------------------------------
  std::vector<R3> f(N, R3{0, 0, 0});
  for (std::size_t i = 0; i < N; ++i) {
    for (const Bond& bij : bonds[i]) {
      const std::size_t j = static_cast<std::size_t>(bij.nbr);
      if (!(i < j)) continue;
      const Bond& bji = bonds[j][static_cast<std::size_t>(bij.sym)];
      double c = bij.Cdbo + bji.Cdbo;
      const double C1dbo = bij.C1dbo * c, C2dbo = bij.C2dbo * c, C3dbo = bij.C3dbo * c;
      c = bij.Cdbopi + bji.Cdbopi;
      const double C1dbopi = bij.C1dbopi * c, C2dbopi = bij.C2dbopi * c, C3dbopi = bij.C3dbopi * c, C4dbopi = bij.C4dbopi * c;
      c = bij.Cdbopi2 + bji.Cdbopi2;
      const double C1dbopi2 = bij.C1dbopi2 * c, C2dbopi2 = bij.C2dbopi2 * c, C3dbopi2 = bij.C3dbopi2 * c, C4dbopi2 = bij.C4dbopi2 * c;
      c = CdDelta[i] + CdDelta[j];
      const double C1dDelta = bij.C1dbo * c, C2dDelta = bij.C2dbo * c, C3dDelta = bij.C3dbo * c;

      R3 t = scaled(C1dbo + C1dDelta + C2dbopi + C2dbopi2, bij.dBOp);
      scaled_add(t, C2dbo + C2dDelta + C3dbopi + C3dbopi2, dDeltap_self[i]);
      scaled_add(t, C1dbopi, bij.dln_BOp_pi);
      scaled_add(t, C1dbopi2, bij.dln_BOp_pi2);
      add(f[i], t);

      t = scaled(-(C1dbo + C1dDelta + C2dbopi + C2dbopi2), bij.dBOp);
      scaled_add(t, C3dbo + C3dDelta + C4dbopi + C4dbopi2, dDeltap_self[j]);
      scaled_add(t, -C1dbopi, bij.dln_BOp_pi);
      scaled_add(t, -C1dbopi2, bij.dln_BOp_pi2);
      add(f[j], t);

      for (const Bond& bk : bonds[i]) {
        const double ck = -(C2dbo + C2dDelta + C3dbopi + C3dbopi2);
        scaled_add(f[static_cast<std::size_t>(bk.nbr)], ck, bk.dBOp);
      }
      for (const Bond& bk : bonds[j]) {
        const double ck = -(C3dbo + C3dDelta + C4dbopi + C4dbopi2);
        scaled_add(f[static_cast<std::size_t>(bk.nbr)], ck, bk.dBOp);
      }
    }
  }

  res.e[EnergyTerm::Bond] = e_bond;
  res.e[EnergyTerm::LonePair] = e_lp;
  res.e[EnergyTerm::Over] = e_ov;
  res.e[EnergyTerm::Under] = e_un;
  res.grad.resize(3 * N);
  for (std::size_t i = 0; i < N; ++i) for (std::size_t c = 0; c < 3; ++c) res.grad[3 * i + c] = f[i][c];
  res.total_bo = total_bo;
  for (std::size_t i = 0; i < N; ++i) {
    res.stats.bonds += bonds[i].size();
    res.stats.max_bonds_per_atom = std::max(res.stats.max_bonds_per_atom, bonds[i].size());
  }
  res.stats.bonds /= 2;
  return res;
}

}  // namespace reaxmetal
