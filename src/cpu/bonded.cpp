// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/bonded.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "reaxmetal/terms_host.hpp"

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
inline double dot(const R3& a, const R3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline R3 cross(const R3& a, const R3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }

constexpr double kConstPi = 3.14159265;  // reaxff_defs.h: constPI (not pi; ENGINE_SPEC section 1)
inline double deg2rad(double a) { return a * kConstPi / 180.0; }
constexpr double kHbThreshold = 1e-2;    // HB_THRESHOLD
constexpr double kMinSine = 1e-10;       // MIN_SINE

// reaxff_valence_angles.cpp:36-70
void calculate_theta(const R3& dvec_ji, double d_ji, const R3& dvec_jk, double d_jk, double& theta, double& cos_theta) {
  cos_theta = dot(dvec_ji, dvec_jk) / (d_ji * d_jk);
  if (cos_theta > 1.) cos_theta = 1.0;
  if (cos_theta < -1.) cos_theta = -1.0;
  theta = std::acos(cos_theta);
}
void calculate_dcos_theta(const R3& dvec_ji, double d_ji, const R3& dvec_jk, double d_jk, R3& di, R3& dj, R3& dk) {
  const double sqr_d_ji = d_ji * d_ji, sqr_d_jk = d_jk * d_jk;
  const double inv_dists = 1.0 / (d_ji * d_jk);
  const double inv_dists3 = inv_dists * inv_dists * inv_dists;
  const double dot_dvecs = dot(dvec_ji, dvec_jk);
  const double Cdot_inv3 = dot_dvecs * inv_dists3;
  for (std::size_t t = 0; t < 3; ++t) {
    di[t] = dvec_jk[t] * inv_dists - Cdot_inv3 * sqr_d_jk * dvec_ji[t];
    dj[t] = -(dvec_jk[t] + dvec_ji[t]) * inv_dists + Cdot_inv3 * (sqr_d_jk * dvec_ji[t] + sqr_d_ji * dvec_jk[t]);
    dk[t] = dvec_ji[t] * inv_dists - Cdot_inv3 * sqr_d_ji * dvec_jk[t];
  }
}

struct ThreeBody {  // three_body_interaction_data
  int thb = 0, pthb = 0;
  double theta = 0;
  R3 dcos_di{}, dcos_dj{}, dcos_dk{};
};
struct HBondEntry { int nbr; double scl; double d; R3 dvec; };

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
  std::vector<std::vector<HBondEntry>> hbonds(N);
  for (std::size_t i = n; i < N; ++i) bond_mark[i] = 1000;  // ghosts start "infinitely" far from any owned atom

  // ---- bond list (Init_Forces_noQEq + BOp) ----------------------------------------------------------------------------------
  for (std::size_t i = 0; i < N; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    const SingleBody& si = ff.single(ti);
    for (std::size_t pj = far.row_start[i]; pj < far.row_start[i + 1]; ++pj) {
      const std::size_t j = static_cast<std::size_t>(far.nbr[pj]);
      const double d = far.dist[pj];
      if (!(d <= std::max(ctl.bond_cut, i < n ? ctl.hbond_cut : 0.0))) continue;   // Init_Forces_noQEq: cutoff = max(hbond_cut, bond_cut) for owned rows
      const int tj = atoms.type[j];
      if (tj < 0) continue;
      const SingleBody& sj = ff.single(tj);
      if (i < n && ctl.hbond_cut > 0 && (si.p_hbond == 1 || si.p_hbond == 2) && d <= ctl.hbond_cut) {   // Init_Forces_noQEq hydrogen-bond lists
        const double* hv = &far.dvec[3 * pj];
        if (si.p_hbond == 1 && sj.p_hbond == 2) hbonds[i].push_back({static_cast<int>(j), 1.0, d, {hv[0], hv[1], hv[2]}});
        else if (j < n && si.p_hbond == 2 && sj.p_hbond == 1) hbonds[j].push_back({static_cast<int>(i), -1.0, d, {hv[0], hv[1], hv[2]}});
      }
      if (!(d <= ctl.bond_cut)) continue;
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


  // ---- Valence_Angles: valence angle, penalty, 3-body conjugation; builds the three-body lists used by the torsions ------------------
  std::vector<R3> f(N, R3{0, 0, 0});
  double e_ang = 0, e_pen = 0, e_coa = 0, e_tor = 0, e_con = 0, e_hb = 0;
  std::vector<std::vector<std::vector<ThreeBody>>> thb(N);
  for (std::size_t j = 0; j < N; ++j) thb[j].resize(bonds[j].size());
  {
    const double p_val6 = gp[14], p_val8 = gp[33], p_val9 = gp[16], p_val10 = gp[17];
    for (std::size_t j = 0; j < N; ++j) {
      const int type_j = atoms.type[j];
      if (type_j < 0) continue;
      auto& bj = bonds[j];
      const double p_val3 = ff.single(type_j).p_val3, p_val5 = ff.single(type_j).p_val5;
      double SBOp = 0, prod_SBO = 1;
      for (const Bond& bt : bj) {
        SBOp += (bt.BO_pi + bt.BO_pi2);
        double temp = bt.BO * bt.BO;
        temp *= temp;
        temp *= temp;
        prod_SBO *= std::exp(-temp);
      }
      double vlpadj, dSBO2;
      if (aq[j].vlpex >= 0) { vlpadj = 0; dSBO2 = prod_SBO - 1; }
      else { vlpadj = aq[j].nlp; dSBO2 = (prod_SBO - 1) * (1 - p_val8 * aq[j].dDelta_lp); }
      const double Dboc = aq[j].Delta_boc;
      const double SBO = SBOp + (1 - prod_SBO) * (-Dboc - p_val8 * vlpadj);
      const double dSBO1 = -8 * prod_SBO * (Dboc + p_val8 * vlpadj);
      double SBO2, CSBO2;
      if (SBO <= 0) { SBO2 = 0; CSBO2 = 0; }
      else if (SBO > 0 && SBO <= 1) { SBO2 = std::pow(SBO, p_val9); CSBO2 = p_val9 * std::pow(SBO, p_val9 - 1); }
      else if (SBO > 1 && SBO < 2) { SBO2 = 2 - std::pow(2 - SBO, p_val9); CSBO2 = p_val9 * std::pow(2 - SBO, p_val9 - 1); }
      else { SBO2 = 2; CSBO2 = 0; }
      const double expval6 = std::exp(p_val6 * Dboc);

      for (std::size_t pi = 0; pi < bj.size(); ++pi) {
        Bond& bij = bj[pi];
        const double BOA_ij = bij.BO - ctl.thb_cut;
        if (BOA_ij > 0.0 && (j < n || static_cast<std::size_t>(bij.nbr) < n)) {
          const std::size_t i = static_cast<std::size_t>(bij.nbr);
          const int type_i = atoms.type[i];
          auto& list_i = thb[j][pi];
          for (std::size_t pk = 0; pk < pi; ++pk) {
            for (const ThreeBody& kji : thb[j][pk]) {
              if (kji.thb == static_cast<int>(i)) {
                ThreeBody t;
                t.thb = bj[pk].nbr; t.pthb = static_cast<int>(pk); t.theta = kji.theta;
                t.dcos_di = kji.dcos_dk; t.dcos_dj = kji.dcos_dj; t.dcos_dk = kji.dcos_di;
                list_i.push_back(t);
                break;
              }
            }
          }
          for (std::size_t pk = pi + 1; pk < bj.size(); ++pk) {
            Bond& bjk = bj[pk];
            const double BOA_jk = bjk.BO - ctl.thb_cut;
            const std::size_t k = static_cast<std::size_t>(bjk.nbr);
            const int type_k = atoms.type[k];
            ThreeBody t;
            double theta, cos_theta;
            calculate_theta(bij.dvec, bij.d, bjk.dvec, bjk.d, theta, cos_theta);
            calculate_dcos_theta(bij.dvec, bij.d, bjk.dvec, bjk.d, t.dcos_di, t.dcos_dj, t.dcos_dk);
            t.thb = static_cast<int>(k); t.pthb = static_cast<int>(pk); t.theta = theta;
            double sin_theta = std::sin(theta);
            if (sin_theta < 1.0e-5) sin_theta = 1.0e-5;
            list_i.push_back(t);

            if ((j < n) && (BOA_jk > 0.0) && (bij.BO > ctl.thb_cut) && (bjk.BO > ctl.thb_cut) && (bij.BO * bjk.BO > ctl.thb_cutsq)) {
              for (const ThreeBodySet& thbp : ff.three_body(type_i, type_j, type_k)) {
                if (!(std::fabs(thbp.p_val1) > 0.001)) continue;
                const double p_val1 = thbp.p_val1, p_val2 = thbp.p_val2, p_val4 = thbp.p_val4, p_val7 = thbp.p_val7, theta_00 = thbp.theta_00;
                // angle energy
                const double exp3ij = std::exp(-p_val3 * std::pow(BOA_ij, p_val4));
                const double f7_ij = 1.0 - exp3ij;
                const double Cf7ij = p_val3 * p_val4 * std::pow(BOA_ij, p_val4 - 1.0) * exp3ij;
                const double exp3jk = std::exp(-p_val3 * std::pow(BOA_jk, p_val4));
                const double f7_jk = 1.0 - exp3jk;
                const double Cf7jk = p_val3 * p_val4 * std::pow(BOA_jk, p_val4 - 1.0) * exp3jk;
                const double expval7 = std::exp(-p_val7 * Dboc);
                const double trm8 = 1.0 + expval6 + expval7;
                const double f8_Dj = p_val5 - ((p_val5 - 1.0) * (2.0 + expval6) / trm8);
                const double Cf8j = ((1.0 - p_val5) / (trm8 * trm8)) * (p_val6 * expval6 * trm8 - (2.0 + expval6) * (p_val6 * expval6 - p_val7 * expval7));
                double theta_0 = 180.0 - theta_00 * (1.0 - std::exp(-p_val10 * (2.0 - SBO2)));
                theta_0 = deg2rad(theta_0);
                const double expval2theta = std::exp(-p_val2 * ((theta_0 - theta) * (theta_0 - theta)));
                const double expval12theta = p_val1 >= 0 ? p_val1 * (1.0 - expval2theta) : p_val1 * -expval2theta;
                const double CEval1 = Cf7ij * f7_jk * f8_Dj * expval12theta;
                const double CEval2 = Cf7jk * f7_ij * f8_Dj * expval12theta;
                const double CEval3 = Cf8j * f7_ij * f7_jk * expval12theta;
                const double CEval4 = -2.0 * p_val1 * p_val2 * f7_ij * f7_jk * f8_Dj * expval2theta * (theta_0 - theta);
                const double Ctheta_0 = p_val10 * deg2rad(theta_00) * std::exp(-p_val10 * (2.0 - SBO2));
                const double CEval5 = -CEval4 * Ctheta_0 * CSBO2;
                const double CEval6 = CEval5 * dSBO1;
                const double CEval7 = CEval5 * dSBO2;
                const double CEval8 = -CEval4 / sin_theta;
                e_ang += f7_ij * f7_jk * f8_Dj * expval12theta;
                // penalty
                const double p_pen1 = thbp.p_pen1, p_pen2 = gp[19], p_pen3 = gp[20], p_pen4 = gp[21];
                const double exp_pen2ij = std::exp(-p_pen2 * ((BOA_ij - 2.0) * (BOA_ij - 2.0)));
                const double exp_pen2jk = std::exp(-p_pen2 * ((BOA_jk - 2.0) * (BOA_jk - 2.0)));
                const double exp_pen3 = std::exp(-p_pen3 * aq[j].Delta);
                const double exp_pen4 = std::exp(p_pen4 * aq[j].Delta);
                const double trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                const double f9_Dj = (2.0 + exp_pen3) / trm_pen34;
                const double Cf9j = (-p_pen3 * exp_pen3 * trm_pen34 - (2.0 + exp_pen3) * (-p_pen3 * exp_pen3 + p_pen4 * exp_pen4)) / (trm_pen34 * trm_pen34);
                const double e_pen_t = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
                e_pen += e_pen_t;
                const double CEpen1 = e_pen_t * Cf9j / f9_Dj;
                const double tmp = -2.0 * p_pen2 * e_pen_t;
                const double CEpen2 = tmp * (BOA_ij - 2.0), CEpen3 = tmp * (BOA_jk - 2.0);
                // coalition
                const double p_coa1 = thbp.p_coa1, p_coa2 = gp[2], p_coa3 = gp[38], p_coa4 = gp[30];
                const double exp_coa2 = std::exp(p_coa2 * aq[j].Delta_val);
                const double dti = total_bo[i] - BOA_ij, dtk = total_bo[k] - BOA_jk;
                const double e_coa_t = p_coa1 / (1. + exp_coa2) * std::exp(-p_coa3 * (dti * dti)) * std::exp(-p_coa3 * (dtk * dtk)) *
                                       std::exp(-p_coa4 * ((BOA_ij - 1.5) * (BOA_ij - 1.5))) * std::exp(-p_coa4 * ((BOA_jk - 1.5) * (BOA_jk - 1.5)));
                e_coa += e_coa_t;
                const double CEcoa1 = -2 * p_coa4 * (BOA_ij - 1.5) * e_coa_t;
                const double CEcoa2 = -2 * p_coa4 * (BOA_jk - 1.5) * e_coa_t;
                const double CEcoa3 = -p_coa2 * exp_coa2 * e_coa_t / (1 + exp_coa2);
                const double CEcoa4 = -2 * p_coa3 * dti * e_coa_t;
                const double CEcoa5 = -2 * p_coa3 * dtk * e_coa_t;
                // forces
                bij.Cdbo += (CEval1 + CEpen2 + (CEcoa1 - CEcoa4));
                bjk.Cdbo += (CEval2 + CEpen3 + (CEcoa2 - CEcoa5));
                CdDelta[j] += ((CEval3 + CEval7) + CEpen1 + CEcoa3);
                CdDelta[i] += CEcoa4;
                CdDelta[k] += CEcoa5;
                for (Bond& bt : bj) {
                  const double temp_bo_jt = bt.BO;
                  const double cube = temp_bo_jt * temp_bo_jt * temp_bo_jt;
                  const double pBOjt7 = cube * cube * temp_bo_jt;
                  bt.Cdbo += (CEval6 * pBOjt7);
                  bt.Cdbopi += CEval5;
                  bt.Cdbopi2 += CEval5;
                }
                scaled_add(f[i], CEval8, t.dcos_di);
                scaled_add(f[j], CEval8, t.dcos_dj);
                scaled_add(f[k], CEval8, t.dcos_dk);
              }
            }
          }
        }
      }
    }
  }

  // ---- Torsion_Angles: torsion and 4-body conjugation (owned j, tag order picks one end of each j-k bond) --------------------------------
  {
    const double p_tor2 = gp[23], p_tor3 = gp[24], p_tor4 = gp[25], p_cot2 = gp[27];
    for (std::size_t j = 0; j < n; ++j) {
      const int type_j = atoms.type[j];
      const double Delta_j = aq[j].Delta_boc;
      for (std::size_t pk = 0; pk < bonds[j].size(); ++pk) {
        Bond& bjk = bonds[j][pk];
        const std::size_t k = static_cast<std::size_t>(bjk.nbr);
        const double BOA_jk = bjk.BO - ctl.thb_cut;
        if (atoms.tag[j] > atoms.tag[k]) continue;
        if (atoms.tag[j] == atoms.tag[k]) {
          const R3 xj = atoms.position(j), xk = atoms.position(k);
          if (xk[2] < xj[2]) continue;
          if (xk[2] == xj[2] && xk[1] < xj[1]) continue;
          if (xk[2] == xj[2] && xk[1] == xj[1] && xk[0] < xj[0]) continue;
        }
        if (!(bjk.BO > ctl.thb_cut && !thb[j][pk].empty())) continue;
        const std::size_t pj = static_cast<std::size_t>(bjk.sym);
        if (thb[k][pj].empty()) continue;
        const int type_k = atoms.type[k];
        const double Delta_k = aq[k].Delta_boc;
        const double r_jk = bjk.d;
        const double exp_tor2_jk = std::exp(-p_tor2 * BOA_jk);
        const double exp_cot2_jk = std::exp(-p_cot2 * ((BOA_jk - 1.5) * (BOA_jk - 1.5)));
        const double exp_tor3_DjDk = std::exp(-p_tor3 * (Delta_j + Delta_k));
        const double exp_tor4_DjDk = std::exp(p_tor4 * (Delta_j + Delta_k));
        const double exp_tor34_inv = 1.0 / (1.0 + exp_tor3_DjDk + exp_tor4_DjDk);
        const double f11_DjDk = (2.0 + exp_tor3_DjDk) * exp_tor34_inv;

        for (const ThreeBody& p_ijk : thb[j][pk]) {
          const std::size_t pij = static_cast<std::size_t>(p_ijk.pthb);
          Bond& bij = bonds[j][pij];
          if (!(bij.BO > ctl.thb_cut)) continue;
          const std::size_t i = static_cast<std::size_t>(p_ijk.thb);
          const int type_i = atoms.type[i];
          const double r_ij = bij.d;
          const double BOA_ij = bij.BO - ctl.thb_cut;
          const double theta_ijk = p_ijk.theta;
          const double sin_ijk = std::sin(theta_ijk), cos_ijk = std::cos(theta_ijk);
          double tan_ijk_i;
          if (sin_ijk >= 0 && sin_ijk <= kMinSine) tan_ijk_i = cos_ijk / kMinSine;
          else if (sin_ijk <= 0 && sin_ijk >= -kMinSine) tan_ijk_i = cos_ijk / -kMinSine;
          else tan_ijk_i = cos_ijk / sin_ijk;
          const double exp_tor2_ij = std::exp(-p_tor2 * BOA_ij);
          const double exp_cot2_ij = std::exp(-p_cot2 * ((BOA_ij - 1.5) * (BOA_ij - 1.5)));

          for (const ThreeBody& p_jkl : thb[k][pj]) {
            const std::size_t l = static_cast<std::size_t>(p_jkl.thb);
            const std::size_t plk = static_cast<std::size_t>(p_jkl.pthb);
            Bond& bkl = bonds[k][plk];
            const int type_l = atoms.type[l];
            const FourBody* fbp = ff.four_body(type_i, type_j, type_k, type_l);
            if (!(i != l && fbp != nullptr && bkl.BO > ctl.thb_cut && bij.BO * bjk.BO * bkl.BO > ctl.thb_cut)) continue;
            const double r_kl = bkl.d;
            const double BOA_kl = bkl.BO - ctl.thb_cut;
            const double theta_jkl = p_jkl.theta;
            const double sin_jkl = std::sin(theta_jkl), cos_jkl = std::cos(theta_jkl);
            double tan_jkl_i;
            if (sin_jkl >= 0 && sin_jkl <= kMinSine) tan_jkl_i = cos_jkl / kMinSine;
            else if (sin_jkl <= 0 && sin_jkl >= -kMinSine) tan_jkl_i = cos_jkl / -kMinSine;
            else tan_jkl_i = cos_jkl / sin_jkl;
            const R3 xi = atoms.position(i), xl = atoms.position(l);
            const R3 dvec_li = {xi[0] - xl[0], xi[1] - xl[1], xi[2] - xl[2]};
            const double r_li = std::sqrt(dot(dvec_li, dvec_li));

            // Calculate_Omega (reaxff_torsion_angles.cpp:36-125)
            R3 dco_i{}, dco_j{}, dco_k{}, dco_l{};
            double omega;
            {
              double s_ijk = std::sin(p_ijk.theta), c_ijk = std::cos(p_ijk.theta), s_jkl = std::sin(p_jkl.theta), c_jkl = std::cos(p_jkl.theta);
              if (s_ijk >= 0 && s_ijk <= kMinSine) s_ijk = kMinSine; else if (s_ijk <= 0 && s_ijk >= -kMinSine) s_ijk = -kMinSine;
              if (s_jkl >= 0 && s_jkl <= kMinSine) s_jkl = kMinSine; else if (s_jkl <= 0 && s_jkl >= -kMinSine) s_jkl = -kMinSine;
              const R3 &dij = bij.dvec, &djk = bjk.dvec, &dkl = bkl.dvec;
              const double unnorm_cos_omega = -dot(dij, djk) * dot(djk, dkl) + (r_jk * r_jk) * dot(dij, dkl);
              const R3 cross_jk_kl = cross(djk, dkl);
              const double unnorm_sin_omega = -r_jk * dot(dij, cross_jk_kl);
              omega = std::atan2(unnorm_sin_omega, unnorm_cos_omega);
              const double htra = r_ij + c_ijk * (r_kl * c_jkl - r_jk);
              const double htrb = r_jk - r_ij * c_ijk - r_kl * c_jkl;
              const double htrc = r_kl + c_jkl * (r_ij * c_ijk - r_jk);
              const double hthd = r_ij * s_ijk * (r_jk - r_kl * c_jkl);
              const double hthe = r_kl * s_jkl * (r_jk - r_ij * c_ijk);
              const double hnra = r_kl * s_ijk * s_jkl;
              const double hnrc = r_ij * s_ijk * s_jkl;
              const double hnhd = r_ij * r_kl * c_ijk * s_jkl;
              const double hnhe = r_ij * r_kl * s_ijk * c_jkl;
              const double tel = (r_ij * r_ij) + (r_jk * r_jk) + (r_kl * r_kl) - (r_li * r_li) -
                                 2.0 * (r_ij * r_jk * c_ijk - r_ij * r_kl * c_ijk * c_jkl + r_jk * r_kl * c_jkl);
              const double poem = 2.0 * r_ij * r_kl * s_ijk * s_jkl;
              double arg = tel / poem;
              if (arg > 1.0) arg = 1.0;
              if (arg < -1.0) arg = -1.0;
              auto scaled_sum = [](double a, const R3& x, double b, const R3& y) { return R3{a * x[0] + b * y[0], a * x[1] + b * y[1], a * x[2] + b * y[2]}; };
              dco_i = scaled_sum((htra - arg * hnra) / r_ij, dij, -1., dvec_li);
              scaled_add(dco_i, -(hthd - arg * hnhd) / s_ijk, p_ijk.dcos_dk);
              dco_i = scaled(2.0 / poem, dco_i);
              dco_j = scaled_sum(-(htra - arg * hnra) / r_ij, dij, -htrb / r_jk, djk);
              scaled_add(dco_j, -(hthd - arg * hnhd) / s_ijk, p_ijk.dcos_dj);
              scaled_add(dco_j, -(hthe - arg * hnhe) / s_jkl, p_jkl.dcos_di);
              dco_j = scaled(2.0 / poem, dco_j);
              dco_k = scaled_sum(-(htrc - arg * hnrc) / r_kl, dkl, htrb / r_jk, djk);
              scaled_add(dco_k, -(hthd - arg * hnhd) / s_ijk, p_ijk.dcos_di);
              scaled_add(dco_k, -(hthe - arg * hnhe) / s_jkl, p_jkl.dcos_dj);
              dco_k = scaled(2.0 / poem, dco_k);
              dco_l = scaled_sum((htrc - arg * hnrc) / r_kl, dkl, 1., dvec_li);
              scaled_add(dco_l, -(hthe - arg * hnhe) / s_jkl, p_jkl.dcos_dk);
              dco_l = scaled(2.0 / poem, dco_l);
            }
            const double cos_omega = std::cos(omega), cos2omega = std::cos(2. * omega), cos3omega = std::cos(3. * omega);

            // torsion energy
            const double exp_tor1 = std::exp(fbp->p_tor1 * ((2.0 - bjk.BO_pi - f11_DjDk) * (2.0 - bjk.BO_pi - f11_DjDk)));
            const double exp_tor2_kl = std::exp(-p_tor2 * BOA_kl);
            const double exp_cot2_kl = std::exp(-p_cot2 * ((BOA_kl - 1.5) * (BOA_kl - 1.5)));
            const double fn10 = (1.0 - exp_tor2_ij) * (1.0 - exp_tor2_jk) * (1.0 - exp_tor2_kl);
            const double CV = 0.5 * (fbp->V1 * (1.0 + cos_omega) + fbp->V2 * exp_tor1 * (1.0 - cos2omega) + fbp->V3 * (1.0 + cos3omega));
            e_tor += fn10 * sin_ijk * sin_jkl * CV;
            const double dfn11 = (-p_tor3 * exp_tor3_DjDk + (p_tor3 * exp_tor3_DjDk - p_tor4 * exp_tor4_DjDk) * (2.0 + exp_tor3_DjDk) * exp_tor34_inv) * exp_tor34_inv;
            const double CEtors1 = sin_ijk * sin_jkl * CV;
            const double CEtors2 = -fn10 * 2.0 * fbp->p_tor1 * fbp->V2 * exp_tor1 * (2.0 - bjk.BO_pi - f11_DjDk) * (1.0 - (cos_omega * cos_omega)) * sin_ijk * sin_jkl;
            const double CEtors3 = CEtors2 * dfn11;
            const double CEtors4 = CEtors1 * p_tor2 * exp_tor2_ij * (1.0 - exp_tor2_jk) * (1.0 - exp_tor2_kl);
            const double CEtors5 = CEtors1 * p_tor2 * (1.0 - exp_tor2_ij) * exp_tor2_jk * (1.0 - exp_tor2_kl);
            const double CEtors6 = CEtors1 * p_tor2 * (1.0 - exp_tor2_ij) * (1.0 - exp_tor2_jk) * exp_tor2_kl;
            const double cmn = -fn10 * CV;
            const double CEtors7 = cmn * sin_jkl * tan_ijk_i;
            const double CEtors8 = cmn * sin_ijk * tan_jkl_i;
            const double CEtors9 = fn10 * sin_ijk * sin_jkl * (0.5 * fbp->V1 - 2.0 * fbp->V2 * exp_tor1 * cos_omega + 1.5 * fbp->V3 * (cos2omega + 2.0 * (cos_omega * cos_omega)));

            // 4-body conjugation
            const double fn12 = exp_cot2_ij * exp_cot2_jk * exp_cot2_kl;
            e_con += fbp->p_cot1 * fn12 * (1.0 + ((cos_omega * cos_omega) - 1.0) * sin_ijk * sin_jkl);
            const double Cconj = -2.0 * fn12 * fbp->p_cot1 * p_cot2 * (1.0 + ((cos_omega * cos_omega) - 1.0) * sin_ijk * sin_jkl);
            const double CEconj1 = Cconj * (BOA_ij - 1.5e0), CEconj2 = Cconj * (BOA_jk - 1.5e0), CEconj3 = Cconj * (BOA_kl - 1.5e0);
            const double CEconj4 = -fbp->p_cot1 * fn12 * ((cos_omega * cos_omega) - 1.0) * sin_jkl * tan_ijk_i;
            const double CEconj5 = -fbp->p_cot1 * fn12 * ((cos_omega * cos_omega) - 1.0) * sin_ijk * tan_jkl_i;
            const double CEconj6 = 2.0 * fbp->p_cot1 * fn12 * cos_omega * sin_ijk * sin_jkl;

            // forces
            bjk.Cdbopi += CEtors2;
            CdDelta[j] += CEtors3;
            CdDelta[k] += CEtors3;
            bij.Cdbo += (CEtors4 + CEconj1);
            bjk.Cdbo += (CEtors5 + CEconj2);
            bkl.Cdbo += (CEtors6 + CEconj3);
            scaled_add(f[i], CEtors7 + CEconj4, p_ijk.dcos_dk);
            scaled_add(f[j], CEtors7 + CEconj4, p_ijk.dcos_dj);
            scaled_add(f[k], CEtors7 + CEconj4, p_ijk.dcos_di);
            scaled_add(f[j], CEtors8 + CEconj5, p_jkl.dcos_di);
            scaled_add(f[k], CEtors8 + CEconj5, p_jkl.dcos_dj);
            scaled_add(f[l], CEtors8 + CEconj5, p_jkl.dcos_dk);
            scaled_add(f[i], CEtors9 + CEconj6, dco_i);
            scaled_add(f[j], CEtors9 + CEconj6, dco_j);
            scaled_add(f[k], CEtors9 + CEconj6, dco_k);
            scaled_add(f[l], CEtors9 + CEconj6, dco_l);
          }
        }
      }
    }
  }

  // ---- Hydrogen_Bonds ----------------------------------------------------------------------------------------------------------
  if (ctl.hbond_cut > 0) {
    for (std::size_t j = 0; j < n; ++j) {
      const int type_j = atoms.type[j];
      if (type_j < 0 || ff.single(type_j).p_hbond != 1) continue;
      std::vector<std::size_t> hblist;
      for (std::size_t pi = 0; pi < bonds[j].size(); ++pi) {
        const Bond& b = bonds[j][pi];
        const int type_i = atoms.type[static_cast<std::size_t>(b.nbr)];
        if (type_i < 0) continue;
        if (ff.single(type_i).p_hbond == 2 && b.BO >= kHbThreshold) hblist.push_back(pi);
      }
      for (const HBondEntry& hk : hbonds[j]) {
        const std::size_t k = static_cast<std::size_t>(hk.nbr);
        const int type_k = atoms.type[k];
        if (type_k < 0) continue;
        const double r_jk = hk.d;
        const R3 dvec_jk = scaled(hk.scl, hk.dvec);
        for (const std::size_t pi : hblist) {
          Bond& bij = bonds[j][pi];
          const std::size_t i = static_cast<std::size_t>(bij.nbr);
          if (atoms.tag[i] == atoms.tag[k]) continue;
          const int type_i = atoms.type[i];
          if (type_i < 0) continue;
          const HBondParams* hbp = ff.hbond(type_i, type_j, type_k);
          if (hbp == nullptr || hbp->r0_hb <= 0.0) continue;
          double theta, cos_theta;
          calculate_theta(bij.dvec, bij.d, dvec_jk, r_jk, theta, cos_theta);
          R3 dti{}, dtj{}, dtk{};
          calculate_dcos_theta(bij.dvec, bij.d, dvec_jk, r_jk, dti, dtj, dtk);
          const double sin_theta2 = std::sin(theta / 2.0);
          double sin_xhz4 = sin_theta2 * sin_theta2;
          sin_xhz4 *= sin_xhz4;
          const double cos_xhz1 = (1.0 - cos_theta);
          const double exp_hb2 = std::exp(-hbp->p_hb2 * bij.BO);
          const double exp_hb3 = std::exp(-hbp->p_hb3 * (hbp->r0_hb / r_jk + r_jk / hbp->r0_hb - 2.0));
          const double e_hb_t = hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
          e_hb += e_hb_t;
          const double CEhb1 = hbp->p_hb1 * hbp->p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
          const double CEhb2 = -hbp->p_hb1 / 2.0 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
          const double CEhb3 = -hbp->p_hb3 * (-hbp->r0_hb / (r_jk * r_jk) + 1.0 / hbp->r0_hb) * e_hb_t;
          bij.Cdbo += CEhb1;
          scaled_add(f[i], +CEhb2, dti);
          scaled_add(f[j], +CEhb2, dtj);
          scaled_add(f[k], +CEhb2, dtk);
          scaled_add(f[j], -CEhb3 / r_jk, dvec_jk);
          scaled_add(f[k], +CEhb3 / r_jk, dvec_jk);
        }
      }
    }
  }

  // ---- Compute_Total_Force: bond-derivative gather (Add_dBond_to_Forces) -------------------------------------------------------
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
  res.e[EnergyTerm::Valence] = e_ang;
  res.e[EnergyTerm::Penalty] = e_pen;
  res.e[EnergyTerm::Coalition] = e_coa;
  res.e[EnergyTerm::Torsion] = e_tor;
  res.e[EnergyTerm::Conjugation] = e_con;
  res.e[EnergyTerm::HBond] = e_hb;
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
