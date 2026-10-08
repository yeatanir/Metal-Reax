// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// Pure ReaxFF term functions (ADR-005): bond order, bond-order correction, lone pair / over- / under-coordination, bond energy.
// Templated on the scalar so that CPU-64 (double), the CPU-32 twin (float) and later the Metal kernels evaluate the SAME formulas.
// No exceptions, no STL containers, no element names or masses (ADR-003): everything element-specific arrives as data.
// Each function follows the pinned LAMMPS source line by line (stable_30Sep2026 @ 8de817dd, src/REAXFF); the expression order is the
// reference's. Literals that are part of the model (75, 25, 1e-8, 1e-10, ...) are kept as written upstream (ENGINE_SPEC section 1).
#include <cmath>

namespace reaxmetal::terms {

template <class T>
struct PairParams {  // the pieces of TwoBody the bonded terms use
  T p_bo1, p_bo2, p_bo3, p_bo4, p_bo5, p_bo6, r_s, r_p, r_pp;
  T p_boc3, p_boc4, p_boc5, ovc, v13cor;
  T p_be1, p_be2, De_s, De_p, De_pp, p_ovun1;
};

// ---- uncorrected bond order (BOp, reaxff_bond_orders.cpp:148-243) -----------------------------------------------------------
template <class T>
struct BoPrime {
  T BO_s, BO_pi, BO_pi2, C12, C34, C56;
  T total() const { return BO_s + BO_pi + BO_pi2; }  // before the bo_cut offset
};
template <class T>
inline BoPrime<T> bo_prime(const PairParams<T>& p, T d, bool s_ok, bool pi_ok, bool pp_ok, T bo_cut) {
  BoPrime<T> o{};
  if (s_ok) { o.C12 = p.p_bo1 * std::pow(d / p.r_s, p.p_bo2); o.BO_s = (T(1) + bo_cut) * std::exp(o.C12); }
  if (pi_ok) { o.C34 = p.p_bo3 * std::pow(d / p.r_p, p.p_bo4); o.BO_pi = std::exp(o.C34); }
  if (pp_ok) { o.C56 = p.p_bo5 * std::pow(d / p.r_pp, p.p_bo6); o.BO_pi2 = std::exp(o.C56); }
  return o;
}

// ---- bond-order correction (BO, reaxff_bond_orders.cpp:245-440), computed for one directed bond i->j ------------------------
// Input BO / BO_s / BO_pi / BO_pi2 are the offset-reduced primes (BO' - bo_cut, BO'_s - bo_cut).
template <class T>
struct BoCorrected {
  T BO, BO_s, BO_pi, BO_pi2;
  T C1dbo, C2dbo, C3dbo, C1dbopi, C2dbopi, C3dbopi, C4dbopi, C1dbopi2, C2dbopi2, C3dbopi2, C4dbopi2;
};
template <class T>
inline BoCorrected<T> bo_correct(const PairParams<T>& p, T p_boc1, T p_boc2, T val_i, T val_j, T Deltap_i, T Deltap_j, T Deltap_boc_i,
                                 T Deltap_boc_j, T BO, T BO_s, T BO_pi, T BO_pi2) {
  BoCorrected<T> o{};
  if (p.ovc < T(0.001) && p.v13cor < T(0.001)) {
    o.BO = BO; o.BO_s = BO_s; o.BO_pi = BO_pi; o.BO_pi2 = BO_pi2;
    o.C1dbo = T(1);
    o.C1dbopi = T(1);
    o.C1dbopi2 = T(1);
  } else {
    T f1, Cf1_ij, Cf1_ji, f4, f5, f4f5, Cf45_ij, Cf45_ji;
    if (p.ovc >= T(0.001)) {
      const T exp_p1i = std::exp(-p_boc1 * Deltap_i), exp_p2i = std::exp(-p_boc2 * Deltap_i);
      const T exp_p1j = std::exp(-p_boc1 * Deltap_j), exp_p2j = std::exp(-p_boc2 * Deltap_j);
      const T f2 = exp_p1i + exp_p1j;
      const T f3 = -T(1) / p_boc2 * std::log(T(0.5) * (exp_p2i + exp_p2j));
      f1 = T(0.5) * ((val_i + f2) / (val_i + f2 + f3) + (val_j + f2) / (val_j + f2 + f3));
      const T temp = f2 + f3;
      const T u1_ij = val_i + temp, u1_ji = val_j + temp;
      const T Cf1A_ij = T(0.5) * f3 * (T(1) / (u1_ij * u1_ij) + T(1) / (u1_ji * u1_ji));
      const T Cf1B_ij = -T(0.5) * ((u1_ij - f3) / (u1_ij * u1_ij) + (u1_ji - f3) / (u1_ji * u1_ji));
      Cf1_ij = T(0.50) * (-p_boc1 * exp_p1i / u1_ij - ((val_i + f2) / (u1_ij * u1_ij)) * (-p_boc1 * exp_p1i + exp_p2i / (exp_p2i + exp_p2j)) +
                          -p_boc1 * exp_p1i / u1_ji - ((val_j + f2) / (u1_ji * u1_ji)) * (-p_boc1 * exp_p1i + exp_p2i / (exp_p2i + exp_p2j)));
      Cf1_ji = -Cf1A_ij * p_boc1 * exp_p1j + Cf1B_ij * exp_p2j / (exp_p2i + exp_p2j);
    } else {
      f1 = T(1); Cf1_ij = Cf1_ji = T(0);
    }
    if (p.v13cor >= T(0.001)) {
      const T exp_f4 = std::exp(-(p.p_boc4 * (BO * BO) - Deltap_boc_i) * p.p_boc3 + p.p_boc5);
      const T exp_f5 = std::exp(-(p.p_boc4 * (BO * BO) - Deltap_boc_j) * p.p_boc3 + p.p_boc5);
      f4 = T(1) / (T(1) + exp_f4);
      f5 = T(1) / (T(1) + exp_f5);
      f4f5 = f4 * f5;
      Cf45_ij = -f4 * exp_f4;
      Cf45_ji = -f5 * exp_f5;
    } else {
      f4 = f5 = f4f5 = T(1); Cf45_ij = Cf45_ji = T(0);
    }
    const T A0_ij = f1 * f4f5;
    const T A1_ij = -2 * p.p_boc3 * p.p_boc4 * BO * (Cf45_ij + Cf45_ji);
    const T A2_ij = Cf1_ij / f1 + p.p_boc3 * Cf45_ij;
    const T A2_ji = Cf1_ji / f1 + p.p_boc3 * Cf45_ji;
    const T A3_ij = A2_ij + Cf1_ij / f1;
    const T A3_ji = A2_ji + Cf1_ji / f1;

    o.BO = BO * A0_ij;
    o.BO_pi = BO_pi * A0_ij * f1;
    o.BO_pi2 = BO_pi2 * A0_ij * f1;
    o.BO_s = o.BO - (o.BO_pi + o.BO_pi2);
    o.C1dbo = A0_ij + o.BO * A1_ij;
    o.C2dbo = o.BO * A2_ij;
    o.C3dbo = o.BO * A2_ji;
    o.C1dbopi = f1 * f1 * f4 * f5;
    o.C2dbopi = o.BO_pi * A1_ij;
    o.C3dbopi = o.BO_pi * A3_ij;
    o.C4dbopi = o.BO_pi * A3_ji;
    o.C1dbopi2 = f1 * f1 * f4 * f5;
    o.C2dbopi2 = o.BO_pi2 * A1_ij;
    o.C3dbopi2 = o.BO_pi2 * A3_ij;
    o.C4dbopi2 = o.BO_pi2 * A3_ji;
  }
  // neglect bonds that are < 1e-10
  if (o.BO < T(1e-10)) o.BO = T(0);
  if (o.BO_s < T(1e-10)) o.BO_s = T(0);
  if (o.BO_pi < T(1e-10)) o.BO_pi = T(0);
  if (o.BO_pi2 < T(1e-10)) o.BO_pi2 = T(0);
  return o;
}

// ---- atom quantities after correction (reaxff_bond_orders.cpp:437-466) ------------------------------------------------------
// `heavy` = mass > 21.0 (compat flag heavy_atom_terms). `trunc` toward zero is (int) in C (ENGINE_SPEC Q-14).
template <class T>
struct AtomQuantities {
  T Delta, Delta_e, Delta_boc, Delta_val, vlpex, nlp, Delta_lp, Clp, dDelta_lp, nlp_temp, Delta_lp_temp, dDelta_lp_temp;
};
template <class T>
inline AtomQuantities<T> atom_quantities(T total_bo, T valency, T valency_e, T valency_boc, T valency_val, T nlp_opt, bool heavy, T p_lp1) {
  AtomQuantities<T> a{};
  a.Delta = total_bo - valency;
  a.Delta_e = total_bo - valency_e;
  a.Delta_boc = total_bo - valency_boc;
  a.Delta_val = total_bo - valency_val;
  const T half_trunc = static_cast<T>(static_cast<int>(a.Delta_e / T(2)));
  a.vlpex = a.Delta_e - T(2) * half_trunc;
  const T explp1 = std::exp(-p_lp1 * ((T(2) + a.vlpex) * (T(2) + a.vlpex)));
  a.nlp = explp1 - half_trunc;
  a.Delta_lp = nlp_opt - a.nlp;
  a.Clp = T(2) * p_lp1 * explp1 * (T(2) + a.vlpex);
  a.dDelta_lp = a.Clp;
  if (heavy) {
    a.nlp_temp = T(0.5) * (valency_e - valency);
    a.Delta_lp_temp = nlp_opt - a.nlp_temp;
    a.dDelta_lp_temp = T(0);
  } else {
    a.nlp_temp = a.nlp;
    a.Delta_lp_temp = nlp_opt - a.nlp_temp;
    a.dDelta_lp_temp = a.Clp;
  }
  return a;
}

// ---- bond energy of one undirected bond (Bonds, reaxff_bonds.cpp:73-93) -----------------------------------------------------
template <class T>
struct BondEnergy { T e, CEbo; };
template <class T>
inline BondEnergy<T> bond_energy(const PairParams<T>& p, T BO_s, T BO_pi, T BO_pi2) {
  const T pow_BOs_be2 = BO_s == T(0) ? T(0) : std::pow(BO_s, p.p_be2);
  const T exp_be12 = std::exp(p.p_be1 * (T(1) - pow_BOs_be2));
  BondEnergy<T> o;
  o.CEbo = -p.De_s * exp_be12 * (T(1) - p.p_be1 * p.p_be2 * pow_BOs_be2);
  o.e = -p.De_s * BO_s * exp_be12 - p.De_p * BO_pi - p.De_pp * BO_pi2;
  return o;
}

// terminal triple-bond stabilisation (reaxff_bonds.cpp:96-124); only called when BO >= 1 and the pair flag is set
template <class T>
struct TripleStab { T e, decobdbo, decobdboua, decobdboub; };
template <class T>
inline TripleStab<T> triple_bond_stabilisation(T gp3, T gp4, T gp7, T gp10, T BO, T tbo_i, T tbo_j, T Delta_i, T Delta_j) {
  const T exphu = std::exp(-gp7 * ((BO - T(2.50)) * (BO - T(2.50))));
  const T exphua1 = std::exp(-gp3 * (tbo_i - BO));
  const T exphub1 = std::exp(-gp3 * (tbo_j - BO));
  const T exphuov = std::exp(gp4 * (Delta_i + Delta_j));
  const T hulpov = T(1) / (T(1) + T(25.0) * exphuov);
  TripleStab<T> o;
  o.e = gp10 * exphu * hulpov * (exphua1 + exphub1);
  o.decobdbo = gp10 * exphu * hulpov * (exphua1 + exphub1) * (gp3 - T(2.0) * gp7 * (BO - T(2.50)));
  o.decobdboua = -gp10 * exphu * hulpov * (gp3 * exphua1 + T(25.0) * gp4 * exphuov * hulpov * (exphua1 + exphub1));
  o.decobdboub = -gp10 * exphu * hulpov * (gp3 * exphub1 + T(25.0) * gp4 * exphuov * hulpov * (exphua1 + exphub1));
  return o;
}

// ---- lone pair (Atom_Energy, reaxff_multi_body.cpp:77-97) -------------------------------------------------------------------
template <class T>
struct LonePair { T e, CElp; };
template <class T>
inline LonePair<T> lone_pair(T p_lp2, T Delta_lp, T dDelta_lp) {
  const T expvd2 = std::exp(-75 * Delta_lp);
  const T inv_expvd2 = T(1) / (T(1) + expvd2);
  LonePair<T> o;
  o.e = p_lp2 * Delta_lp * inv_expvd2;
  const T dElp = p_lp2 * inv_expvd2 + 75 * p_lp2 * Delta_lp * expvd2 * (inv_expvd2 * inv_expvd2);
  o.CElp = dElp * dDelta_lp;
  return o;
}

// C2 correction of one directed C-C bond (reaxff_multi_body.cpp:100-125); vov3 > 3 is the caller's gate
template <class T>
struct C2Corr { T vov3, e, deahu2dbo, deahu2dsbo; };
template <class T>
inline C2Corr<T> c2_correction(T p_lp3, T BO, T Di) {
  C2Corr<T> o;
  o.vov3 = BO - Di - T(0.040) * std::pow(Di, T(4.));
  const T d = o.vov3 - T(3.0);
  o.e = p_lp3 * (d * d);
  o.deahu2dbo = T(2.) * p_lp3 * (o.vov3 - T(3.));
  o.deahu2dsbo = T(2.) * p_lp3 * (o.vov3 - T(3.)) * (-T(1.) - T(0.16) * std::pow(Di, T(3.)));
  return o;
}

// ---- over- and under-coordination of one atom (Atom_Energy, reaxff_multi_body.cpp:131-208) ---------------------------------
template <class T>
struct OverUnder {
  T e_ov, e_un, CEover1, CEover3, CEover4, CEunder3, CEunder4;
};
template <class T>
inline OverUnder<T> over_under(T sum_ovun1, T sum_ovun2, T Delta, T Delta_lp_temp, T dDelta_lp, T dfvl, T valency, T p_ovun2, T p_ovun5,
                               T p_ovun3, T p_ovun4, T p_ovun6, T p_ovun7, T p_ovun8, bool under_active) {
  const T exp_ovun1 = p_ovun3 * std::exp(p_ovun4 * sum_ovun2);
  const T inv_exp_ovun1 = T(1.0) / (1 + exp_ovun1);
  const T Delta_lpcorr = Delta - (dfvl * Delta_lp_temp) * inv_exp_ovun1;
  const T exp_ovun2 = std::exp(p_ovun2 * Delta_lpcorr);
  const T inv_exp_ovun2 = T(1.0) / (T(1.0) + exp_ovun2);
  const T DlpVi = T(1.0) / (Delta_lpcorr + valency + T(1e-8));
  OverUnder<T> o{};
  const T CEover1 = Delta_lpcorr * DlpVi * inv_exp_ovun2;
  o.CEover1 = CEover1;
  o.e_ov = sum_ovun1 * CEover1;
  const T CEover2 = sum_ovun1 * DlpVi * inv_exp_ovun2 * (T(1.0) - Delta_lpcorr * (DlpVi + p_ovun2 * exp_ovun2 * inv_exp_ovun2));
  o.CEover3 = CEover2 * (T(1.0) - dfvl * dDelta_lp * inv_exp_ovun1);
  o.CEover4 = CEover2 * (dfvl * Delta_lp_temp) * p_ovun4 * exp_ovun1 * (inv_exp_ovun1 * inv_exp_ovun1);

  const T exp_ovun2n = T(1.0) / exp_ovun2;
  const T exp_ovun6 = std::exp(p_ovun6 * Delta_lpcorr);
  const T exp_ovun8 = p_ovun7 * std::exp(p_ovun8 * sum_ovun2);
  const T inv_exp_ovun2n = T(1.0) / (T(1.0) + exp_ovun2n);
  const T inv_exp_ovun8 = T(1.0) / (T(1.0) + exp_ovun8);
  // the reference evaluates e_un (and CEunder2 from it) unconditionally, but only adds it to the energy and CdDelta when `under_active`
  const T e_un = under_active ? -p_ovun5 * (T(1.0) - exp_ovun6) * inv_exp_ovun2n * inv_exp_ovun8 : T(0);
  o.e_un = e_un;
  const T CEunder1 = inv_exp_ovun2n * (p_ovun5 * p_ovun6 * exp_ovun6 * inv_exp_ovun8 + p_ovun2 * e_un * exp_ovun2n);
  const T CEunder2 = -e_un * p_ovun8 * exp_ovun8 * inv_exp_ovun8;
  o.CEunder3 = CEunder1 * (T(1.0) - dfvl * dDelta_lp * inv_exp_ovun1);
  o.CEunder4 = CEunder1 * (dfvl * Delta_lp_temp) * p_ovun4 * exp_ovun1 * (inv_exp_ovun1 * inv_exp_ovun1) + CEunder2;
  return o;
}

// ---- nonbonded: taper, van der Waals (types 1/2/3, optional lg), shielded Coulomb, polarization -------------------------------
// reaxff_init_md.cpp:72-106 (coefficients, computed in double on the host), reaxff_nonbonded.cpp:36-215 (pair), :36-57 (polarization).
// ENGINE_SPEC 5.7-5.9. Constants as written upstream (truncated ones included): C_ele, KCALpMOL_to_EV, 0.33333333333333.
inline constexpr double kCele = 332.06371;        // reaxff_defs.h: C_ele
inline constexpr double kKcalpmolToEv = 23.02;    // reaxff_defs.h: KCALpMOL_to_EV (really eV -> kcal/mol, imprecise as upstream)

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
// Algebraically identical form in t = (r - swa)/(swb - swa): Tap = 1 - 35t^4 + 84t^5 - 70t^6 + 20t^7 (no cancellation of O(10) terms in
// float; ADR-007 stable form, used by the CPU-32 twin and the Metal kernels). dTap includes the 1/r factor like the Horner version.
template <class T>
inline void taper_stable(T swa, T swb, T r, T& Tap, T& dTap) {
  const T d = swb - swa, t = (r - swa) / d, t2 = t * t, t3 = t2 * t, t4 = t2 * t2;
  Tap = T(1) - t4 * (T(35) - T(84) * t + T(70) * t2 - T(20) * t3);
  dTap = (t3 * (T(-140) + T(420) * t - T(420) * t2 + T(140) * t3)) / (d * r);
}

template <class T>
struct NbPair {  // the TwoBody members the nonbonded terms use
  T alpha, D, r_vdW, gamma_w, gamma, ecore, acore, rcore, lgcij, lgre;
};
template <class T>
struct NbResult { T e_vdW, e_ele, CE; };  // CE = CEvd + CEclmb: gradient on i is +CE*dvec? (see bonded/nonbonded engine: f[i] += -CE*dvec)
// vdw_type: 1 shielding, 2 inner wall, 3 both (gp.vdw_type); lg only when `lg` and vdw_type is 2 or 3 (nested as upstream, Q-06).
template <class T>
inline NbResult<T> nonbonded_pair(const NbPair<T>& p, int vdw_type, bool lg, T p_vdW1, T qi, T qj, T r, T Tap, T dTap) {
  const T p_vdW1i = T(1.0) / p_vdW1;
  T e_vdW, CEvd;
  NbResult<T> o{};
  if (vdw_type == 1 || vdw_type == 3) {
    const T powr_vdW1 = std::pow(r, p_vdW1);
    const T powgi_vdW1 = std::pow(T(1.0) / p.gamma_w, p_vdW1);
    const T fn13 = std::pow(powr_vdW1 + powgi_vdW1, p_vdW1i);
    const T exp1 = std::exp(p.alpha * (T(1.0) - fn13 / p.r_vdW));
    const T exp2 = std::exp(T(0.5) * p.alpha * (T(1.0) - fn13 / p.r_vdW));
    e_vdW = p.D * (exp1 - T(2.0) * exp2);
    o.e_vdW += Tap * e_vdW;
    const T dfn13 = std::pow(powr_vdW1 + powgi_vdW1, p_vdW1i - T(1.0)) * std::pow(r, p_vdW1 - T(2.0));
    CEvd = dTap * e_vdW - Tap * p.D * (p.alpha / p.r_vdW) * (exp1 - exp2) * dfn13;
  } else {
    const T exp1 = std::exp(p.alpha * (T(1.0) - r / p.r_vdW));
    const T exp2 = std::exp(T(0.5) * p.alpha * (T(1.0) - r / p.r_vdW));
    e_vdW = p.D * (exp1 - T(2.0) * exp2);
    o.e_vdW += Tap * e_vdW;
    CEvd = dTap * e_vdW - Tap * p.D * (p.alpha / p.r_vdW) * (exp1 - exp2) / r;
  }
  if (vdw_type == 2 || vdw_type == 3) {
    const T e_core = p.ecore * std::exp(p.acore * (T(1.0) - (r / p.rcore)));
    o.e_vdW += Tap * e_core;
    const T de_core = -(p.acore / p.rcore) * e_core;
    CEvd += dTap * e_core + Tap * de_core / r;
    if (lg) {
      const T r_ij5 = std::pow(r, T(5.0)), r_ij6 = std::pow(r, T(6.0)), re6 = std::pow(p.lgre, T(6.0));
      const T e_lg = -(p.lgcij / (r_ij6 + re6));
      o.e_vdW += Tap * e_lg;
      const T de_lg = -T(6.0) * e_lg * r_ij5 / (r_ij6 + re6);
      CEvd += dTap * e_lg + Tap * de_lg / r;
    }
  }
  const T dr3gamij_1 = (r * r * r + p.gamma);
  const T dr3gamij_3 = std::pow(dr3gamij_1, T(0.33333333333333));
  const T tmp = Tap / dr3gamij_3;
  o.e_ele = T(kCele) * qi * qj * tmp;
  const T CEclmb = T(kCele) * qi * qj * (dTap - Tap * r / dr3gamij_1) / dr3gamij_3;
  o.CE = CEvd + CEclmb;
  return o;
}
template <class T>
inline T polarization(T chi, T eta, T q) { return T(kKcalpmolToEv) * (chi * q + (eta / T(2.)) * (q * q)); }

}  // namespace reaxmetal::terms
