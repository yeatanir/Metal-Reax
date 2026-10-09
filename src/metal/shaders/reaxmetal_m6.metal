// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Metal kernels of the bonded terms (milestone M4/M6): bond list, bond orders, bond / lone pair / over- / under-coordination, valence angle,
// penalty, 3-body conjugation, torsion, 4-body conjugation, hydrogen bonds, and the force assembly, all in FP32 on the GPU.
// Appended after terms.hpp and the M4 kernels in the assembled shader source. Same rules as M3: no double, no atomics, no threadgroup
// communication; every result is a function of the inputs only (deterministic).
//
// Parallel decomposition (ADR-006, no atomics, no floating-point scatter):
//   * thread per atom (all owned + ghost atoms where the reference needs them, owned centres for the energy terms);
//   * every directed bond a->j lives in slot s of atom a; slots are sorted by ascending neighbor index (the order of the CPU-64 engine);
//   * a term evaluated at centre a that must update OTHER atoms/bonds writes the update into a slot-addressed scratch owned by a
//     (RM_SF_CDN, RM_SF_FN*, the torsion scratch tkl/tfl, the H-bond force scratch); the receiving atom later GATHERS those scratch values in
//     a fixed order through the symmetric slot index (RM_SI_SYM). Each memory location is written by one thread only.
// The CPU-64 engine src/cpu/bonded.cpp is the specification of every formula here (checked against pinned LAMMPS to 3.5e-15).

#define RM_B_ARGS                                                                                                                              \
  device const float* x [[buffer(0)]], device const int* type [[buffer(1)]], device const int* tag [[buffer(2)]],                             \
  device const uint* atom_cell [[buffer(3)]], device const uint* cell_start [[buffer(4)]], device const uint* cell_items [[buffer(5)]],      \
  device const float* sb_f [[buffer(6)]], device const int* sb_i [[buffer(7)]], device const float* tb_f [[buffer(8)]],                      \
  device const int* tb_i [[buffer(9)]], device const int* thb_idx [[buffer(10)]], device const float* thb_sets [[buffer(11)]],              \
  device const float* fb_f [[buffer(12)]], device const int* fb_has [[buffer(13)]], device const float* hb_f [[buffer(14)]],                \
  device const float* gp [[buffer(15)]], device float* wf [[buffer(16)]], device int* wi [[buffer(17)]],                                      \
  device const float* xlo [[buffer(18)]], constant RmBParams& p [[buffer(19)]], uint a [[thread_position_in_grid]]

#define SFi(f, at, s) wf[(uint)(f) * p.NB + (at) * p.B + (s)]
#define AFi(f, at) wf[p.o_atom + (uint)(f) * p.N + (at)]
#define SIi(f, at, s) wi[(uint)(f) * p.NB + (at) * p.B + (s)]
#define AIi(f, at) wi[p.o_iatom + (uint)(f) * p.N + (at)]
#define RMEXP(v) reaxmetal::terms::safe_exp<float>(v)

// coordinate difference x_j - x_a (component c) from the hi/lo position pair: exact for close atoms, so the error is relative to the distance
inline float rm_dx(device const float* x, device const float* xlo, uint j, uint a, uint c) {
  return (x[3 * j + c] - x[3 * a + c]) + (xlo[3 * j + c] - xlo[3 * a + c]);
}

inline uint rm_nslots(device const int* wi, constant RmBParams& p, uint at) {
  const int nb = wi[p.o_iatom + (uint)RM_AI_NB * p.N + at];
  return (uint)nb < p.B ? (uint)nb : p.B;
}

inline reaxmetal::terms::PairParams<float> rm_pair(device const float* tb_f, uint idx) {
  reaxmetal::terms::PairParams<float> q;
  device const float* t = tb_f + idx * RM_B_TB_F;
  q.p_bo1 = t[0]; q.p_bo2 = t[1]; q.p_bo3 = t[2]; q.p_bo4 = t[3]; q.p_bo5 = t[4]; q.p_bo6 = t[5];
  q.r_s = t[6]; q.r_p = t[7]; q.r_pp = t[8]; q.p_boc3 = t[9]; q.p_boc4 = t[10]; q.p_boc5 = t[11]; q.ovc = t[12]; q.v13cor = t[13];
  q.p_be1 = t[14]; q.p_be2 = t[15]; q.De_s = t[16]; q.De_p = t[17]; q.De_pp = t[18]; q.p_ovun1 = t[19];
  return q;
}

// true if the directed bond a -> j is the end that counts the bond in the tag-ordered terms (reaxff_bonds.cpp:50-62, reaxff_torsion_angles.cpp:166-176)
inline bool rm_counts_bond(device const float* x, device const int* tag, uint a, uint j) {
  if (tag[a] > tag[j]) return false;
  if (tag[a] == tag[j]) {
    const float xa0 = x[3 * a], xa1 = x[3 * a + 1], xa2 = x[3 * a + 2];
    const float xj0 = x[3 * j], xj1 = x[3 * j + 1], xj2 = x[3 * j + 2];
    if (xj2 < xa2) return false;
    if (xj2 == xa2 && xj1 < xa1) return false;
    if (xj2 == xa2 && xj1 == xa1 && xj0 < xa0) return false;
  }
  return true;
}

// ---- 1. bond list: all neighbors within bond_cut whose uncorrected bond order reaches bo_cut, ascending neighbor index ----------------------
kernel void rm_b_build(RM_B_ARGS) {
  if (a >= p.N) return;
  const int ta = type[a];
  uint cnt = 0;
  if (ta >= 0) {
    const uint c = atom_cell[a];
    const uint cx = c % p.ncx, cy = (c / p.ncx) % p.ncy, cz = c / (p.ncx * p.ncy);
    const uint z0 = (cz > 0) ? cz - 1 : 0, z1 = (cz + 1 < p.ncz) ? cz + 1 : p.ncz - 1;
    const uint y0 = (cy > 0) ? cy - 1 : 0, y1 = (cy + 1 < p.ncy) ? cy + 1 : p.ncy - 1;
    const uint x0 = (cx > 0) ? cx - 1 : 0, x1 = (cx + 1 < p.ncx) ? cx + 1 : p.ncx - 1;
    for (uint gz = z0; gz <= z1; ++gz)
      for (uint gy = y0; gy <= y1; ++gy)
        for (uint gx = x0; gx <= x1; ++gx) {
          const uint cell = (gz * p.ncy + gy) * p.ncx + gx;
          for (uint q = cell_start[cell]; q < cell_start[cell + 1]; ++q) {
            const uint j = cell_items[q];
            if (j == a) continue;
            const int tj = type[j];
            if (tj < 0) continue;
            const float dx = rm_dx(x, xlo, j, a, 0), dy = rm_dx(x, xlo, j, a, 1), dz = rm_dx(x, xlo, j, a, 2);
            const float d = sqrt(dx * dx + dy * dy + dz * dz);
            if (!(d <= p.bond_cut)) continue;
            const uint ti = (uint)ta, tjj = (uint)tj;
            const reaxmetal::terms::PairParams<float> pp = rm_pair(tb_f, ti * p.ntypes + tjj);
            const bool s_ok = sb_f[ti * RM_B_SB_F + 5] > 0.0f && sb_f[tjj * RM_B_SB_F + 5] > 0.0f;
            const bool pi_ok = sb_f[ti * RM_B_SB_F + 6] > 0.0f && sb_f[tjj * RM_B_SB_F + 6] > 0.0f;
            const bool pp_ok = sb_f[ti * RM_B_SB_F + 7] > 0.0f && sb_f[tjj * RM_B_SB_F + 7] > 0.0f;
            const reaxmetal::terms::BoPrime<float> bp = reaxmetal::terms::bo_prime<float>(pp, d, s_ok, pi_ok, pp_ok, p.bo_cut);
            if (!(bp.total() >= p.bo_cut)) continue;
            if (cnt < p.B) {
              uint s = cnt;
              while (s > 0 && SIi(RM_SI_NBR, a, s - 1) > (int)j) { SIi(RM_SI_NBR, a, s) = SIi(RM_SI_NBR, a, s - 1); --s; }
              SIi(RM_SI_NBR, a, s) = (int)j;
            }
            ++cnt;
          }
        }
  }
  AIi(RM_AI_NB, a) = (int)cnt;
}

// ---- 2. per-slot uncorrected bond order, derivative geometry, symmetric slot, Delta' sums (BOp, reaxff_bond_orders.cpp:148-243) -----------------
kernel void rm_b_prime(RM_B_ARGS) {
  if (a >= p.N) return;
  const int ta = type[a];
  if (ta < 0) return;
  const uint n = rm_nslots(wi, p, a);
  float total = 0.0f, ddx = 0.0f, ddy = 0.0f, ddz = 0.0f;
  for (uint s = 0; s < n; ++s) {
    const uint j = (uint)SIi(RM_SI_NBR, a, s);
    const uint ti = (uint)ta, tj = (uint)type[j];
    const float dx = rm_dx(x, xlo, j, a, 0), dy = rm_dx(x, xlo, j, a, 1), dz = rm_dx(x, xlo, j, a, 2);
    const float d = sqrt(dx * dx + dy * dy + dz * dz);
    const reaxmetal::terms::PairParams<float> pp = rm_pair(tb_f, ti * p.ntypes + tj);
    const bool s_ok = sb_f[ti * RM_B_SB_F + 5] > 0.0f && sb_f[tj * RM_B_SB_F + 5] > 0.0f;
    const bool pi_ok = sb_f[ti * RM_B_SB_F + 6] > 0.0f && sb_f[tj * RM_B_SB_F + 6] > 0.0f;
    const bool pp_ok = sb_f[ti * RM_B_SB_F + 7] > 0.0f && sb_f[tj * RM_B_SB_F + 7] > 0.0f;
    const reaxmetal::terms::BoPrime<float> bp = reaxmetal::terms::bo_prime<float>(pp, d, s_ok, pi_ok, pp_ok, p.bo_cut);
    const float rr2 = 1.0f / (d * d);
    const float cs = pp.p_bo2 * bp.C12 * rr2, cpi = pp.p_bo4 * bp.C34 * rr2, cpi2 = pp.p_bo6 * bp.C56 * rr2;
    const float kdb = -(bp.BO_s * cs + bp.BO_pi * cpi + bp.BO_pi2 * cpi2);
    SFi(RM_SF_D, a, s) = d;
    SFi(RM_SF_DVX, a, s) = dx; SFi(RM_SF_DVY, a, s) = dy; SFi(RM_SF_DVZ, a, s) = dz;
    SFi(RM_SF_DBX, a, s) = kdb * dx; SFi(RM_SF_DBY, a, s) = kdb * dy; SFi(RM_SF_DBZ, a, s) = kdb * dz;
    SFi(RM_SF_DPX, a, s) = -bp.BO_pi * cpi * dx; SFi(RM_SF_DPY, a, s) = -bp.BO_pi * cpi * dy; SFi(RM_SF_DPZ, a, s) = -bp.BO_pi * cpi * dz;
    SFi(RM_SF_DQX, a, s) = -bp.BO_pi2 * cpi2 * dx; SFi(RM_SF_DQY, a, s) = -bp.BO_pi2 * cpi2 * dy; SFi(RM_SF_DQZ, a, s) = -bp.BO_pi2 * cpi2 * dz;
    SFi(RM_SF_BOPS, a, s) = bp.BO_s - p.bo_cut;
    SFi(RM_SF_BOPPI, a, s) = bp.BO_pi;
    SFi(RM_SF_BOPPI2, a, s) = bp.BO_pi2;
    const float bop = bp.total() - p.bo_cut;
    SFi(RM_SF_BOP, a, s) = bop;
    total += bop;
    ddx += kdb * dx; ddy += kdb * dy; ddz += kdb * dz;
    uint sym = 0;
    const uint nj = rm_nslots(wi, p, j);
    for (uint u = 0; u < nj; ++u) if ((uint)SIi(RM_SI_NBR, j, u) == a) { sym = u; break; }
    SIi(RM_SI_SYM, a, s) = (int)sym;
  }
  AFi(RM_AF_DELTAP, a) = total - sb_f[(uint)ta * RM_B_SB_F + 0];
  AFi(RM_AF_DELTAP_BOC, a) = total - sb_f[(uint)ta * RM_B_SB_F + 3];
  AFi(RM_AF_DDPX, a) = ddx; AFi(RM_AF_DDPY, a) = ddy; AFi(RM_AF_DDPZ, a) = ddz;
}

// ---- 3. corrected bond orders (each directed bond from its own end: the correction is symmetric) and atom quantities -----------------------
kernel void rm_b_correct(RM_B_ARGS) {
  if (a >= p.N) return;
  const int ta = type[a];
  if (ta < 0) return;
  const uint n = rm_nslots(wi, p, a);
  const uint ti = (uint)ta;
  float total = 0.0f;
  for (uint s = 0; s < n; ++s) {
    const uint j = (uint)SIi(RM_SI_NBR, a, s);
    const uint tj = (uint)type[j];
    const reaxmetal::terms::PairParams<float> pp = rm_pair(tb_f, ti * p.ntypes + tj);
    const reaxmetal::terms::BoCorrected<float> c = reaxmetal::terms::bo_correct<float>(
        pp, gp[0], gp[1], sb_f[ti * RM_B_SB_F + 0], sb_f[tj * RM_B_SB_F + 0], AFi(RM_AF_DELTAP, a), AFi(RM_AF_DELTAP, j), AFi(RM_AF_DELTAP_BOC, a),
        AFi(RM_AF_DELTAP_BOC, j), SFi(RM_SF_BOP, a, s), SFi(RM_SF_BOPS, a, s), SFi(RM_SF_BOPPI, a, s), SFi(RM_SF_BOPPI2, a, s));
    SFi(RM_SF_BO, a, s) = c.BO; SFi(RM_SF_BOS, a, s) = c.BO_s; SFi(RM_SF_BOPI, a, s) = c.BO_pi; SFi(RM_SF_BOPI2, a, s) = c.BO_pi2;
    SFi(RM_SF_C1, a, s) = c.C1dbo; SFi(RM_SF_C2, a, s) = c.C2dbo; SFi(RM_SF_C3, a, s) = c.C3dbo;
    SFi(RM_SF_C1P, a, s) = c.C1dbopi; SFi(RM_SF_C2P, a, s) = c.C2dbopi; SFi(RM_SF_C3P, a, s) = c.C3dbopi; SFi(RM_SF_C4P, a, s) = c.C4dbopi;
    SFi(RM_SF_C1Q, a, s) = c.C1dbopi2; SFi(RM_SF_C2Q, a, s) = c.C2dbopi2; SFi(RM_SF_C3Q, a, s) = c.C3dbopi2; SFi(RM_SF_C4Q, a, s) = c.C4dbopi2;
    total += c.BO;
  }
  const reaxmetal::terms::AtomQuantities<float> q = reaxmetal::terms::atom_quantities<float>(
      total, sb_f[ti * RM_B_SB_F + 0], sb_f[ti * RM_B_SB_F + 1], sb_f[ti * RM_B_SB_F + 2], sb_f[ti * RM_B_SB_F + 3], sb_f[ti * RM_B_SB_F + 4],
      sb_i[ti * RM_B_SB_I + 0] != 0, gp[15]);
  AFi(RM_AF_TOTBO, a) = total;
  AIi(RM_AI_CLP, a) = (int)(q.Delta_e / 2.0f);   // decision census: the truncation of Delta_e / 2
  AFi(RM_AF_DELTA, a) = q.Delta; AFi(RM_AF_DELTA_BOC, a) = q.Delta_boc; AFi(RM_AF_DELTA_VAL, a) = q.Delta_val; AFi(RM_AF_VLPEX, a) = q.vlpex;
  AFi(RM_AF_NLP, a) = q.nlp; AFi(RM_AF_DELTA_LP, a) = q.Delta_lp; AFi(RM_AF_DDELTA_LP, a) = q.dDelta_lp; AFi(RM_AF_DELTA_LP_TEMP, a) = q.Delta_lp_temp;
}

// ---- 4. bond energy, lone pair (+ C2), over- and under-coordination at owned atoms --------------------------------------------------------
kernel void rm_b_atom(RM_B_ARGS) {
  if (a >= p.nlocal) return;
  const int ta = type[a];
  if (ta < 0) return;
  const uint ti = (uint)ta;
  const uint n = rm_nslots(wi, p, a);
  float e_bond = 0.0f, e_lp = 0.0f, e_ov = 0.0f, e_un = 0.0f, cdself = 0.0f;
  // Bonds
  for (uint s = 0; s < n; ++s) {
    const uint j = (uint)SIi(RM_SI_NBR, a, s);
    if (!rm_counts_bond(x, tag, a, j)) continue;
    const uint tj = (uint)type[j];
    const reaxmetal::terms::PairParams<float> pp = rm_pair(tb_f, ti * p.ntypes + tj);
    const reaxmetal::terms::BondEnergy<float> be = reaxmetal::terms::bond_energy<float>(pp, SFi(RM_SF_BOS, a, s), SFi(RM_SF_BOPI, a, s), SFi(RM_SF_BOPI2, a, s));
    e_bond += be.e;
    SFi(RM_SF_CDBO, a, s) += be.CEbo;
    SFi(RM_SF_CDBOPI, a, s) -= (be.CEbo + pp.De_p);
    SFi(RM_SF_CDBOPI2, a, s) -= (be.CEbo + pp.De_pp);
    const float bo = SFi(RM_SF_BO, a, s);
    if (bo >= 1.00f && tb_i[ti * p.ntypes + tj] != 0) {
      const reaxmetal::terms::TripleStab<float> ts = reaxmetal::terms::triple_bond_stabilisation<float>(
          gp[3], gp[4], gp[7], gp[10], bo, AFi(RM_AF_TOTBO, a), AFi(RM_AF_TOTBO, j), AFi(RM_AF_DELTA, a), AFi(RM_AF_DELTA, j));
      e_bond += ts.e;
      SFi(RM_SF_CDBO, a, s) += ts.decobdbo;
      cdself += ts.decobdboua;
      SFi(RM_SF_CDN, a, s) += ts.decobdboub;
    }
  }
  // lone pair and C2 correction
  const bool active = n > 0 || p.enobonds != 0;
  if (active) {
    const reaxmetal::terms::LonePair<float> lp = reaxmetal::terms::lone_pair<float>(sb_f[ti * RM_B_SB_F + 8], AFi(RM_AF_DELTA_LP, a), AFi(RM_AF_DDELTA_LP, a));
    e_lp += lp.e;
    cdself += lp.CElp;
  }
  if (gp[5] > 0.001f && sb_i[ti * RM_B_SB_I + 1] != 0) {
    for (uint s = 0; s < n; ++s) {
      const uint j = (uint)SIi(RM_SI_NBR, a, s);
      if (sb_i[(uint)type[j] * RM_B_SB_I + 1] == 0) continue;
      const reaxmetal::terms::C2Corr<float> c2 = reaxmetal::terms::c2_correction<float>(gp[5], SFi(RM_SF_BO, a, s), AFi(RM_AF_DELTA, a));
      if (c2.vov3 > 3.0f) {
        e_lp += c2.e;
        SFi(RM_SF_CDBO, a, s) += c2.deahu2dbo;
        cdself += c2.deahu2dsbo;
      }
    }
  }
  // over / under
  const float dfvl = sb_i[ti * RM_B_SB_I + 0] != 0 ? 0.0f : 1.0f;
  float sum1 = 0.0f, sum2 = 0.0f;
  for (uint s = 0; s < n; ++s) {
    const uint j = (uint)SIi(RM_SI_NBR, a, s);
    const uint tj = (uint)type[j];
    const reaxmetal::terms::PairParams<float> pp = rm_pair(tb_f, ti * p.ntypes + tj);
    sum1 += pp.p_ovun1 * pp.De_s * SFi(RM_SF_BO, a, s);
    sum2 += (AFi(RM_AF_DELTA, j) - dfvl * AFi(RM_AF_DELTA_LP_TEMP, j)) * (SFi(RM_SF_BOPI, a, s) + SFi(RM_SF_BOPI2, a, s));
  }
  const reaxmetal::terms::OverUnder<float> ou = reaxmetal::terms::over_under<float>(
      sum1, sum2, AFi(RM_AF_DELTA, a), AFi(RM_AF_DELTA_LP_TEMP, a), AFi(RM_AF_DDELTA_LP, a), dfvl, sb_f[ti * RM_B_SB_F + 0], sb_f[ti * RM_B_SB_F + 9],
      sb_f[ti * RM_B_SB_F + 10], gp[32], gp[31], gp[6], gp[8], gp[9], active);
  e_ov += ou.e_ov;
  if (active) e_un += ou.e_un;
  cdself += ou.CEover3;
  if (active) cdself += ou.CEunder3;
  for (uint s = 0; s < n; ++s) {
    const uint j = (uint)SIi(RM_SI_NBR, a, s);
    const uint tj = (uint)type[j];
    const reaxmetal::terms::PairParams<float> pp = rm_pair(tb_f, ti * p.ntypes + tj);
    SFi(RM_SF_CDBO, a, s) += ou.CEover1 * pp.p_ovun1 * pp.De_s;
    const float pis = SFi(RM_SF_BOPI, a, s) + SFi(RM_SF_BOPI2, a, s);
    const float dj = AFi(RM_AF_DELTA, j) - dfvl * AFi(RM_AF_DELTA_LP_TEMP, j);
    SFi(RM_SF_CDN, a, s) += ou.CEover4 * (1.0f - dfvl * AFi(RM_AF_DDELTA_LP, j)) * pis;
    SFi(RM_SF_CDBOPI, a, s) += ou.CEover4 * dj;
    SFi(RM_SF_CDBOPI2, a, s) += ou.CEover4 * dj;
    SFi(RM_SF_CDN, a, s) += ou.CEunder4 * (1.0f - dfvl * AFi(RM_AF_DDELTA_LP, j)) * pis;
    SFi(RM_SF_CDBOPI, a, s) += ou.CEunder4 * dj;
    SFi(RM_SF_CDBOPI2, a, s) += ou.CEunder4 * dj;
  }
  AFi(RM_AF_EBOND, a) = e_bond; AFi(RM_AF_ELP, a) = e_lp; AFi(RM_AF_EOV, a) = e_ov; AFi(RM_AF_EUN, a) = e_un;
  AFi(RM_AF_CDSELF, a) += cdself;
}

// ---- helpers for the angle terms (reaxff_valence_angles.cpp:36-70) -------------------------------------------------------------------------
inline float rm_dot3(float ax, float ay, float az, float bx, float by, float bz) { return ax * bx + ay * by + az * bz; }

// theta between vectors u (length du) and v (length dv); cos clamped like the reference
// Angle between two bond vectors. theta = atan2(|u x v|, u.v) and sin(theta) = |u x v| / (|u||v|) are accurate to float rounding over the whole range, whereas
// acos(cos) of a float cosine near +-1 loses half the digits (error ~3e-4 rad) and sin(theta) then has relative errors of order 1 for nearly collinear atoms,
// where the angle forces carry a factor 1 / sin(theta). cth is the (clamped) cosine as the reference computes it.
inline void rm_theta(float ux, float uy, float uz, float du, float vx, float vy, float vz, float dv, thread float& theta, thread float& cth, thread float& sth) {
  const float dot = rm_dot3(ux, uy, uz, vx, vy, vz);
  cth = dot / (du * dv);
  if (cth > 1.0f) cth = 1.0f;
  if (cth < -1.0f) cth = -1.0f;
  const float cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
  const float cr = sqrt(cx * cx + cy * cy + cz * cz);
  theta = atan2(cr, dot);
  sth = cr / (du * dv);
  if (sth > 1.0f) sth = 1.0f;
}

// derivative of cos(theta) with respect to the three atoms; arguments are dvec_ji (centre -> i) and dvec_jk (centre -> k)
inline void rm_dcos(float ux, float uy, float uz, float du, float vx, float vy, float vz, float dv, thread float* di, thread float* dj, thread float* dk) {
  const float sqr_u = du * du, sqr_v = dv * dv;
  const float inv_dists = 1.0f / (du * dv);
  const float inv_dists3 = inv_dists * inv_dists * inv_dists;
  const float cdot = rm_dot3(ux, uy, uz, vx, vy, vz) * inv_dists3;
  const float u[3] = {ux, uy, uz}, v[3] = {vx, vy, vz};
  for (uint t = 0; t < 3; ++t) {
    di[t] = v[t] * inv_dists - cdot * sqr_v * u[t];
    dj[t] = -(v[t] + u[t]) * inv_dists + cdot * (sqr_v * u[t] + sqr_u * v[t]);
    dk[t] = u[t] * inv_dists - cdot * sqr_u * v[t];
  }
}

// ---- 5. valence angle, penalty, 3-body conjugation (centre j = owned atom) --------------------------------------------------------------
kernel void rm_b_valence(RM_B_ARGS) {
  const uint j = a;
  if (j >= p.nlocal) return;
  const int tjj = type[j];
  if (tjj < 0) return;
  const uint tj = (uint)tjj;
  const uint n = rm_nslots(wi, p, j);
  const float p_val6 = gp[14], p_val8 = gp[33], p_val9 = gp[16], p_val10 = gp[17];
  const float p_val3 = sb_f[tj * RM_B_SB_F + 11], p_val5 = sb_f[tj * RM_B_SB_F + 12];
  float SBOp = 0.0f, prod_SBO = 1.0f;
  for (uint t = 0; t < n; ++t) {
    SBOp += (SFi(RM_SF_BOPI, j, t) + SFi(RM_SF_BOPI2, j, t));
    float temp = SFi(RM_SF_BO, j, t) * SFi(RM_SF_BO, j, t);
    temp *= temp;
    temp *= temp;
    prod_SBO *= RMEXP(-temp);
  }
  float vlpadj, dSBO2;
  if (AFi(RM_AF_VLPEX, j) >= 0.0f) { vlpadj = 0.0f; dSBO2 = prod_SBO - 1.0f; }
  else { vlpadj = AFi(RM_AF_NLP, j); dSBO2 = (prod_SBO - 1.0f) * (1.0f - p_val8 * AFi(RM_AF_DDELTA_LP, j)); }
  const float Dboc = AFi(RM_AF_DELTA_BOC, j);
  const float SBO = SBOp + (1.0f - prod_SBO) * (-Dboc - p_val8 * vlpadj);
  const float dSBO1 = -8.0f * prod_SBO * (Dboc + p_val8 * vlpadj);
  float SBO2, CSBO2;
  if (SBO <= 0.0f) { SBO2 = 0.0f; CSBO2 = 0.0f; }
  else if (SBO > 0.0f && SBO <= 1.0f) { SBO2 = pow(SBO, p_val9); CSBO2 = p_val9 * pow(SBO, p_val9 - 1.0f); }
  else if (SBO > 1.0f && SBO < 2.0f) { SBO2 = 2.0f - pow(2.0f - SBO, p_val9); CSBO2 = p_val9 * pow(2.0f - SBO, p_val9 - 1.0f); }
  else { SBO2 = 2.0f; CSBO2 = 0.0f; }
  const float expval6 = RMEXP(p_val6 * Dboc);
  AIi(RM_AI_CSBO, j) = SBO <= 0.0f ? 0 : (SBO <= 1.0f ? 1 : (SBO < 2.0f ? 2 : 3));   // decision census
  const float constpi = 3.14159265f;
  float e_ang = 0.0f, e_pen = 0.0f, e_coa = 0.0f;
  float fsx = 0.0f, fsy = 0.0f, fsz = 0.0f;
  float cdself = 0.0f;

  for (uint pi = 0; pi < n; ++pi) {
    const float BOA_ij = SFi(RM_SF_BO, j, pi) - p.thb_cut;
    if (!(BOA_ij > 0.0f)) continue;
    const uint i = (uint)SIi(RM_SI_NBR, j, pi);
    const uint ti = (uint)type[i];
    for (uint pk = pi + 1; pk < n; ++pk) {
      const float BOA_jk = SFi(RM_SF_BO, j, pk) - p.thb_cut;
      const uint k = (uint)SIi(RM_SI_NBR, j, pk);
      const uint tk = (uint)type[k];
      if (!((BOA_jk > 0.0f) && (SFi(RM_SF_BO, j, pi) > p.thb_cut) && (SFi(RM_SF_BO, j, pk) > p.thb_cut) &&
            (SFi(RM_SF_BO, j, pi) * SFi(RM_SF_BO, j, pk) > p.thb_cutsq)))
        continue;
      AIi(RM_AI_CTHB, j) += 1;
      const float uxi = SFi(RM_SF_DVX, j, pi), uyi = SFi(RM_SF_DVY, j, pi), uzi = SFi(RM_SF_DVZ, j, pi), dij = SFi(RM_SF_D, j, pi);
      const float uxk = SFi(RM_SF_DVX, j, pk), uyk = SFi(RM_SF_DVY, j, pk), uzk = SFi(RM_SF_DVZ, j, pk), djk = SFi(RM_SF_D, j, pk);
      float theta, cos_theta, sth_valence;
      rm_theta(uxi, uyi, uzi, dij, uxk, uyk, uzk, djk, theta, cos_theta, sth_valence);
      float dti[3], dtj[3], dtk[3];
      rm_dcos(uxi, uyi, uzi, dij, uxk, uyk, uzk, djk, dti, dtj, dtk);
      float sin_theta = sth_valence;
      if (sin_theta < 1.0e-5f) sin_theta = 1.0e-5f;
      const uint tri = (ti * p.ntypes + tj) * p.ntypes + tk;
      const int start = thb_idx[2 * tri], cnt = thb_idx[2 * tri + 1];
      for (int c = 0; c < cnt; ++c) {
        device const float* ts = thb_sets + (uint)(start + c) * RM_B_THB_F;
        const float theta_00 = ts[0], p_val1 = ts[1], p_val2 = ts[2], p_val4 = ts[3], p_val7 = ts[4], p_pen1 = ts[5], p_coa1 = ts[6];
        if (!(fabs(p_val1) > 0.001f)) continue;
        // angle energy
        const float exp3ij = RMEXP(-p_val3 * pow(BOA_ij, p_val4));
        const float f7_ij = 1.0f - exp3ij;
        const float Cf7ij = p_val3 * p_val4 * pow(BOA_ij, p_val4 - 1.0f) * exp3ij;
        const float exp3jk = RMEXP(-p_val3 * pow(BOA_jk, p_val4));
        const float f7_jk = 1.0f - exp3jk;
        const float Cf7jk = p_val3 * p_val4 * pow(BOA_jk, p_val4 - 1.0f) * exp3jk;
        const float expval7 = RMEXP(-p_val7 * Dboc);
        const float trm8 = 1.0f + expval6 + expval7;
        const float f8_Dj = p_val5 - ((p_val5 - 1.0f) * (2.0f + expval6) / trm8);
        const float Cf8j = ((1.0f - p_val5) / (trm8 * trm8)) * (p_val6 * expval6 * trm8 - (2.0f + expval6) * (p_val6 * expval6 - p_val7 * expval7));
        float theta_0 = 180.0f - theta_00 * (1.0f - RMEXP(-p_val10 * (2.0f - SBO2)));
        theta_0 = theta_0 * constpi / 180.0f;
        const float expval2theta = RMEXP(-p_val2 * ((theta_0 - theta) * (theta_0 - theta)));
        const float expval12theta = p_val1 >= 0.0f ? p_val1 * (1.0f - expval2theta) : p_val1 * -expval2theta;
        const float CEval1 = Cf7ij * f7_jk * f8_Dj * expval12theta;
        const float CEval2 = Cf7jk * f7_ij * f8_Dj * expval12theta;
        const float CEval3 = Cf8j * f7_ij * f7_jk * expval12theta;
        const float CEval4 = -2.0f * p_val1 * p_val2 * f7_ij * f7_jk * f8_Dj * expval2theta * (theta_0 - theta);
        const float Ctheta_0 = p_val10 * (theta_00 * constpi / 180.0f) * RMEXP(-p_val10 * (2.0f - SBO2));
        const float CEval5 = -CEval4 * Ctheta_0 * CSBO2;
        const float CEval6 = CEval5 * dSBO1;
        const float CEval7 = CEval5 * dSBO2;
        const float CEval8 = -CEval4 / sin_theta;
        e_ang += f7_ij * f7_jk * f8_Dj * expval12theta;
        // penalty
        const float p_pen2 = gp[19], p_pen3 = gp[20], p_pen4 = gp[21];
        const float exp_pen2ij = RMEXP(-p_pen2 * ((BOA_ij - 2.0f) * (BOA_ij - 2.0f)));
        const float exp_pen2jk = RMEXP(-p_pen2 * ((BOA_jk - 2.0f) * (BOA_jk - 2.0f)));
        const float exp_pen3 = RMEXP(-p_pen3 * AFi(RM_AF_DELTA, j));
        const float exp_pen4 = RMEXP(p_pen4 * AFi(RM_AF_DELTA, j));
        const float trm_pen34 = 1.0f + exp_pen3 + exp_pen4;
        const float f9_Dj = (2.0f + exp_pen3) / trm_pen34;
        const float Cf9j = (-p_pen3 * exp_pen3 * trm_pen34 - (2.0f + exp_pen3) * (-p_pen3 * exp_pen3 + p_pen4 * exp_pen4)) / (trm_pen34 * trm_pen34);
        const float e_pen_t = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
        e_pen += e_pen_t;
        const float CEpen1 = e_pen_t * Cf9j / f9_Dj;
        const float tmp = -2.0f * p_pen2 * e_pen_t;
        const float CEpen2 = tmp * (BOA_ij - 2.0f), CEpen3 = tmp * (BOA_jk - 2.0f);
        // coalition
        const float p_coa2 = gp[2], p_coa3 = gp[38], p_coa4 = gp[30];
        const float exp_coa2 = RMEXP(p_coa2 * AFi(RM_AF_DELTA_VAL, j));
        const float dti_ = AFi(RM_AF_TOTBO, i) - BOA_ij, dtk_ = AFi(RM_AF_TOTBO, k) - BOA_jk;
        const float e_coa_t = p_coa1 / (1.0f + exp_coa2) * RMEXP(-p_coa3 * (dti_ * dti_)) * RMEXP(-p_coa3 * (dtk_ * dtk_)) *
                              RMEXP(-p_coa4 * ((BOA_ij - 1.5f) * (BOA_ij - 1.5f))) * RMEXP(-p_coa4 * ((BOA_jk - 1.5f) * (BOA_jk - 1.5f)));
        e_coa += e_coa_t;
        const float CEcoa1 = -2.0f * p_coa4 * (BOA_ij - 1.5f) * e_coa_t;
        const float CEcoa2 = -2.0f * p_coa4 * (BOA_jk - 1.5f) * e_coa_t;
        const float CEcoa3 = -p_coa2 * exp_coa2 * e_coa_t / (1.0f + exp_coa2);
        const float CEcoa4 = -2.0f * p_coa3 * dti_ * e_coa_t;
        const float CEcoa5 = -2.0f * p_coa3 * dtk_ * e_coa_t;
        // forces
        SFi(RM_SF_CDBO, j, pi) += (CEval1 + CEpen2 + (CEcoa1 - CEcoa4));
        SFi(RM_SF_CDBO, j, pk) += (CEval2 + CEpen3 + (CEcoa2 - CEcoa5));
        cdself += ((CEval3 + CEval7) + CEpen1 + CEcoa3);
        SFi(RM_SF_CDN, j, pi) += CEcoa4;
        SFi(RM_SF_CDN, j, pk) += CEcoa5;
        for (uint t = 0; t < n; ++t) {
          const float bo_t = SFi(RM_SF_BO, j, t);
          const float cube = bo_t * bo_t * bo_t;
          const float pBOjt7 = cube * cube * bo_t;
          SFi(RM_SF_CDBO, j, t) += (CEval6 * pBOjt7);
          SFi(RM_SF_CDBOPI, j, t) += CEval5;
          SFi(RM_SF_CDBOPI2, j, t) += CEval5;
        }
        SFi(RM_SF_FNX, j, pi) += CEval8 * dti[0]; SFi(RM_SF_FNY, j, pi) += CEval8 * dti[1]; SFi(RM_SF_FNZ, j, pi) += CEval8 * dti[2];
        SFi(RM_SF_FNX, j, pk) += CEval8 * dtk[0]; SFi(RM_SF_FNY, j, pk) += CEval8 * dtk[1]; SFi(RM_SF_FNZ, j, pk) += CEval8 * dtk[2];
        fsx += CEval8 * dtj[0]; fsy += CEval8 * dtj[1]; fsz += CEval8 * dtj[2];
      }
    }
  }
  AFi(RM_AF_EANG, j) = e_ang; AFi(RM_AF_EPEN, j) = e_pen; AFi(RM_AF_ECOA, j) = e_coa;
  AFi(RM_AF_CDSELF, j) += cdself;
  AFi(RM_AF_FSX, j) += fsx; AFi(RM_AF_FSY, j) += fsy; AFi(RM_AF_FSZ, j) += fsz;
}

// ---- 6. torsion and 4-body conjugation: one thread per owned centre atom j, central bonds j-k in tag order (reaxff_torsion_angles.cpp) ---------
// Entries: i = every other bond of j with BO_ij > thb_cut; l = every other bond of k with BO_kl > thb_cut that the reference's three-body list
// contains: a later slot always, an earlier slot only if k or l is owned (gate of reaxff_valence_angles.cpp:208, mirrored entries).
kernel void rm_b_torsion(RM_B_ARGS) {
  const uint j = a;
  if (j >= p.nlocal) return;
  const int tjj = type[j];
  if (tjj < 0) return;
  const uint tj = (uint)tjj;
  const uint n = rm_nslots(wi, p, j);
  const float p_tor2 = gp[23], p_tor3 = gp[24], p_tor4 = gp[25], p_cot2 = gp[27];
  const float Delta_j = AFi(RM_AF_DELTA_BOC, j);
  const float MIN_SINE = 1e-10f;
  float e_tor = 0.0f, e_con = 0.0f, cdself = 0.0f, fsx = 0.0f, fsy = 0.0f, fsz = 0.0f;
  for (uint pk = 0; pk < n; ++pk) {
    const uint k = (uint)SIi(RM_SI_NBR, j, pk);
    if (!rm_counts_bond(x, tag, j, k)) continue;
    const float BO_jk = SFi(RM_SF_BO, j, pk);
    if (!(BO_jk > p.thb_cut)) continue;
    const uint pj = (uint)SIi(RM_SI_SYM, j, pk);
    const uint nk = rm_nslots(wi, p, k);
    const uint tk = (uint)type[k];
    const float BOA_jk = BO_jk - p.thb_cut;
    const float Delta_k = AFi(RM_AF_DELTA_BOC, k);
    const float r_jk = SFi(RM_SF_D, j, pk);
    const float jkx = SFi(RM_SF_DVX, j, pk), jky = SFi(RM_SF_DVY, j, pk), jkz = SFi(RM_SF_DVZ, j, pk);
    const float exp_tor2_jk = RMEXP(-p_tor2 * BOA_jk);
    const float exp_cot2_jk = RMEXP(-p_cot2 * ((BOA_jk - 1.5f) * (BOA_jk - 1.5f)));
    const float exp_tor3_DjDk = RMEXP(-p_tor3 * (Delta_j + Delta_k));
    const float exp_tor4_DjDk = RMEXP(p_tor4 * (Delta_j + Delta_k));
    const float exp_tor34_inv = 1.0f / (1.0f + exp_tor3_DjDk + exp_tor4_DjDk);
    const float f11_DjDk = (2.0f + exp_tor3_DjDk) * exp_tor34_inv;
    const float BOpi_jk = SFi(RM_SF_BOPI, j, pk);

    for (uint pij = 0; pij < n; ++pij) {
      if (pij == pk) continue;
      const float BO_ij = SFi(RM_SF_BO, j, pij);
      if (!(BO_ij > p.thb_cut)) continue;
      const uint i = (uint)SIi(RM_SI_NBR, j, pij);
      const uint ti = (uint)type[i];
      const float r_ij = SFi(RM_SF_D, j, pij);
      const float BOA_ij = BO_ij - p.thb_cut;
      const float ijx = SFi(RM_SF_DVX, j, pij), ijy = SFi(RM_SF_DVY, j, pij), ijz = SFi(RM_SF_DVZ, j, pij);
      // angle at j: entry of the list of bond j->k (bond atom k) with partner i
      float theta_ijk, cos_t, sin_t;
      rm_theta(jkx, jky, jkz, r_jk, ijx, ijy, ijz, r_ij, theta_ijk, cos_t, sin_t);
      float c1i[3], c1j[3], c1k[3];   // d cos(theta_ijk): wrt k (list "di"), wrt j, wrt i (list "dk")
      rm_dcos(jkx, jky, jkz, r_jk, ijx, ijy, ijz, r_ij, c1i, c1j, c1k);
      const float sin_ijk = sin_t, cos_ijk = cos_t;
      float tan_ijk_i;
      if (sin_ijk >= 0.0f && sin_ijk <= MIN_SINE) tan_ijk_i = cos_ijk / MIN_SINE;
      else if (sin_ijk <= 0.0f && sin_ijk >= -MIN_SINE) tan_ijk_i = cos_ijk / -MIN_SINE;
      else tan_ijk_i = cos_ijk / sin_ijk;
      const float exp_tor2_ij = RMEXP(-p_tor2 * BOA_ij);
      const float exp_cot2_ij = RMEXP(-p_cot2 * ((BOA_ij - 1.5f) * (BOA_ij - 1.5f)));

      for (uint plk = 0; plk < nk; ++plk) {
        if (plk == pj) continue;
        const uint l = (uint)SIi(RM_SI_NBR, k, plk);
        if (!(plk > pj || (k < p.nlocal || l < p.nlocal))) continue;
        const float BO_kl = SFi(RM_SF_BO, k, plk);
        if (!(BO_kl > p.thb_cut)) continue;
        const uint tl = (uint)type[l];
        const uint fidx = ((ti * p.ntypes + tj) * p.ntypes + tk) * p.ntypes + tl;
        if (!(i != l && fb_has[fidx] != 0 && BO_ij * BO_jk * BO_kl > p.thb_cut)) continue;
        AIi(RM_AI_CTOR, j) += 1;
        device const float* fb = fb_f + fidx * RM_B_FB_F;
        const float V1 = fb[0], V2 = fb[1], V3 = fb[2], p_tor1 = fb[3], p_cot1 = fb[4];
        const float r_kl = SFi(RM_SF_D, k, plk);
        const float BOA_kl = BO_kl - p.thb_cut;
        const float klx = SFi(RM_SF_DVX, k, plk), kly = SFi(RM_SF_DVY, k, plk), klz = SFi(RM_SF_DVZ, k, plk);
        const float kjx = SFi(RM_SF_DVX, k, pj), kjy = SFi(RM_SF_DVY, k, pj), kjz = SFi(RM_SF_DVZ, k, pj), r_kj = SFi(RM_SF_D, k, pj);
        float theta_jkl, cos_t2, sin_t2;
        rm_theta(kjx, kjy, kjz, r_kj, klx, kly, klz, r_kl, theta_jkl, cos_t2, sin_t2);
        float c2i[3], c2j[3], c2k[3];   // d cos(theta_jkl): wrt j, wrt k, wrt l
        rm_dcos(kjx, kjy, kjz, r_kj, klx, kly, klz, r_kl, c2i, c2j, c2k);
        const float sin_jkl = sin_t2, cos_jkl = cos_t2;
        float tan_jkl_i;
        if (sin_jkl >= 0.0f && sin_jkl <= MIN_SINE) tan_jkl_i = cos_jkl / MIN_SINE;
        else if (sin_jkl <= 0.0f && sin_jkl >= -MIN_SINE) tan_jkl_i = cos_jkl / -MIN_SINE;
        else tan_jkl_i = cos_jkl / sin_jkl;
        const float lix = rm_dx(x, xlo, i, l, 0), liy = rm_dx(x, xlo, i, l, 1), liz = rm_dx(x, xlo, i, l, 2);
        const float r_li = sqrt(lix * lix + liy * liy + liz * liz);

        // Calculate_Omega (reaxff_torsion_angles.cpp:36-125)
        float s_ijk = sin_ijk, s_jkl = sin_jkl;
        if (s_ijk >= 0.0f && s_ijk <= MIN_SINE) s_ijk = MIN_SINE; else if (s_ijk <= 0.0f && s_ijk >= -MIN_SINE) s_ijk = -MIN_SINE;
        if (s_jkl >= 0.0f && s_jkl <= MIN_SINE) s_jkl = MIN_SINE; else if (s_jkl <= 0.0f && s_jkl >= -MIN_SINE) s_jkl = -MIN_SINE;
        const float unnorm_cos_omega = -rm_dot3(ijx, ijy, ijz, jkx, jky, jkz) * rm_dot3(jkx, jky, jkz, klx, kly, klz) + (r_jk * r_jk) * rm_dot3(ijx, ijy, ijz, klx, kly, klz);
        const float cx = jky * klz - jkz * kly, cy = jkz * klx - jkx * klz, cz = jkx * kly - jky * klx;
        const float unnorm_sin_omega = -r_jk * rm_dot3(ijx, ijy, ijz, cx, cy, cz);
        const float omega = atan2(unnorm_sin_omega, unnorm_cos_omega);
        const float htra = r_ij + cos_ijk * (r_kl * cos_jkl - r_jk);
        const float htrb = r_jk - r_ij * cos_ijk - r_kl * cos_jkl;
        const float htrc = r_kl + cos_jkl * (r_ij * cos_ijk - r_jk);
        const float hthd = r_ij * s_ijk * (r_jk - r_kl * cos_jkl);
        const float hthe = r_kl * s_jkl * (r_jk - r_ij * cos_ijk);
        const float hnra = r_kl * s_ijk * s_jkl;
        const float hnrc = r_ij * s_ijk * s_jkl;
        const float hnhd = r_ij * r_kl * cos_ijk * s_jkl;
        const float hnhe = r_ij * r_kl * s_ijk * cos_jkl;
        const float tel = (r_ij * r_ij) + (r_jk * r_jk) + (r_kl * r_kl) - (r_li * r_li) -
                          2.0f * (r_ij * r_jk * cos_ijk - r_ij * r_kl * cos_ijk * cos_jkl + r_jk * r_kl * cos_jkl);
        const float poem = 2.0f * r_ij * r_kl * s_ijk * s_jkl;
        float arg = tel / poem;
        if (arg > 1.0f) arg = 1.0f;
        if (arg < -1.0f) arg = -1.0f;
        float dco_i[3], dco_j[3], dco_k[3], dco_l[3];
        const float vij[3] = {ijx, ijy, ijz}, vjk[3] = {jkx, jky, jkz}, vkl[3] = {klx, kly, klz}, vli[3] = {lix, liy, liz};
        for (uint t = 0; t < 3; ++t) {
          dco_i[t] = ((htra - arg * hnra) / r_ij * vij[t] + -1.0f * vli[t]) + (-(hthd - arg * hnhd) / s_ijk) * c1k[t];
          dco_i[t] = 2.0f / poem * dco_i[t];
          dco_j[t] = (-(htra - arg * hnra) / r_ij * vij[t] + -htrb / r_jk * vjk[t]) + (-(hthd - arg * hnhd) / s_ijk) * c1j[t] + (-(hthe - arg * hnhe) / s_jkl) * c2i[t];
          dco_j[t] = 2.0f / poem * dco_j[t];
          dco_k[t] = (-(htrc - arg * hnrc) / r_kl * vkl[t] + htrb / r_jk * vjk[t]) + (-(hthd - arg * hnhd) / s_ijk) * c1i[t] + (-(hthe - arg * hnhe) / s_jkl) * c2j[t];
          dco_k[t] = 2.0f / poem * dco_k[t];
          dco_l[t] = ((htrc - arg * hnrc) / r_kl * vkl[t] + 1.0f * vli[t]) + (-(hthe - arg * hnhe) / s_jkl) * c2k[t];
          dco_l[t] = 2.0f / poem * dco_l[t];
        }
        const float cos_omega = cos(omega), cos2omega = cos(2.0f * omega), cos3omega = cos(3.0f * omega);

        // torsion energy
        const float u2 = 2.0f - BOpi_jk - f11_DjDk;
        const float exp_tor1 = RMEXP(p_tor1 * (u2 * u2));
        const float exp_tor2_kl = RMEXP(-p_tor2 * BOA_kl);
        const float exp_cot2_kl = RMEXP(-p_cot2 * ((BOA_kl - 1.5f) * (BOA_kl - 1.5f)));
        const float fn10 = (1.0f - exp_tor2_ij) * (1.0f - exp_tor2_jk) * (1.0f - exp_tor2_kl);
        const float CV = 0.5f * (V1 * (1.0f + cos_omega) + V2 * exp_tor1 * (1.0f - cos2omega) + V3 * (1.0f + cos3omega));
        e_tor += fn10 * sin_ijk * sin_jkl * CV;
        const float dfn11 = (-p_tor3 * exp_tor3_DjDk + (p_tor3 * exp_tor3_DjDk - p_tor4 * exp_tor4_DjDk) * (2.0f + exp_tor3_DjDk) * exp_tor34_inv) * exp_tor34_inv;
        const float CEtors1 = sin_ijk * sin_jkl * CV;
        const float CEtors2 = -fn10 * 2.0f * p_tor1 * V2 * exp_tor1 * u2 * (1.0f - (cos_omega * cos_omega)) * sin_ijk * sin_jkl;
        const float CEtors3 = CEtors2 * dfn11;
        const float CEtors4 = CEtors1 * p_tor2 * exp_tor2_ij * (1.0f - exp_tor2_jk) * (1.0f - exp_tor2_kl);
        const float CEtors5 = CEtors1 * p_tor2 * (1.0f - exp_tor2_ij) * exp_tor2_jk * (1.0f - exp_tor2_kl);
        const float CEtors6 = CEtors1 * p_tor2 * (1.0f - exp_tor2_ij) * (1.0f - exp_tor2_jk) * exp_tor2_kl;
        const float cmn = -fn10 * CV;
        const float CEtors7 = cmn * sin_jkl * tan_ijk_i;
        const float CEtors8 = cmn * sin_ijk * tan_jkl_i;
        const float CEtors9 = fn10 * sin_ijk * sin_jkl * (0.5f * V1 - 2.0f * V2 * exp_tor1 * cos_omega + 1.5f * V3 * (cos2omega + 2.0f * (cos_omega * cos_omega)));
        // 4-body conjugation
        const float fn12 = exp_cot2_ij * exp_cot2_jk * exp_cot2_kl;
        e_con += p_cot1 * fn12 * (1.0f + ((cos_omega * cos_omega) - 1.0f) * sin_ijk * sin_jkl);
        const float Cconj = -2.0f * fn12 * p_cot1 * p_cot2 * (1.0f + ((cos_omega * cos_omega) - 1.0f) * sin_ijk * sin_jkl);
        const float CEconj1 = Cconj * (BOA_ij - 1.5f), CEconj2 = Cconj * (BOA_jk - 1.5f), CEconj3 = Cconj * (BOA_kl - 1.5f);
        const float CEconj4 = -p_cot1 * fn12 * ((cos_omega * cos_omega) - 1.0f) * sin_jkl * tan_ijk_i;
        const float CEconj5 = -p_cot1 * fn12 * ((cos_omega * cos_omega) - 1.0f) * sin_ijk * tan_jkl_i;
        const float CEconj6 = 2.0f * p_cot1 * fn12 * cos_omega * sin_ijk * sin_jkl;

        // forces / coefficients
        SFi(RM_SF_CDBOPI, j, pk) += CEtors2;
        cdself += CEtors3;
        SFi(RM_SF_CDN, j, pk) += CEtors3;
        SFi(RM_SF_CDBO, j, pij) += (CEtors4 + CEconj1);
        SFi(RM_SF_CDBO, j, pk) += (CEtors5 + CEconj2);
        wf[p.o_tkl + (j * p.B + pk) * p.B + plk] += (CEtors6 + CEconj3);
        const float a7 = CEtors7 + CEconj4, a8 = CEtors8 + CEconj5, a9 = CEtors9 + CEconj6;
        SFi(RM_SF_FNX, j, pij) += a7 * c1k[0] + a9 * dco_i[0]; SFi(RM_SF_FNY, j, pij) += a7 * c1k[1] + a9 * dco_i[1]; SFi(RM_SF_FNZ, j, pij) += a7 * c1k[2] + a9 * dco_i[2];
        fsx += a7 * c1j[0] + a8 * c2i[0] + a9 * dco_j[0]; fsy += a7 * c1j[1] + a8 * c2i[1] + a9 * dco_j[1]; fsz += a7 * c1j[2] + a8 * c2i[2] + a9 * dco_j[2];
        SFi(RM_SF_FNX, j, pk) += a7 * c1i[0] + a8 * c2j[0] + a9 * dco_k[0]; SFi(RM_SF_FNY, j, pk) += a7 * c1i[1] + a8 * c2j[1] + a9 * dco_k[1]; SFi(RM_SF_FNZ, j, pk) += a7 * c1i[2] + a8 * c2j[2] + a9 * dco_k[2];
        const uint tf = p.o_tfl + ((j * p.B + pk) * p.B + plk) * 3;
        wf[tf] += a8 * c2k[0] + a9 * dco_l[0]; wf[tf + 1] += a8 * c2k[1] + a9 * dco_l[1]; wf[tf + 2] += a8 * c2k[2] + a9 * dco_l[2];
      }
    }
  }
  AFi(RM_AF_ETOR, j) = e_tor; AFi(RM_AF_ECON, j) = e_con;
  AFi(RM_AF_CDSELF, j) += cdself;
  AFi(RM_AF_FSX, j) += fsx; AFi(RM_AF_FSY, j) += fsy; AFi(RM_AF_FSZ, j) += fsz;
}

// ---- 7. hydrogen bonds ---------------------------------------------------------------------------------------------------------------------
// donor list: for every owned donor atom j (p_hbond == 1) all acceptors k (p_hbond == 2) within hbond_cut, in cell-scan order (deterministic).
kernel void rm_h_build(RM_B_ARGS) {
  if (a >= p.N) return;
  uint cnt = 0;
  const int ta = type[a];
  if (a < p.nlocal && ta >= 0 && sb_i[(uint)ta * RM_B_SB_I + 2] == 1 && p.hbond_cut > 0.0f) {
    const uint c = atom_cell[a];
    const uint cx = c % p.ncx, cy = (c / p.ncx) % p.ncy, cz = c / (p.ncx * p.ncy);
    const uint z0 = (cz > 0) ? cz - 1 : 0, z1 = (cz + 1 < p.ncz) ? cz + 1 : p.ncz - 1;
    const uint y0 = (cy > 0) ? cy - 1 : 0, y1 = (cy + 1 < p.ncy) ? cy + 1 : p.ncy - 1;
    const uint x0 = (cx > 0) ? cx - 1 : 0, x1 = (cx + 1 < p.ncx) ? cx + 1 : p.ncx - 1;
    for (uint gz = z0; gz <= z1; ++gz)
      for (uint gy = y0; gy <= y1; ++gy)
        for (uint gx = x0; gx <= x1; ++gx) {
          const uint cell = (gz * p.ncy + gy) * p.ncx + gx;
          for (uint q = cell_start[cell]; q < cell_start[cell + 1]; ++q) {
            const uint k = cell_items[q];
            if (k == a) continue;
            const int tk = type[k];
            if (tk < 0 || sb_i[(uint)tk * RM_B_SB_I + 2] != 2) continue;
            const float dx = rm_dx(x, xlo, k, a, 0), dy = rm_dx(x, xlo, k, a, 1), dz = rm_dx(x, xlo, k, a, 2);
            const float d = sqrt(dx * dx + dy * dy + dz * dz);
            if (!(d <= p.hbond_cut)) continue;
            if (cnt < p.H) wi[p.o_hi + a * p.H + cnt] = (int)k;
            ++cnt;
          }
        }
  }
  AIi(RM_AI_HNB, a) = (int)cnt;
}

kernel void rm_b_hbond(RM_B_ARGS) {
  const uint j = a;
  if (j >= p.nlocal) return;
  const int tjj = type[j];
  if (tjj < 0 || sb_i[(uint)tjj * RM_B_SB_I + 2] != 1 || !(p.hbond_cut > 0.0f)) return;
  const uint tj = (uint)tjj;
  const uint n = rm_nslots(wi, p, j);
  const uint nh = (uint)AIi(RM_AI_HNB, j) < p.H ? (uint)AIi(RM_AI_HNB, j) : p.H;
  float e_hb = 0.0f, fsx = 0.0f, fsy = 0.0f, fsz = 0.0f;
  for (uint h = 0; h < nh; ++h) {
    const uint k = (uint)wi[p.o_hi + j * p.H + h];
    const uint tk = (uint)type[k];
    const float jkx = rm_dx(x, xlo, k, j, 0), jky = rm_dx(x, xlo, k, j, 1), jkz = rm_dx(x, xlo, k, j, 2);
    const float r_jk = sqrt(jkx * jkx + jky * jky + jkz * jkz);
    float fkx = 0.0f, fky = 0.0f, fkz = 0.0f;
    for (uint pi = 0; pi < n; ++pi) {
      const uint i = (uint)SIi(RM_SI_NBR, j, pi);
      const uint ti = (uint)type[i];
      const float BO = SFi(RM_SF_BO, j, pi);
      if (!(sb_i[ti * RM_B_SB_I + 2] == 2 && BO >= 0.01f)) continue;
      if (tag[i] == tag[k]) continue;
      device const float* hb = hb_f + ((ti * p.ntypes + tj) * p.ntypes + tk) * RM_B_HB_F;
      const float r0_hb = hb[0], p_hb1 = hb[1], p_hb2 = hb[2], p_hb3 = hb[3];
      if (r0_hb <= 0.0f) continue;
      AIi(RM_AI_CHB, j) += 1;
      const float ux = SFi(RM_SF_DVX, j, pi), uy = SFi(RM_SF_DVY, j, pi), uz = SFi(RM_SF_DVZ, j, pi), du = SFi(RM_SF_D, j, pi);
      float theta, cos_theta, sth_hb;
      rm_theta(ux, uy, uz, du, jkx, jky, jkz, r_jk, theta, cos_theta, sth_hb);
      float dti[3], dtj[3], dtk[3];
      rm_dcos(ux, uy, uz, du, jkx, jky, jkz, r_jk, dti, dtj, dtk);
      const float sin_theta2 = sin(theta / 2.0f);
      float sin_xhz4 = sin_theta2 * sin_theta2;
      sin_xhz4 *= sin_xhz4;
      const float cos_xhz1 = (1.0f - cos_theta);
      const float exp_hb2 = RMEXP(-p_hb2 * BO);
      const float exp_hb3 = RMEXP(-p_hb3 * (r0_hb / r_jk + r_jk / r0_hb - 2.0f));
      const float e_hb_t = p_hb1 * (1.0f - exp_hb2) * exp_hb3 * sin_xhz4;
      e_hb += e_hb_t;
      const float CEhb1 = p_hb1 * p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
      const float CEhb2 = -p_hb1 / 2.0f * (1.0f - exp_hb2) * exp_hb3 * cos_xhz1;
      const float CEhb3 = -p_hb3 * (-r0_hb / (r_jk * r_jk) + 1.0f / r0_hb) * e_hb_t;
      SFi(RM_SF_CDBO, j, pi) += CEhb1;
      SFi(RM_SF_FNX, j, pi) += CEhb2 * dti[0]; SFi(RM_SF_FNY, j, pi) += CEhb2 * dti[1]; SFi(RM_SF_FNZ, j, pi) += CEhb2 * dti[2];
      fsx += CEhb2 * dtj[0] + (-CEhb3 / r_jk) * jkx; fsy += CEhb2 * dtj[1] + (-CEhb3 / r_jk) * jky; fsz += CEhb2 * dtj[2] + (-CEhb3 / r_jk) * jkz;
      fkx += CEhb2 * dtk[0] + (CEhb3 / r_jk) * jkx; fky += CEhb2 * dtk[1] + (CEhb3 / r_jk) * jky; fkz += CEhb2 * dtk[2] + (CEhb3 / r_jk) * jkz;
    }
    const uint o = p.o_hf + (j * p.H + h) * 3;
    wf[o] = fkx; wf[o + 1] = fky; wf[o + 2] = fkz;
  }
  AFi(RM_AF_EHB, j) = e_hb;
  AFi(RM_AF_FSX, j) += fsx; AFi(RM_AF_FSY, j) += fsy; AFi(RM_AF_FSZ, j) += fsz;
}

// acceptor atom a gathers the H-bond forces that owned donors computed for it
kernel void rm_b_hbgather(RM_B_ARGS) {
  if (a >= p.N) return;
  const int ta = type[a];
  if (ta < 0 || sb_i[(uint)ta * RM_B_SB_I + 2] != 2 || !(p.hbond_cut > 0.0f)) return;
  const uint c = atom_cell[a];
  const uint cx = c % p.ncx, cy = (c / p.ncx) % p.ncy, cz = c / (p.ncx * p.ncy);
  const uint z0 = (cz > 0) ? cz - 1 : 0, z1 = (cz + 1 < p.ncz) ? cz + 1 : p.ncz - 1;
  const uint y0 = (cy > 0) ? cy - 1 : 0, y1 = (cy + 1 < p.ncy) ? cy + 1 : p.ncy - 1;
  const uint x0 = (cx > 0) ? cx - 1 : 0, x1 = (cx + 1 < p.ncx) ? cx + 1 : p.ncx - 1;
  float gx = 0.0f, gy = 0.0f, gz = 0.0f;
  for (uint tz = z0; tz <= z1; ++tz)
    for (uint ty = y0; ty <= y1; ++ty)
      for (uint tx = x0; tx <= x1; ++tx) {
        const uint cell = (tz * p.ncy + ty) * p.ncx + tx;
        for (uint q = cell_start[cell]; q < cell_start[cell + 1]; ++q) {
          const uint j = cell_items[q];
          if (j >= p.nlocal || j == a) continue;
          const int tj = type[j];
          if (tj < 0 || sb_i[(uint)tj * RM_B_SB_I + 2] != 1) continue;
          const uint nh = (uint)AIi(RM_AI_HNB, j) < p.H ? (uint)AIi(RM_AI_HNB, j) : p.H;
          for (uint h = 0; h < nh; ++h)
            if ((uint)wi[p.o_hi + j * p.H + h] == a) {
              const uint o = p.o_hf + (j * p.H + h) * 3;
              gx += wf[o]; gy += wf[o + 1]; gz += wf[o + 2];
              break;
            }
        }
      }
  AFi(RM_AF_FHX, a) = gx; AFi(RM_AF_FHY, a) = gy; AFi(RM_AF_FHZ, a) = gz;
}

// ---- 8. CdDelta gather and the torsion's bond-order coefficient gather ------------------------------------------------------------------------
kernel void rm_b_cdgather(RM_B_ARGS) {
  if (a >= p.N) return;
  if (type[a] < 0) return;
  const uint n = rm_nslots(wi, p, a);
  float cd = AFi(RM_AF_CDSELF, a);
  for (uint s = 0; s < n; ++s) {
    const uint j = (uint)SIi(RM_SI_NBR, a, s);
    if (j < p.nlocal) cd += SFi(RM_SF_CDN, j, (uint)SIi(RM_SI_SYM, a, s));
  }
  AFi(RM_AF_CD, a) = cd;
  // torsions with central bond (j, a), j owned, add to Cdbo of the bond a -> l (scratch tkl, indexed [j][slot of a in j][slot of l in a])
  for (uint sj = 0; sj < n; ++sj) {
    const uint j = (uint)SIi(RM_SI_NBR, a, sj);
    if (j >= p.nlocal) continue;
    const uint pk = (uint)SIi(RM_SI_SYM, a, sj);
    for (uint plk = 0; plk < n; ++plk) SFi(RM_SF_CDBO, a, plk) += wf[p.o_tkl + (j * p.B + pk) * p.B + plk];
  }
}

// ---- 9. Add_dBond_to_Forces coefficients per undirected bond (stored at the lower-index end) (reaxff_bond_orders.cpp:36-108) --------------------
kernel void rm_b_dbond(RM_B_ARGS) {
  if (a >= p.N) return;
  if (type[a] < 0) return;
  const uint n = rm_nslots(wi, p, a);
  for (uint s = 0; s < n; ++s) {
    const uint j = (uint)SIi(RM_SI_NBR, a, s);
    if (!(a < j)) continue;
    const uint sy = (uint)SIi(RM_SI_SYM, a, s);
    float c = SFi(RM_SF_CDBO, a, s) + SFi(RM_SF_CDBO, j, sy);
    const float C1dbo = SFi(RM_SF_C1, a, s) * c, C2dbo = SFi(RM_SF_C2, a, s) * c, C3dbo = SFi(RM_SF_C3, a, s) * c;
    c = SFi(RM_SF_CDBOPI, a, s) + SFi(RM_SF_CDBOPI, j, sy);
    const float C1dbopi = SFi(RM_SF_C1P, a, s) * c, C2dbopi = SFi(RM_SF_C2P, a, s) * c, C3dbopi = SFi(RM_SF_C3P, a, s) * c, C4dbopi = SFi(RM_SF_C4P, a, s) * c;
    c = SFi(RM_SF_CDBOPI2, a, s) + SFi(RM_SF_CDBOPI2, j, sy);
    const float C1dbopi2 = SFi(RM_SF_C1Q, a, s) * c, C2dbopi2 = SFi(RM_SF_C2Q, a, s) * c, C3dbopi2 = SFi(RM_SF_C3Q, a, s) * c, C4dbopi2 = SFi(RM_SF_C4Q, a, s) * c;
    c = AFi(RM_AF_CD, a) + AFi(RM_AF_CD, j);
    const float C1dDelta = SFi(RM_SF_C1, a, s) * c, C2dDelta = SFi(RM_SF_C2, a, s) * c, C3dDelta = SFi(RM_SF_C3, a, s) * c;
    const float ci = C1dbo + C1dDelta + C2dbopi + C2dbopi2;
    const float cj = C2dbo + C2dDelta + C3dbopi + C3dbopi2;
    const float ck = C3dbo + C3dDelta + C4dbopi + C4dbopi2;
    SFi(RM_SF_TIX, a, s) = ci * SFi(RM_SF_DBX, a, s) + cj * AFi(RM_AF_DDPX, a) + C1dbopi * SFi(RM_SF_DPX, a, s) + C1dbopi2 * SFi(RM_SF_DQX, a, s);
    SFi(RM_SF_TIY, a, s) = ci * SFi(RM_SF_DBY, a, s) + cj * AFi(RM_AF_DDPY, a) + C1dbopi * SFi(RM_SF_DPY, a, s) + C1dbopi2 * SFi(RM_SF_DQY, a, s);
    SFi(RM_SF_TIZ, a, s) = ci * SFi(RM_SF_DBZ, a, s) + cj * AFi(RM_AF_DDPZ, a) + C1dbopi * SFi(RM_SF_DPZ, a, s) + C1dbopi2 * SFi(RM_SF_DQZ, a, s);
    SFi(RM_SF_TJX, a, s) = -ci * SFi(RM_SF_DBX, a, s) + ck * AFi(RM_AF_DDPX, j) - C1dbopi * SFi(RM_SF_DPX, a, s) - C1dbopi2 * SFi(RM_SF_DQX, a, s);
    SFi(RM_SF_TJY, a, s) = -ci * SFi(RM_SF_DBY, a, s) + ck * AFi(RM_AF_DDPY, j) - C1dbopi * SFi(RM_SF_DPY, a, s) - C1dbopi2 * SFi(RM_SF_DQY, a, s);
    SFi(RM_SF_TJZ, a, s) = -ci * SFi(RM_SF_DBZ, a, s) + ck * AFi(RM_AF_DDPZ, j) - C1dbopi * SFi(RM_SF_DPZ, a, s) - C1dbopi2 * SFi(RM_SF_DQZ, a, s);
    SFi(RM_SF_CKI, a, s) = -cj;
    SFi(RM_SF_CKJ, a, s) = -ck;
  }
}

// ---- 10. final gradient of every atom (owned and ghost), gathered in a fixed order -------------------------------------------------------------
kernel void rm_b_force(RM_B_ARGS) {
  if (a >= p.N) return;
  if (type[a] < 0) return;
  const uint n = rm_nslots(wi, p, a);
  float gx = AFi(RM_AF_FSX, a) + AFi(RM_AF_FHX, a), gy = AFi(RM_AF_FSY, a) + AFi(RM_AF_FHY, a), gz = AFi(RM_AF_FSZ, a) + AFi(RM_AF_FHZ, a);
  for (uint s = 0; s < n; ++s) {
    const uint m = (uint)SIi(RM_SI_NBR, a, s);
    const uint sy = (uint)SIi(RM_SI_SYM, a, s);          // slot of a in m's list
    // (a) contributions that owned centre atoms handed to their neighbor a
    if (m < p.nlocal) { gx += SFi(RM_SF_FNX, m, sy); gy += SFi(RM_SF_FNY, m, sy); gz += SFi(RM_SF_FNZ, m, sy); }
    // (b) Add_dBond direct terms of the bond a-m
    if (a < m) { gx += SFi(RM_SF_TIX, a, s); gy += SFi(RM_SF_TIY, a, s); gz += SFi(RM_SF_TIZ, a, s); }
    else { gx += SFi(RM_SF_TJX, m, sy); gy += SFi(RM_SF_TJY, m, sy); gz += SFi(RM_SF_TJZ, m, sy); }
    // (c) a is a bond partner k of m: every bond (m, y) adds ck * dBOp(m -> a)
    const uint nm = rm_nslots(wi, p, m);
    for (uint t = 0; t < nm; ++t) {
      const uint y = (uint)SIi(RM_SI_NBR, m, t);
      float ck;
      if (m < y) ck = SFi(RM_SF_CKI, m, t);
      else ck = SFi(RM_SF_CKJ, y, (uint)SIi(RM_SI_SYM, m, t));
      gx += ck * SFi(RM_SF_DBX, m, sy); gy += ck * SFi(RM_SF_DBY, m, sy); gz += ck * SFi(RM_SF_DBZ, m, sy);
    }
    // (d) a is the atom l of torsions with central bond (j, m), j owned: tfl[j][slot of m in j][slot of a in m]
    for (uint t = 0; t < nm; ++t) {
      const uint j = (uint)SIi(RM_SI_NBR, m, t);
      if (j >= p.nlocal) continue;
      const uint pk = (uint)SIi(RM_SI_SYM, m, t);        // slot of m in j's list
      const uint o = p.o_tfl + ((j * p.B + pk) * p.B + sy) * 3;
      gx += wf[o]; gy += wf[o + 1]; gz += wf[o + 2];
    }
  }
  AFi(RM_AF_GX, a) = gx; AFi(RM_AF_GY, a) = gy; AFi(RM_AF_GZ, a) = gz;
}
