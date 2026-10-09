// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Layout of the bonded-term work arrays shared by the host and the M6 kernels (appended after reaxmetal_m4_types.h). Two big buffers carry all
// state: `wf` (float) and `wi` (int), addressed through the accessor macros of reaxmetal_m6.metal. Slot arrays have one entry per
// (atom, bond slot) = N*B entries per field; atom arrays N entries per field. 4-byte scalars only.
#ifndef REAXMETAL_M6_TYPES_H
#define REAXMETAL_M6_TYPES_H

// ---- per-slot float fields (index = field * N*B + atom * B + slot); slot s of atom a is the bond a -> nbr(a, s), slots sorted by ascending neighbor index
enum RmSlotF {
  RM_SF_D = 0, RM_SF_DVX, RM_SF_DVY, RM_SF_DVZ,                 // distance and dvec = x_nbr - x_a
  RM_SF_BOP, RM_SF_BOPS, RM_SF_BOPPI, RM_SF_BOPPI2,             // uncorrected, offset-reduced: BO', BO'_s, BO'_pi, BO'_pi2
  RM_SF_DBX, RM_SF_DBY, RM_SF_DBZ,                              // dBOp
  RM_SF_DPX, RM_SF_DPY, RM_SF_DPZ,                              // dln_BOp_pi
  RM_SF_DQX, RM_SF_DQY, RM_SF_DQZ,                              // dln_BOp_pi2
  RM_SF_BO, RM_SF_BOS, RM_SF_BOPI, RM_SF_BOPI2,                 // corrected
  RM_SF_C1, RM_SF_C2, RM_SF_C3,                                 // C1dbo C2dbo C3dbo
  RM_SF_C1P, RM_SF_C2P, RM_SF_C3P, RM_SF_C4P,                   // C*dbopi
  RM_SF_C1Q, RM_SF_C2Q, RM_SF_C3Q, RM_SF_C4Q,                   // C*dbopi2
  RM_SF_CDBO, RM_SF_CDBOPI, RM_SF_CDBOPI2,                      // accumulated coefficients of this directed bond
  RM_SF_CDN,                                                    // CdDelta contribution this slot hands to its NEIGHBOR atom
  RM_SF_FNX, RM_SF_FNY, RM_SF_FNZ,                              // force (gradient) contribution this slot hands to its NEIGHBOR atom
  RM_SF_TIX, RM_SF_TIY, RM_SF_TIZ, RM_SF_TJX, RM_SF_TJY, RM_SF_TJZ, RM_SF_CKI, RM_SF_CKJ,   // Add_dBond_to_Forces terms (bond stored at the lower-index end)
  RM_SF_COUNT
};
// ---- per-atom float fields (index = o_atom + field * N + atom)
enum RmAtomF {
  RM_AF_DELTAP = 0, RM_AF_DELTAP_BOC, RM_AF_DDPX, RM_AF_DDPY, RM_AF_DDPZ,
  RM_AF_TOTBO, RM_AF_DELTA, RM_AF_DELTA_BOC, RM_AF_DELTA_VAL, RM_AF_VLPEX, RM_AF_NLP, RM_AF_DELTA_LP, RM_AF_DDELTA_LP, RM_AF_DELTA_LP_TEMP,
  RM_AF_CDSELF, RM_AF_CD,                                       // CdDelta: own part, final value
  RM_AF_FSX, RM_AF_FSY, RM_AF_FSZ,                              // forces on the centre atom of valence / torsion / H-bond terms
  RM_AF_FHX, RM_AF_FHY, RM_AF_FHZ,                              // H-bond force on this atom as acceptor (gathered)
  RM_AF_GX, RM_AF_GY, RM_AF_GZ,                                 // final gradient dE/dx of all bonded terms
  RM_AF_EBOND, RM_AF_ELP, RM_AF_EOV, RM_AF_EUN, RM_AF_EANG, RM_AF_EPEN, RM_AF_ECOA, RM_AF_ETOR, RM_AF_ECON, RM_AF_EHB,
  RM_AF_COUNT
};
// ---- per-slot / per-atom int fields (wi)
enum RmSlotI { RM_SI_NBR = 0, RM_SI_SYM, RM_SI_COUNT };
enum RmAtomI { RM_AI_NB = 0, RM_AI_HNB, RM_AI_CTHB, RM_AI_CTOR, RM_AI_CHB, RM_AI_CSBO, RM_AI_CLP, RM_AI_COUNT };   // CTHB/CTOR/CHB: decision census (angle sets, torsions, H-bonds)

#define RM_B_SB_F 13   // floats per type: valency valency_e valency_boc valency_val nlp_opt r_s r_pi r_pi_pi p_lp2 p_ovun2 p_ovun5 p_val3 p_val5
#define RM_B_SB_I 3    // ints per type: heavy_atom_terms c2_species p_hbond
#define RM_B_TB_F 20   // floats per ordered type pair: p_bo1..6 r_s r_p r_pp p_boc3 p_boc4 p_boc5 ovc v13cor p_be1 p_be2 De_s De_p De_pp p_ovun1
#define RM_B_THB_F 7   // floats per three-body set: theta_00 p_val1 p_val2 p_val4 p_val7 p_pen1 p_coa1
#define RM_B_FB_F 5    // floats per four-body entry: V1 V2 V3 p_tor1 p_cot1
#define RM_B_HB_F 4    // floats per hydrogen-bond entry: r0_hb p_hb1 p_hb2 p_hb3

struct RmBParams {
  rm_u32 N, nlocal, B, H, NB, ntypes;
  rm_u32 ncx, ncy, ncz;
  rm_u32 o_atom, o_iatom, o_tkl, o_tfl, o_hi, o_hf;
  rm_u32 enobonds;
  rm_f32 bond_cut, bo_cut, thb_cut, thb_cutsq, hbond_cut;
};

#endif
