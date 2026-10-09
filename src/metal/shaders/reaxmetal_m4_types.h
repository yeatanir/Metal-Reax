// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Parameter blocks of the M4/M5 physics kernels (nonbonded first). Appended after reaxmetal_m3_types.h (which defines rm_u32/rm_f32) when
// the shader source is assembled; included as plain C++ by the host. 4-byte scalars only.
#ifndef REAXMETAL_M4_TYPES_H
#define REAXMETAL_M4_TYPES_H

#ifdef __METAL_VERSION__
#define RM_POW pow
#define RM_EXP exp
#define RM_LOG log
#endif

// per-pair parameter table layout: RM_NB_PAIR_FLOATS floats per ordered type pair (ti * ntypes + tj), in this order
// alpha, D, r_vdW, gamma_w, gamma, ecore, acore, rcore, lgcij, lgre
#define RM_NB_PAIR_FLOATS 10

// rm_nb_pairs: one thread per OWNED atom, loops over its far row (neighbor rows of M3)
struct RmNbParams {
  rm_u32 nlocal;
  rm_u32 cap;        // row capacity of the far rows
  rm_u32 ntypes;
  rm_u32 vdw_type;   // gp.vdw_type: 1 shielding, 2 inner wall, 3 both
  rm_u32 lg;         // lgvdw yes
  rm_f32 p_vdW1;     // gp[28]
  rm_f32 swa;        // taper lower radius (nonb_low)
  rm_f32 swb;        // taper upper radius == nonb_cut
};

// rm_nb_gather: one thread per atom (owned + ghost): deterministic gather of the pair gradients
struct RmNbGatherParams {
  rm_u32 nall;
  rm_u32 nlocal;
  rm_u32 cap;
};

// fix qeq/reaxff/metal: H assembly and the symmetric matvec (the CG itself stays on the host in double)
struct RmQeqParams {
  rm_u32 nlocal;
  rm_u32 cap;        // row capacity of the far rows
  rm_u32 ntypes;
  rm_f32 swa;        // fix taper radii (fix arguments), not the force field's
  rm_f32 swb;
};

#endif
