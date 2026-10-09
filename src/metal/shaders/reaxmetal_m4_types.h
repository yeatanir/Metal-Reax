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

// double-single parameters of the EEM matrix (every value is a (hi, lo) float pair, hi + lo being the double precision value rounded to ~2^-46)
struct RmQeqDfParams {
  rm_u32 nlocal;
  rm_u32 cap;
  rm_u32 ntypes;
  rm_f32 swb;                  // cut-off for the decision r <= swb (float: the taper vanishes to 4th order there)
  rm_f32 swa_hi, swa_lo;
  rm_f32 d_hi, d_lo;           // swb - swa
  rm_f32 c_hi, c_lo;           // 14.4 (EV_TO_KCAL_PER_MOL of the stock fix)
};

// fix qeq/reaxff/metal 'resident': the preconditioned CG of both EEM systems (s: H s = -chi, t: H t = -1) on the device, one thread per (system, atom)
struct RmCgParams {
  rm_u32 nlocal;     // atoms per system; the batch holds 2 * nlocal entries (system 0 then system 1)
  rm_u32 cap;        // row capacity of the far rows
  rm_u32 chunk;      // elements per partial sum
  rm_u32 nchunk;     // ceil(nlocal / chunk)
  rm_u32 maxiter;    // stock 'imax': at most maxiter - 1 iterations are performed
  rm_f32 tol;        // relative residual to reach (sqrt(r.Minv.r) / |b|)
  rm_u32 mode;       // rm_cg_scalar: 0 start, 1 alpha, 2 beta and convergence
};
// scalar state per system (floats): sig, sig_old, alpha, beta, bnorm, rel, done (0 running, 1 converged, 2 breakdown, 3 iteration limit), performed iterations
#define RM_CG_SCALARS 8

#endif
