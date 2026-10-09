// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Metal kernels of the M4/M5 physics: nonbonded (van der Waals, shielded Coulomb) with FROZEN charges (stock fix qeq/reaxff solves them).
// Appended after reaxmetal_m3.metal and after terms.hpp in the assembled shader source (terms.hpp supplies reaxmetal::terms, shared with the
// CPU-64 / CPU-32 code, ADR-005). Same rules as M3: no double, no atomics, no threadgroup communication; every result is a function of the
// inputs only. Owner-computes counting is exactly classify_nonbonded_entry (neighbor.hpp, ENGINE_SPEC 3.1).
//
// Data flow (no atomics): rm_nb_pairs writes one gradient contribution CE*dvec per far-row entry (zero when the entry is not counted);
// rm_nb_gather sums, for every atom k, -(own row entries, ascending) then +(entries of other rows that name k, ascending, from the host
// column index). Both orders are fixed, so the float result is identical on every run.

kernel void rm_nb_pairs(device const float* x [[buffer(0)]],
                        device const int* type [[buffer(1)]],
                        device const int* tag [[buffer(2)]],
                        device const float* q [[buffer(3)]],
                        device const float* pair_table [[buffer(4)]],
                        device const int* nbr [[buffer(5)]],
                        device const uint* count [[buffer(6)]],
                        device float* pf [[buffer(7)]],
                        device float* row_e [[buffer(8)]],
                        device const float* xlo [[buffer(9)]],
                        device uint* row_cnt [[buffer(10)]],   // counted pairs of the row (decision census)
                        constant RmNbParams& p [[buffer(11)]],
                        uint i [[thread_position_in_grid]]) {
  if (i >= p.nlocal) return;
  float ev = 0.0f;
  float ee = 0.0f;
  uint counted_pairs = 0;
  const int ti = type[i];
  if (ti >= 0) {
    const uint n = (count[i] < p.cap) ? count[i] : p.cap;
    for (uint e = 0; e < n; ++e) {
      const uint id = i * p.cap + e;
      pf[3 * id] = 0.0f; pf[3 * id + 1] = 0.0f; pf[3 * id + 2] = 0.0f;   // not-counted entries must read as zero by the gather
      const uint j = (uint)nbr[i * p.cap + e];
      const int tj = type[j];
      if (tj < 0) continue;
      const float dx = (x[3 * j] - x[3 * i]) + (xlo[3 * j] - xlo[3 * i]);   // hi/lo coordinates: error relative to the distance
      const float dy = (x[3 * j + 1] - x[3 * i + 1]) + (xlo[3 * j + 1] - xlo[3 * i + 1]);
      const float dz = (x[3 * j + 2] - x[3 * i + 2]) + (xlo[3 * j + 2] - xlo[3 * i + 2]);
      const float r = sqrt(dx * dx + dy * dy + dz * dz);
      if (!(r <= p.swb)) continue;
      // owner-computes rule (reaxff_nonbonded.cpp:104-117)
      bool counted = false;
      if (j < p.nlocal) {
        counted = true;
      } else if (tag[i] < tag[j]) {
        counted = true;
      } else if (tag[i] == tag[j]) {
        if (dz > 0.0001f) counted = true;
        else if (fabs(dz) < 0.0001f) {
          if (dy > 0.0001f) counted = true;
          else if (fabs(dy) < 0.0001f && dx > 0.0001f) counted = true;
        }
      }
      if (!counted) continue;
      ++counted_pairs;
      const uint t = ((uint)ti * p.ntypes + (uint)tj) * RM_NB_PAIR_FLOATS;
      reaxmetal::terms::NbPair<float> np;
      np.alpha = pair_table[t]; np.D = pair_table[t + 1]; np.r_vdW = pair_table[t + 2]; np.gamma_w = pair_table[t + 3];
      np.gamma = pair_table[t + 4]; np.ecore = pair_table[t + 5]; np.acore = pair_table[t + 6]; np.rcore = pair_table[t + 7];
      np.lgcij = pair_table[t + 8]; np.lgre = pair_table[t + 9];
      float Tap, dTap;
      reaxmetal::terms::taper_stable<float>(p.swa, p.swb, r, Tap, dTap);
      const reaxmetal::terms::NbResult<float> o =
          reaxmetal::terms::nonbonded_pair<float>(np, (int)p.vdw_type, p.lg != 0, p.p_vdW1, q[i], q[j], r, Tap, dTap);
      ev += o.e_vdW;
      ee += o.e_ele;
      pf[3 * id] = o.CE * dx;
      pf[3 * id + 1] = o.CE * dy;
      pf[3 * id + 2] = o.CE * dz;
    }
  }
  row_e[2 * i] = ev;
  row_e[2 * i + 1] = ee;
  row_cnt[i] = counted_pairs;
}

kernel void rm_nb_gather(device const float* pf [[buffer(0)]],
                         device const uint* count [[buffer(1)]],
                         device const uint* col_start [[buffer(2)]],
                         device const uint* col_items [[buffer(3)]],
                         device float* grad [[buffer(4)]],
                         constant RmNbGatherParams& p [[buffer(5)]],
                         uint k [[thread_position_in_grid]]) {
  if (k >= p.nall) return;
  float gx = 0.0f;
  float gy = 0.0f;
  float gz = 0.0f;
  if (k < p.nlocal) {
    const uint n = (count[k] < p.cap) ? count[k] : p.cap;
    for (uint e = 0; e < n; ++e) {
      const uint id = k * p.cap + e;
      gx -= pf[3 * id];
      gy -= pf[3 * id + 1];
      gz -= pf[3 * id + 2];
    }
  }
  for (uint t = col_start[k]; t < col_start[k + 1]; ++t) {
    const uint id = col_items[t];
    gx += pf[3 * id];
    gy += pf[3 * id + 1];
    gz += pf[3 * id + 2];
  }
  grad[3 * k] = gx;
  grad[3 * k + 1] = gy;
  grad[3 * k + 2] = gz;
}

// ---- double-single ("float-float") arithmetic: a value is hi + lo with |lo| <= ulp(hi)/2, giving ~46 bits. Used for the EEM matrix and the residual, where
// single precision limits the charges to ~1e-5 e. Error-free transformations after Dekker/Knuth; needs IEEE float arithmetic without reassociation and an
// explicit fma (the Metal library is compiled with fast math off, ADR-008).
struct DF { float hi; float lo; };
inline DF df_make(float h, float l) { DF r; r.hi = h; r.lo = l; return r; }
inline DF df_two_sum(float a, float b) { const float s = a + b; const float bb = s - a; return df_make(s, (a - (s - bb)) + (b - bb)); }
inline DF df_fast_two_sum(float a, float b) { const float s = a + b; return df_make(s, b - (s - a)); }
inline DF df_two_prod(float a, float b) { const float p = a * b; return df_make(p, fma(a, b, -p)); }
inline DF df_add(DF a, DF b) {
  DF s = df_two_sum(a.hi, b.hi);
  const DF t = df_two_sum(a.lo, b.lo);
  s.lo += t.hi;
  s = df_fast_two_sum(s.hi, s.lo);
  s.lo += t.lo;
  return df_fast_two_sum(s.hi, s.lo);
}
inline DF df_neg(DF a) { return df_make(-a.hi, -a.lo); }
inline DF df_sub(DF a, DF b) { return df_add(a, df_neg(b)); }
inline DF df_mul(DF a, DF b) { DF p = df_two_prod(a.hi, b.hi); p.lo += a.hi * b.lo + a.lo * b.hi; return df_fast_two_sum(p.hi, p.lo); }
inline DF df_mul_f(DF a, float b) { DF p = df_two_prod(a.hi, b); p.lo += a.lo * b; return df_fast_two_sum(p.hi, p.lo); }
inline DF df_div(DF a, DF b) {
  const float q1 = a.hi / b.hi;
  DF r = df_sub(a, df_mul_f(b, q1));
  const float q2 = r.hi / b.hi;
  r = df_sub(r, df_mul_f(b, q2));
  const float q3 = r.hi / b.hi;
  return df_add(df_fast_two_sum(q1, q2), df_make(q3, 0.0f));
}
inline DF df_sqrt(DF a) {
  if (!(a.hi > 0.0f)) return df_make(0.0f, 0.0f);
  const float x = 1.0f / sqrt(a.hi);
  const float ax = a.hi * x;
  const DF d = df_sub(a, df_two_prod(ax, ax));
  return df_add(df_make(ax, 0.0f), df_make(d.hi * (x * 0.5f), 0.0f));
}

// ---- fix qeq/reaxff/metal ---------------------------------------------------------------------------------------------------------------
// rm_qeq_h: H_ij for every entry of the OWNED far rows (zero beyond the taper radius): Tap(r) * 14.4 / cbrt(r^3 + shld[ti][tj]), all in double-single
// arithmetic from the hi/lo coordinates; stored as hv (hi) and hv_lo. The taper is the stable form in t = (r - swa) / (swb - swa) of terms.hpp.
kernel void rm_qeq_h(device const float* x [[buffer(0)]],
                     device const float* xlo [[buffer(1)]],
                     device const int* type [[buffer(2)]],
                     device const float* shld [[buffer(3)]],
                     device const float* shld_lo [[buffer(4)]],
                     device const int* nbr [[buffer(5)]],
                     device const uint* count [[buffer(6)]],
                     device float* hv [[buffer(7)]],
                     device float* hv_lo [[buffer(8)]],
                     constant RmQeqDfParams& p [[buffer(9)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= p.nlocal) return;
  const uint n = (count[i] < p.cap) ? count[i] : p.cap;
  const DF swa = df_make(p.swa_hi, p.swa_lo), dd = df_make(p.d_hi, p.d_lo), c144 = df_make(p.c_hi, p.c_lo);
  for (uint e = 0; e < n; ++e) {
    const uint j = (uint)nbr[i * p.cap + e];
    const DF dx = df_add(df_two_sum(x[3 * j], -x[3 * i]), df_two_sum(xlo[3 * j], -xlo[3 * i]));
    const DF dy = df_add(df_two_sum(x[3 * j + 1], -x[3 * i + 1]), df_two_sum(xlo[3 * j + 1], -xlo[3 * i + 1]));
    const DF dz = df_add(df_two_sum(x[3 * j + 2], -x[3 * i + 2]), df_two_sum(xlo[3 * j + 2], -xlo[3 * i + 2]));
    const DF r = df_sqrt(df_add(df_add(df_mul(dx, dx), df_mul(dy, dy)), df_mul(dz, dz)));
    DF v = df_make(0.0f, 0.0f);
    if (r.hi <= p.swb) {
      const DF t = df_div(df_sub(r, swa), dd);
      const DF t2 = df_mul(t, t), t3 = df_mul(t2, t), t4 = df_mul(t2, t2);
      // Tap = 1 - t^4 (35 - 84 t + 70 t^2 - 20 t^3)
      DF inner = df_sub(df_make(35.0f, 0.0f), df_mul_f(t, 84.0f));
      inner = df_add(inner, df_mul_f(t2, 70.0f));
      inner = df_sub(inner, df_mul_f(t3, 20.0f));
      const DF tap = df_sub(df_make(1.0f, 0.0f), df_mul(t4, inner));
      const uint k = (uint)type[i] * p.ntypes + (uint)type[j];
      const DF a = df_add(df_mul(df_mul(r, r), r), df_make(shld[k], shld_lo[k]));
      // cbrt(a): float start, one Newton step in double-single: y = y0 + (a - y0^3) / (3 y0^2)
      const float y0 = pow(a.hi, 1.0f / 3.0f);
      const DF y03 = df_mul_f(df_two_prod(y0, y0), y0);
      const DF err = df_sub(a, y03);
      const DF y = df_fast_two_sum(y0, err.hi / (3.0f * y0 * y0));
      v = df_div(df_mul(tap, c144), y);
    }
    hv[i * p.cap + e] = v.hi;
    hv_lo[i * p.cap + e] = v.lo;
  }
}

// rm_cg_res_df: r = b - (diag(eta) + H) x for a batch of two vectors with x, b, eta and H in double-single arithmetic (the accumulation too); r is returned as a float
// (its own relative error 6e-8 does not matter: r is the right-hand side of the correction solve). Ghost entries through the owner map (single rank).
kernel void rm_cg_res_df(device const float* hv [[buffer(0)]],
                        device const float* hv_lo [[buffer(1)]],
                        device const int* nbr [[buffer(2)]],
                        device const uint* count [[buffer(3)]],
                        device const int* owner [[buffer(4)]],
                        device const uint* col_start [[buffer(5)]],
                        device const uint* col_items [[buffer(6)]],
                        device const float* x_hi [[buffer(7)]],
                        device const float* x_lo [[buffer(8)]],
                        device const float* eta_hi [[buffer(9)]],
                        device const float* eta_lo [[buffer(10)]],
                        device const float* b_hi [[buffer(11)]],
                        device const float* b_lo [[buffer(12)]],
                        device float* rout [[buffer(13)]],
                        constant RmCgParams& p [[buffer(14)]],
                        uint g [[thread_position_in_grid]]) {
  if (g >= 2 * p.nlocal) return;
  const uint sys = g / p.nlocal, i = g % p.nlocal;
  const uint base = sys * p.nlocal;
  DF y = df_mul(df_make(eta_hi[i], eta_lo[i]), df_make(x_hi[base + i], x_lo[base + i]));
  const uint n = (count[i] < p.cap) ? count[i] : p.cap;
  for (uint e = 0; e < n; ++e) {
    const uint id = i * p.cap + e;
    const uint o = (uint)owner[nbr[id]];
    y = df_add(y, df_mul(df_make(hv[id], hv_lo[id]), df_make(x_hi[base + o], x_lo[base + o])));
  }
  for (uint t = col_start[i]; t < col_start[i + 1]; ++t) {
    const uint id = col_items[t];
    const uint row = id / p.cap;
    y = df_add(y, df_mul(df_make(hv[id], hv_lo[id]), df_make(x_hi[base + row], x_lo[base + row])));
  }
  const DF r = df_sub(df_make(b_hi[g], b_lo[g]), y);
  rout[g] = r.hi + r.lo;
}

// rm_qeq_mv: y = (diag(eta) + H) x for the owned atoms of a serial run. H is stored as upper rows; the symmetric product gathers
//   row part  : sum over the own row of hv * x[j]   (x holds owned AND ghost entries, as LAMMPS' forward communication keeps them equal to their
//               owners': an owned atom sees all its neighbors, so its row sum is complete on every rank and needs no reverse communication)
//   column part: sum over the owned rows that name i       (owned-owned pairs, from the host column index)
// both in ascending entry order, so the float result is the same on every run.
kernel void rm_qeq_mv(device const float* hv [[buffer(0)]],
                      device const int* nbr [[buffer(1)]],
                      device const uint* count [[buffer(2)]],
                      device const uint* col_start [[buffer(3)]],
                      device const uint* col_items [[buffer(4)]],
                      device const float* xv [[buffer(5)]],
                      device const float* eta_atom [[buffer(6)]],
                      device float* yv [[buffer(7)]],
                      constant RmQeqParams& p [[buffer(8)]],
                      uint i [[thread_position_in_grid]]) {
  if (i >= p.nlocal) return;
  float y = eta_atom[i] * xv[i];
  const uint n = (count[i] < p.cap) ? count[i] : p.cap;
  for (uint e = 0; e < n; ++e) y += hv[i * p.cap + e] * xv[nbr[i * p.cap + e]];
  for (uint t = col_start[i]; t < col_start[i + 1]; ++t) {
    const uint id = col_items[t];
    y += hv[id] * xv[id / p.cap];
  }
  yv[i] = y;
}

// ---- fix qeq/reaxff/metal, resident preconditioned CG (both systems in one batch) ---------------------------------------------------------
// Same product as rm_qeq_mv but for a batch of two vectors; ghost entries are read through the owner map (single rank).
kernel void rm_cg_mv(device const float* hv [[buffer(0)]],
                     device const int* nbr [[buffer(1)]],
                     device const uint* count [[buffer(2)]],
                     device const int* owner [[buffer(3)]],
                     device const uint* col_start [[buffer(4)]],
                     device const uint* col_items [[buffer(5)]],
                     device const float* xv [[buffer(6)]],
                     device const float* eta_atom [[buffer(7)]],
                     device float* yv [[buffer(8)]],
                     constant RmCgParams& p [[buffer(9)]],
                     uint g [[thread_position_in_grid]]) {
  if (g >= 2 * p.nlocal) return;
  const uint sys = g / p.nlocal, i = g % p.nlocal;
  device const float* x = xv + sys * p.nlocal;
  float y = eta_atom[i] * x[i];
  const uint n = (count[i] < p.cap) ? count[i] : p.cap;
  for (uint e = 0; e < n; ++e) y += hv[i * p.cap + e] * x[owner[nbr[i * p.cap + e]]];
  for (uint t = col_start[i]; t < col_start[i + 1]; ++t) {
    const uint id = col_items[t];
    y += hv[id] * x[id / p.cap];
  }
  yv[g] = y;
}

// r = b - q ; d = r * hinv    (start of the CG)
kernel void rm_cg_start(device const float* b [[buffer(0)]], device const float* q [[buffer(1)]], device const float* hinv [[buffer(2)]],
                        device float* r [[buffer(3)]], device float* d [[buffer(4)]], constant RmCgParams& p [[buffer(5)]], uint g [[thread_position_in_grid]]) {
  if (g >= 2 * p.nlocal) return;
  const float rr = b[g] - q[g];
  r[g] = rr;
  d[g] = rr * hinv[g % p.nlocal];
}

// partial dot products over fixed chunks: out[sys * nchunk + c] = sum_{e in chunk c} u[e] * v[e]
kernel void rm_cg_dot(device const float* u [[buffer(0)]], device const float* v [[buffer(1)]], device float* part [[buffer(2)]],
                      constant RmCgParams& p [[buffer(3)]], uint g [[thread_position_in_grid]]) {
  if (g >= 2 * p.nchunk) return;
  const uint sys = g / p.nchunk, c = g % p.nchunk;
  const uint lo = c * p.chunk, hi = (lo + p.chunk < p.nlocal) ? lo + p.chunk : p.nlocal;
  float s = 0.0f;
  for (uint e = lo; e < hi; ++e) s += u[sys * p.nlocal + e] * v[sys * p.nlocal + e];
  part[g] = s;
}

// one thread per system. mode 0: sig = sum(part) from (r, d) at the start, bnorm from the second partial array; mode 1: alpha = sig / sum(part) (d.q);
// mode 2: sig_new = sum(part) (r.p), beta, convergence test.   Fixed summation order.
kernel void rm_cg_scalar(device const float* part [[buffer(0)]], device const float* part_b [[buffer(1)]], device float* sc [[buffer(2)]],
                         constant RmCgParams& p [[buffer(3)]], uint sys [[thread_position_in_grid]]) {
  const uint mode = p.mode;
  if (sys >= 2) return;
  device float* s = sc + sys * RM_CG_SCALARS;   // 0 sig, 1 sig_old, 2 alpha, 3 beta, 4 bnorm, 5 rel, 6 done, 7 performed
  float total = 0.0f;
  for (uint c = 0; c < p.nchunk; ++c) total += part[sys * p.nchunk + c];
  if (mode == 0) {
    float bb = 0.0f;
    for (uint c = 0; c < p.nchunk; ++c) bb += part_b[sys * p.nchunk + c];
    s[0] = total; s[1] = total; s[2] = 0.0f; s[3] = 0.0f; s[4] = sqrt(bb); s[6] = 0.0f; s[7] = 0.0f;
    s[5] = (s[4] > 0.0f) ? sqrt(total > 0.0f ? total : 0.0f) / s[4] : 0.0f;
    if (!(s[5] > p.tol)) s[6] = 1.0f;                                  // already converged (the stock loop does not start)
    return;
  }
  if (mode == 1) {
    if (s[6] != 0.0f) { s[2] = 0.0f; return; }
    if (!(total > 0.0f) || !isfinite(total)) { s[6] = 2.0f; s[2] = 0.0f; return; }   // breakdown: d.q <= 0 (should not happen for a positive definite matrix)
    s[2] = s[0] / total;
    s[1] = s[0];
    return;
  }
  if (s[6] != 0.0f) { s[3] = 0.0f; return; }
  s[7] += 1.0f;
  s[0] = total;
  s[3] = (s[1] > 0.0f) ? total / s[1] : 0.0f;
  s[5] = (s[4] > 0.0f) ? sqrt(total > 0.0f ? total : 0.0f) / s[4] : 0.0f;
  if (!isfinite(total)) s[6] = 2.0f;
  else if (!(s[5] > p.tol)) s[6] = 1.0f;
  else if (s[7] + 1.0f >= (float)p.maxiter) s[6] = 3.0f;                // the stock loop runs i = 1 .. maxiter-1
}

// x += alpha d ; r -= alpha q ; p = r * hinv
kernel void rm_cg_update(device float* x [[buffer(0)]], device float* r [[buffer(1)]], device const float* d [[buffer(2)]], device const float* q [[buffer(3)]],
                         device const float* hinv [[buffer(4)]], device float* pv [[buffer(5)]], device const float* sc [[buffer(6)]],
                         constant RmCgParams& p [[buffer(7)]], uint g [[thread_position_in_grid]]) {
  if (g >= 2 * p.nlocal) return;
  const float alpha = sc[(g / p.nlocal) * RM_CG_SCALARS + 2];
  x[g] += alpha * d[g];
  const float rr = r[g] - alpha * q[g];
  r[g] = rr;
  pv[g] = rr * hinv[g % p.nlocal];
}

// d = p + beta d
kernel void rm_cg_dir(device float* d [[buffer(0)]], device const float* pv [[buffer(1)]], device const float* sc [[buffer(2)]], constant RmCgParams& p [[buffer(3)]],
                      uint g [[thread_position_in_grid]]) {
  if (g >= 2 * p.nlocal) return;
  d[g] = pv[g] + sc[(g / p.nlocal) * RM_CG_SCALARS + 3] * d[g];
}
