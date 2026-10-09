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
                        constant RmNbParams& p [[buffer(10)]],
                        uint i [[thread_position_in_grid]]) {
  if (i >= p.nlocal) return;
  float ev = 0.0f;
  float ee = 0.0f;
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

// ---- fix qeq/reaxff/metal ---------------------------------------------------------------------------------------------------------------
// rm_qeq_h: H_ij for every entry of the OWNED far rows (zero beyond the taper radius): Tap(r) * 14.4 / cbrt(r^3 + shld[ti][tj]).
kernel void rm_qeq_h(device const float* x [[buffer(0)]],
                     device const float* xlo [[buffer(1)]],
                     device const int* type [[buffer(2)]],
                     device const float* shld [[buffer(3)]],
                     device const int* nbr [[buffer(4)]],
                     device const uint* count [[buffer(5)]],
                     device float* hv [[buffer(6)]],
                     constant RmQeqParams& p [[buffer(7)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= p.nlocal) return;
  const uint n = (count[i] < p.cap) ? count[i] : p.cap;
  for (uint e = 0; e < n; ++e) {
    const uint j = (uint)nbr[i * p.cap + e];
    const float dx = (x[3 * j] - x[3 * i]) + (xlo[3 * j] - xlo[3 * i]);
    const float dy = (x[3 * j + 1] - x[3 * i + 1]) + (xlo[3 * j + 1] - xlo[3 * i + 1]);
    const float dz = (x[3 * j + 2] - x[3 * i + 2]) + (xlo[3 * j + 2] - xlo[3 * i + 2]);
    const float r = sqrt(dx * dx + dy * dy + dz * dz);
    float v = 0.0f;
    if (r <= p.swb) {
      float Tap, dTap;
      reaxmetal::terms::taper_stable<float>(p.swa, p.swb, r, Tap, dTap);
      v = Tap * 14.4f / pow(r * r * r + shld[(uint)type[i] * p.ntypes + (uint)type[j]], 1.0f / 3.0f);
    }
    hv[i * p.cap + e] = v;
  }
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
