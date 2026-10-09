// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Metal Shading Language kernels of milestone M3 (bring-up, neighbor rows, fixed-order reductions).
//
// STATUS: written; executed only as C++ through tests/metal_shim (CPU emulation). NOT compiled by the Metal compiler and
// NOT run on a GPU until tools/mac/step1_bringup.sh has been run on the Apple machine (docs/VALIDATION.md MET-1).
//
// Rules this file follows (checked by tests/test_metal_source.cpp):
//   * no `double` (Apple GPUs have no FP64), no atomics (deterministic results, ADR-007 / FORCE-2), no local #include;
//   * reductions add in a fixed, documented order; there is no threadgroup communication, so every result is a function of
//     the inputs only and the same on every run and every GPU;
//   * the host compiles with fast-math OFF; rm_math_probe reports whether that took effect.
// reaxmetal_m3_types.h is prepended to this text by the host before compilation.
#include <metal_stdlib>
using namespace metal;

// ---- MET-1: bring-up kernels ---------------------------------------------------------------------------------------
kernel void rm_saxpy(device const float* x [[buffer(0)]],
                     device float* y [[buffer(1)]],
                     constant RmSaxpyParams& p [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  y[i] = p.a * x[i] + y[i];
}

// out[0] = 1 if NaN is detected (fails to detect only under fast-math), out[1] = a*b + c with a = b = 1 + 2^-13, c = -(1 + 2^-12):
// 0 when the multiply is rounded before the add, 2^-26 when the compiler contracted it into an FMA. Informational.
kernel void rm_math_probe(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
  if (i != 0) return;
  const float nan_in = in[0];
  out[0] = (nan_in != nan_in) ? 1.0f : 0.0f;
  const float a = in[1];
  const float b = in[2];
  const float c = in[3];
  out[1] = a * b + c;
}

// ---- NBR-1: far neighbor rows ---------------------------------------------------------------------------------------
// Row i lists the atoms j > i with |x_j - x_i|^2 <= row cutoff^2, scanning the 27 surrounding cells in (z, y, x) order and
// the atoms of a cell in ascending index order: the order is a function of the inputs only. Positions are float, relative to
// the box lower corner; the host enlarged the cutoffs by a margin that covers the float rounding (neighbor.hpp).
kernel void rm_far_rows(device const float* x [[buffer(0)]],
                        device const uint* atom_cell [[buffer(1)]],
                        device const uint* cell_start [[buffer(2)]],
                        device const uint* cell_items [[buffer(3)]],
                        device int* nbr [[buffer(4)]],
                        device float* r2 [[buffer(5)]],
                        device uint* count [[buffer(6)]],
                        constant RmFarRowsParams& p [[buffer(7)]],
                        uint i [[thread_position_in_grid]]) {
  if (i >= p.nall) return;
  const float rc2 = (i < p.nlocal) ? p.rc2_owned : p.rc2_ghost;
  const uint c = atom_cell[i];
  const uint cx = c % p.ncx;
  const uint cy = (c / p.ncx) % p.ncy;
  const uint cz = c / (p.ncx * p.ncy);
  const float xi = x[3 * i];
  const float yi = x[3 * i + 1];
  const float zi = x[3 * i + 2];
  const uint z0 = (cz > 0) ? cz - 1 : 0;
  const uint z1 = (cz + 1 < p.ncz) ? cz + 1 : p.ncz - 1;
  const uint y0 = (cy > 0) ? cy - 1 : 0;
  const uint y1 = (cy + 1 < p.ncy) ? cy + 1 : p.ncy - 1;
  const uint x0 = (cx > 0) ? cx - 1 : 0;
  const uint x1 = (cx + 1 < p.ncx) ? cx + 1 : p.ncx - 1;
  uint n = 0;
  for (uint gz = z0; gz <= z1; ++gz) {
    for (uint gy = y0; gy <= y1; ++gy) {
      for (uint gx = x0; gx <= x1; ++gx) {
        const uint cell = (gz * p.ncy + gy) * p.ncx + gx;
        const uint end = cell_start[cell + 1];
        for (uint q = cell_start[cell]; q < end; ++q) {
          const uint j = cell_items[q];
          if (j <= i) continue;
          const float dx = x[3 * j] - xi;
          const float dy = x[3 * j + 1] - yi;
          const float dz = x[3 * j + 2] - zi;
          const float d2 = dx * dx + dy * dy + dz * dz;
          if (d2 <= rc2) {
            if (n < p.cap) {
              nbr[i * p.cap + n] = (int)j;
              r2[i * p.cap + n] = d2;
            }
            ++n;
          }
        }
      }
    }
  }
  count[i] = n;
  // unused tail of the row: fixed bytes (the host and the tests rely on entries >= min(count, cap) reading -1 / 0)
  for (uint e = (n < p.cap) ? n : p.cap; e < p.cap; ++e) {
    nbr[i * p.cap + e] = -1;
    r2[i * p.cap + e] = 0.0f;
  }
}

// ---- FORCE-2: deterministic reductions -------------------------------------------------------------------------------
// Chunk c sums elements [c*chunk, min(n, (c+1)*chunk)) sequentially in float. A second launch with ONE thread then adds the
// partials sequentially in float. This is exactly fixed_order_partials_f32 / fixed_order_sum_f32 on the host, so CPU and
// GPU agree bit for bit. (A tree reduction would be faster; speed is M8, reproducibility is now.)
kernel void rm_partial_sums(device const float* v [[buffer(0)]],
                            device float* part [[buffer(1)]],
                            constant RmReduceParams& p [[buffer(2)]],
                            uint c [[thread_position_in_grid]]) {
  const uint nchunk = (p.n / p.chunk) + ((p.n % p.chunk) != 0 ? 1 : 0);
  if (c >= nchunk) return;
  const uint start = c * p.chunk;
  const uint len = ((p.n - start) < p.chunk) ? (p.n - start) : p.chunk;
  float s = 0.0f;
  for (uint e = 0; e < len; ++e) s += v[start + e];
  part[c] = s;
}

kernel void rm_sum_partials(device const float* part [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant RmReduceParams& p [[buffer(2)]],
                            uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  const uint nchunk = (p.n / p.chunk) + ((p.n % p.chunk) != 0 ? 1 : 0);
  float s = 0.0f;
  for (uint c = 0; c < nchunk; ++c) s += part[c];
  out[0] = s;
}
