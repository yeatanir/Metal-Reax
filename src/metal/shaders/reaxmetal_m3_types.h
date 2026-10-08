// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Parameter blocks shared by the host (C++/Objective-C++) and the Metal kernels of milestone M3.
// This file is PREPENDED to reaxmetal_m3.metal when the shader source is assembled (the runtime compiler cannot resolve
// local #includes), and is included as plain C++ by the host. Keep it to 4-byte scalars so the layout is identical on both
// sides, and keep it free of anything the other side's compiler would reject.
#ifndef REAXMETAL_M3_TYPES_H
#define REAXMETAL_M3_TYPES_H

#ifdef __METAL_VERSION__
typedef uint rm_u32;
typedef float rm_f32;
#else
typedef unsigned int rm_u32;   // 32 bits on every platform this project supports (asserted in the host code)
typedef float rm_f32;
#endif

// rm_far_rows: one thread per atom, fixed-capacity rows (ENGINE_SPEC 3, neighbor.hpp)
struct RmFarRowsParams {
  rm_u32 nall;
  rm_u32 nlocal;
  rm_u32 cap;          // row capacity; the true count is still stored when it exceeds cap
  rm_u32 ncx;          // cell grid dimensions (cells are numbered (cz * ncy + cy) * ncx + cx)
  rm_u32 ncy;
  rm_u32 ncz;
  rm_f32 rc2_owned;    // squared row cutoff of owned atoms (nonb_cut + margin)^2
  rm_f32 rc2_ghost;    // squared row cutoff of ghost atoms (bond_cut + margin)^2
};

// rm_partial_sums / rm_sum_partials: canonical fixed-order reduction (neighbor.hpp: fixed_order_partials_f32)
struct RmReduceParams {
  rm_u32 n;
  rm_u32 chunk;
};

struct RmSaxpyParams {
  rm_u32 n;
  rm_f32 a;
};

#endif
