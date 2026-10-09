// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// Host side of the device nonbonded path (FP32 on the GPU, FP64 bookkeeping on the host): packing of the kernel inputs, the column index
// that makes the force gather atomics-free, and the fixed-order double-precision finish. Backend-independent: the Metal context runs the
// kernels (metal_backend.hpp), the CPU emulation in tests runs the same kernels as C++.
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/neighbor.hpp"
#include "reaxmetal/nonbonded.hpp"
#include "reaxmetal/system.hpp"

namespace reaxmetal {

// CSR by neighbor atom over the entries of the OWNED rows of `rows` (entry id = row * cap + slot, ascending): the entries that name atom k
// as their j. Deterministic (counting sort by atom, stable in entry id).
struct ColumnIndex {
  std::vector<std::uint32_t> start;   // nall + 1
  std::vector<std::uint32_t> items;
};
ColumnIndex build_column_index(const FarRowsF32& rows, std::uint32_t nlocal);

struct NonbondedDeviceInput {
  DeviceListInput list;                // float positions + grid (M3)
  std::shared_ptr<const FarRowsF32> rows;   // grown far rows built from `list` (shared with the other consumers of the same positions)
  ColumnIndex column;
  std::vector<std::int32_t> type, tag; // per atom (nall)
  std::vector<float> q;                // per atom (nall), ghosts carry their owner's charge
  std::vector<float> pair_table;       // ntypes * ntypes * 10 floats (RM_NB_PAIR_FLOATS)
  std::uint32_t ntypes = 0, vdw_type = 0, lg = 0;
  float p_vdW1 = 0, swa = 0, swb = 0;
};

struct NonbondedDeviceOutput {
  std::vector<float> grad;             // 3 * nall
  std::vector<float> row_e;            // 2 * nlocal: per owned row {e_vdW, e_ele} (float partial sums, fixed order)
};

// `rows` must already be the grown far rows of `list` (Context::far_rows or the emulated launch).
NonbondedDeviceInput make_nonbonded_device_input(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const Box& box,
                                                 const std::vector<double>& q_owned, const NonbondedOptions& opt,
                                                 const std::function<FarRowsF32(const DeviceListInput&)>& build_rows,
                                                 const DeviceListInput* shared_list = nullptr,
                                                 std::shared_ptr<const FarRowsF32> shared_rows = nullptr);   // with shared_rows, build_rows is not called

// FP64 finish: energies = fixed-order double sums of the row partials; polarization on the host in double; gradient widened to double.
NonbondedResult finish_nonbonded(const ForceField& ff, const AtomSet& atoms, const std::vector<double>& q_owned, const NonbondedDeviceOutput& out);

}  // namespace reaxmetal
