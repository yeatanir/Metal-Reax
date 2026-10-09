// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// Device input of fix qeq/reaxff/metal: the EEM matrix H (taper + shielded Coulomb) over the owned far rows and its symmetric matvec on the GPU.
// The conjugate-gradient iteration stays on the host in double (as in the stock fix); only H assembly and y = (diag(eta) + H) x run in FP32.
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "reaxmetal/nonbonded_device.hpp"

namespace reaxmetal {

struct QeqDeviceInput {
  DeviceListInput list;
  std::shared_ptr<const FarRowsF32> rows;
  ColumnIndex column;
  std::vector<std::int32_t> type;    // per atom (nall), LAMMPS type - 1
  std::vector<std::int32_t> owner;   // per atom (nall): the owned atom a ghost images; EMPTY on a multi-rank host (needed only by the resident solve)
  std::vector<float> shld, shld_lo;  // ntypes * ntypes: (gamma_i * gamma_j)^-1.5 as a double-single pair (hi, lo)
  std::vector<float> eta_atom, eta_lo;   // per owned atom: eta as a double-single pair
  std::uint32_t ntypes = 0;
  float swa = 0, swb = 0;
  float swa_lo = 0, d_hi = 0, d_lo = 0, c_hi = 0, c_lo = 0;   // double-single taper constants: swa = swa + swa_lo, swb - swa = d_hi + d_lo, 14.4 = c_hi + c_lo
};

// `lammps_type` has nall entries (1-based LAMMPS types); chi/eta/gamma are indexed by LAMMPS type (index 0 unused), as fix qeq/reaxff holds them.
// Throws SystemError when the taper radius exceeds the far-row cutoff of the owned rows (the rows would miss pairs).
QeqDeviceInput make_qeq_device_input(const NeighborCutoffs& cut, const AtomSet& atoms, const Box& box, const std::vector<int>& lammps_type,
                                     const double* eta, const double* gamma, int ntypes, double swa, double swb,
                                     const std::function<FarRowsF32(const DeviceListInput&)>& build_rows,
                                     const DeviceListInput* shared_list = nullptr,
                                     std::shared_ptr<const FarRowsF32> shared_rows = nullptr);

}  // namespace reaxmetal
