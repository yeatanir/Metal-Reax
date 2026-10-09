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
  std::vector<std::int32_t> owner;   // per atom (nall): index of the owned atom a ghost images
  std::vector<float> shld;           // ntypes * ntypes: (gamma_i * gamma_j)^-1.5
  std::vector<float> eta_atom;       // per owned atom
  std::uint32_t ntypes = 0;
  float swa = 0, swb = 0;
};

// `lammps_type` has nall entries (1-based LAMMPS types); chi/eta/gamma are indexed by LAMMPS type (index 0 unused), as fix qeq/reaxff holds them.
// Throws SystemError when the taper radius exceeds the far-row cutoff of the owned rows (the rows would miss pairs).
QeqDeviceInput make_qeq_device_input(const NeighborCutoffs& cut, const AtomSet& atoms, const Box& box, const std::vector<int>& lammps_type,
                                     const double* eta, const double* gamma, int ntypes, double swa, double swb,
                                     const std::function<FarRowsF32(const DeviceListInput&)>& build_rows,
                                     const DeviceListInput* shared_list = nullptr,
                                     std::shared_ptr<const FarRowsF32> shared_rows = nullptr);

}  // namespace reaxmetal
