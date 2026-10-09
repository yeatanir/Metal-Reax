// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/qeq_device.hpp"

#include <cmath>

namespace reaxmetal {

QeqDeviceInput make_qeq_device_input(const NeighborCutoffs& cut, const AtomSet& atoms, const Box& box, const std::vector<int>& lammps_type,
                                     const double* eta, const double* gamma, int ntypes, double swa, double swb,
                                     const std::function<FarRowsF32(const DeviceListInput&)>& build_rows,
                                     const DeviceListInput* shared_list,
                                     std::shared_ptr<const FarRowsF32> shared_rows) {
  if (swb > cut.nonb) throw SystemError("fix qeq/reaxff/metal: the taper radius exceeds the force field's nonb_cut (the far rows would miss pairs)");
  const std::size_t N = atoms.nall();
  if (lammps_type.size() != N) throw SystemError("make_qeq_device_input: need one type per atom");
  QeqDeviceInput in;
  in.list = shared_list ? *shared_list : make_device_list_input(atoms, box, cut);
  in.rows = shared_rows ? shared_rows : std::make_shared<const FarRowsF32>(build_rows(in.list));
  in.column = build_column_index(*in.rows, in.list.nlocal);
  in.type.resize(N);
  in.owner.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    in.type[i] = lammps_type[i] - 1;
    in.owner[i] = static_cast<std::int32_t>(i < atoms.nlocal ? i : static_cast<std::size_t>(atoms.owner[i]));
  }
  in.ntypes = static_cast<std::uint32_t>(ntypes);
  in.shld.resize(static_cast<std::size_t>(ntypes) * static_cast<std::size_t>(ntypes));
  for (int a = 0; a < ntypes; ++a)
    for (int b = 0; b < ntypes; ++b) in.shld[static_cast<std::size_t>(a) * static_cast<std::size_t>(ntypes) + static_cast<std::size_t>(b)] = static_cast<float>(std::pow(gamma[a + 1] * gamma[b + 1], -1.5));
  in.eta_atom.resize(atoms.nlocal);
  for (std::size_t i = 0; i < atoms.nlocal; ++i) in.eta_atom[i] = static_cast<float>(eta[lammps_type[i]]);
  in.swa = static_cast<float>(swa);
  in.swb = static_cast<float>(swb);
  return in;
}

}  // namespace reaxmetal
