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
  if (!atoms.distributed) in.owner.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    in.type[i] = lammps_type[i] - 1;
    if (!atoms.distributed) in.owner[i] = static_cast<std::int32_t>(i < atoms.nlocal ? i : static_cast<std::size_t>(atoms.owner[i]));
  }
  in.ntypes = static_cast<std::uint32_t>(ntypes);
  auto split = [](double v, float& hi, float& lo) { hi = static_cast<float>(v); lo = static_cast<float>(v - static_cast<double>(hi)); };
  in.shld.resize(static_cast<std::size_t>(ntypes) * static_cast<std::size_t>(ntypes));
  in.shld_lo.resize(in.shld.size());
  for (int a = 0; a < ntypes; ++a)
    for (int b = 0; b < ntypes; ++b) {
      const std::size_t k = static_cast<std::size_t>(a) * static_cast<std::size_t>(ntypes) + static_cast<std::size_t>(b);
      split(std::pow(gamma[a + 1] * gamma[b + 1], -1.5), in.shld[k], in.shld_lo[k]);
    }
  in.eta_atom.resize(atoms.nlocal);
  in.eta_lo.resize(atoms.nlocal);
  for (std::size_t i = 0; i < atoms.nlocal; ++i) split(eta[lammps_type[i]], in.eta_atom[i], in.eta_lo[i]);
  in.swa = static_cast<float>(swa);
  in.swb = static_cast<float>(swb);
  float swa_hi_unused;
  split(swa, swa_hi_unused, in.swa_lo);
  split(swb - swa, in.d_hi, in.d_lo);
  split(14.4, in.c_hi, in.c_lo);
  return in;
}

}  // namespace reaxmetal
