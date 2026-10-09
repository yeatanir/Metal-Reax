// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/nonbonded_device.hpp"

#include <algorithm>
#include <functional>

#include "reaxmetal/terms_host.hpp"

namespace reaxmetal {

ColumnIndex build_column_index(const FarRowsF32& rows, std::uint32_t nlocal) {
  ColumnIndex ci;
  ci.start.assign(static_cast<std::size_t>(rows.nall) + 1, 0u);
  for (std::uint32_t i = 0; i < nlocal; ++i) {
    const std::uint32_t n = std::min(rows.count[i], rows.cap);
    for (std::uint32_t e = 0; e < n; ++e) ++ci.start[static_cast<std::size_t>(rows.nbr[static_cast<std::size_t>(i) * rows.cap + e]) + 1];
  }
  for (std::size_t k = 0; k < rows.nall; ++k) ci.start[k + 1] += ci.start[k];
  ci.items.assign(ci.start[rows.nall], 0u);
  std::vector<std::uint32_t> fill(ci.start.begin(), ci.start.end() - 1);
  for (std::uint32_t i = 0; i < nlocal; ++i) {
    const std::uint32_t n = std::min(rows.count[i], rows.cap);
    for (std::uint32_t e = 0; e < n; ++e) {
      const std::size_t j = static_cast<std::size_t>(rows.nbr[static_cast<std::size_t>(i) * rows.cap + e]);
      ci.items[fill[j]++] = i * rows.cap + e;
    }
  }
  return ci;
}

NonbondedDeviceInput make_nonbonded_device_input(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const Box& box,
                                                 const std::vector<double>& q_owned, const NonbondedOptions& opt,
                                                 const std::function<FarRowsF32(const DeviceListInput&)>& build_rows,
                                                 const DeviceListInput* shared_list,
                                                 std::shared_ptr<const FarRowsF32> shared_rows) {
  if (q_owned.size() != atoms.nlocal) throw SystemError("make_nonbonded_device_input: need one charge per owned atom");
  NonbondedDeviceInput in;
  in.list = shared_list ? *shared_list : make_device_list_input(atoms, box, cut);
  in.rows = shared_rows ? shared_rows : std::make_shared<const FarRowsF32>(build_rows(in.list));
  in.column = build_column_index(*in.rows, in.list.nlocal);
  const std::size_t N = atoms.nall();
  in.type.assign(atoms.type.begin(), atoms.type.end());
  in.tag.resize(N);
  in.q.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    const std::int64_t t = atoms.tag[i];
    if (t > 0x7FFFFFFF || t < -0x7FFFFFFF) throw SystemError("make_nonbonded_device_input: atom tag does not fit 32 bits");
    in.tag[i] = static_cast<std::int32_t>(t);
    in.q[i] = static_cast<float>(q_owned[static_cast<std::size_t>(atoms.owner[i])]);
  }
  const int nt = ff.num_types();
  in.ntypes = static_cast<std::uint32_t>(nt);
  in.pair_table.assign(static_cast<std::size_t>(nt) * static_cast<std::size_t>(nt) * 10, 0.0f);
  for (int a = 0; a < nt; ++a)
    for (int b = 0; b < nt; ++b) {
      const TwoBody& t = ff.pair(a, b);
      const double v[10] = {t.alpha, t.D, t.r_vdW, t.gamma_w, t.gamma, t.ecore, t.acore, t.rcore, t.lgcij, t.lgre};
      for (std::size_t k = 0; k < 10; ++k) in.pair_table[(static_cast<std::size_t>(a) * static_cast<std::size_t>(nt) + static_cast<std::size_t>(b)) * 10 + k] = static_cast<float>(v[k]);
    }
  in.vdw_type = static_cast<std::uint32_t>(ff.global().vdw_type);
  in.lg = opt.lgvdw ? 1u : 0u;
  in.p_vdW1 = static_cast<float>(ff.global().l[28]);
  in.swa = static_cast<float>(ff.file_control().nonb_low);
  in.swb = static_cast<float>(ff.file_control().nonb_cut);
  return in;
}

NonbondedResult finish_nonbonded(const ForceField& ff, const AtomSet& atoms, const std::vector<double>& q_owned, const NonbondedDeviceOutput& out) {
  NonbondedResult r;
  double ev = 0, ee = 0;
  for (std::size_t i = 0; i < atoms.nlocal; ++i) { ev += static_cast<double>(out.row_e[2 * i]); ee += static_cast<double>(out.row_e[2 * i + 1]); }
  double ep = 0;
  for (std::size_t i = 0; i < atoms.nlocal; ++i) {
    const int t = atoms.type[i];
    if (t < 0) continue;
    const SingleBody& s = ff.single(t);
    ep += terms::polarization<double>(s.chi, s.eta, q_owned[i]);
  }
  r.e[EnergyTerm::VdW] = ev;
  r.e[EnergyTerm::Coulomb] = ee;
  r.e[EnergyTerm::Polarization] = ep;
  r.grad.assign(out.grad.begin(), out.grad.end());
  return r;
}

}  // namespace reaxmetal
