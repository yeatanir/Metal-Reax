// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// M3 kernels, EMULATED: src/metal/shaders/reaxmetal_m3.metal is compiled as C++ through tests/metal_shim and executed one
// "thread" at a time on the CPU. This verifies the kernels' logic and the host-side contracts around them (row growth, row
// verification, reduction order). It says nothing about the Metal compiler or the GPU: those are verified on the Apple machine
// (docs/VALIDATION.md MET-1..MET-4) and until then the Metal status is "written", not "executed".
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "reaxmetal/metal_backend.hpp"
#include "m3_systems.hpp"
#include "reaxmetal/neighbor.hpp"
#include "test_util.hpp"

#include "reaxmetal/bonded.hpp"
#include "reaxmetal/bonded_device.hpp"
#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/nonbonded.hpp"
#include "reaxmetal/nonbonded_device.hpp"
#include "reaxmetal/terms_host.hpp"

#include "metal_shim.hpp"
#include "reaxmetal_m3_types.h"
#include "reaxmetal_m4_types.h"
#include "reaxmetal_m6_types.h"
#include "reaxmetal_m3.metal"
#include "reaxmetal_m4.metal"
#include "reaxmetal_m6.metal"
#include "metal_shim_end.hpp"

using namespace reaxmetal;

namespace {
struct Rng {
  std::mt19937_64 g;
  explicit Rng(std::uint64_t s) : g(s) {}
  double u() { return static_cast<double>(g() >> 11) * (1.0 / 9007199254740992.0); }
};

FarRowsF32 emulate_far_rows(const DeviceListInput& in, std::uint32_t cap) {
  FarRowsF32 rows;
  rows.nall = in.nall;
  rows.cap = cap;
  rows.count.assign(in.nall, 0);
  rows.nbr.assign(static_cast<std::size_t>(in.nall) * cap, -1);
  rows.r2.assign(static_cast<std::size_t>(in.nall) * cap, 0.0f);
  RmFarRowsParams p{};
  p.nall = in.nall; p.nlocal = in.nlocal; p.cap = cap;
  p.ncx = in.grid.ncell[0]; p.ncy = in.grid.ncell[1]; p.ncz = in.grid.ncell[2];
  p.rc2_owned = in.rc2_owned; p.rc2_ghost = in.rc2_ghost;
  for (std::uint32_t i = 0; i < in.nall; ++i)
    rm_far_rows(in.x.data(), in.grid.atom_cell.data(), in.grid.cell_start.data(), in.grid.cell_items.data(), rows.nbr.data(), rows.r2.data(), rows.count.data(), p, i);
  return rows;
}

AtomSet make_system(const Box& box, std::size_t n, Rng& r, double spread, double shell) {
  std::vector<double> x;
  std::vector<int> type(n, 0);
  std::vector<std::int64_t> tag;
  for (std::size_t i = 0; i < n; ++i) {
    const Vec3 c = box.to_cartesian({r.u() * spread, r.u() * spread, r.u() * spread});
    x.insert(x.end(), c.begin(), c.end());
    tag.push_back(static_cast<std::int64_t>(i) + 1);
  }
  ExpandOptions opt;
  opt.shell = shell;
  return expand_images(box, x, type, tag, opt);
}
}  // namespace

static std::string strip_comments(const std::string& s) {
  std::string out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s.compare(i, 2, "//") == 0) {
      while (i < s.size() && s[i] != '\n') ++i;
      out += '\n';
    } else if (s.compare(i, 2, "/*") == 0) {
      const std::size_t e = s.find("*/", i + 2);
      i = (e == std::string::npos) ? s.size() : e + 1;
    } else {
      out += s[i];
    }
  }
  return out;
}

static void test_shader_source_lints() {
  const std::string full = mtl::shader_source();
  const std::string src = strip_comments(full);
  RM_CHECK(!src.empty());
  RM_CHECK_MSG(src.find("double") == std::string::npos, "MSL has no double (Apple GPUs have no FP64)");
  RM_CHECK_MSG(src.find("atomic") == std::string::npos, "no atomics in the validated path (ADR-007)");
  RM_CHECK_MSG(src.find("std::") == std::string::npos, "no C++ standard library in MSL");
  RM_CHECK_MSG(src.find("#include \"") == std::string::npos, "the runtime compiler cannot resolve local includes");
  RM_CHECK_MSG(src.find("printf") == std::string::npos, "no printf");
  RM_CHECK(src.find("#include <metal_stdlib>") != std::string::npos);
  RM_CHECK(src.find("struct RmFarRowsParams") != std::string::npos);        // types header was prepended
  RM_CHECK(src.find("struct RmFarRowsParams") < src.find("kernel void rm_far_rows"));
  for (const char* k : {"rm_saxpy", "rm_math_probe", "rm_far_rows", "rm_partial_sums", "rm_sum_partials", "rm_nb_pairs", "rm_nb_gather", "rm_b_build", "rm_h_build", "rm_b_prime", "rm_b_correct", "rm_b_atom", "rm_b_valence", "rm_b_torsion", "rm_b_hbond", "rm_b_hbgather", "rm_b_cdgather", "rm_b_dbond", "rm_b_force"}) {
    RM_CHECK_MSG(src.find(std::string("kernel void ") + k) != std::string::npos, k);
    const auto names = mtl::kernel_names();
    RM_CHECK_MSG(std::find(names.begin(), names.end(), std::string(k)) != names.end(), k);
  }
  // the assembled source is exactly: M3 types, M4 types, M6 types, M3 kernels, terms.hpp, M4 kernels, M6 kernels
  const std::string dir = std::string(REAXMETAL_SOURCE_DIR);
  const std::string t3 = rmtest::read_file(dir + "/src/metal/shaders/reaxmetal_m3_types.h");
  const std::string t4 = rmtest::read_file(dir + "/src/metal/shaders/reaxmetal_m4_types.h");
  const std::string k3 = rmtest::read_file(dir + "/src/metal/shaders/reaxmetal_m3.metal");
  const std::string tm = rmtest::read_file(dir + "/include/reaxmetal/terms.hpp");
  const std::string k4 = rmtest::read_file(dir + "/src/metal/shaders/reaxmetal_m4.metal");
  const std::string t6 = rmtest::read_file(dir + "/src/metal/shaders/reaxmetal_m6_types.h");
  const std::string k6 = rmtest::read_file(dir + "/src/metal/shaders/reaxmetal_m6.metal");
  RM_CHECK(full == t3 + "\n" + t4 + "\n" + t6 + "\n" + k3 + "\n" + tm + "\n" + k4 + "\n" + k6);
  RM_CHECK_MSG(src.find("RM_POW") != std::string::npos && src.find("#define RM_POW pow") != std::string::npos, "math macros must reach MSL");
  // struct layout the host relies on
  static_assert(sizeof(rm_u32) == 4 && sizeof(rm_f32) == 4, "scalar widths");
  static_assert(sizeof(RmFarRowsParams) == 32 && sizeof(RmReduceParams) == 8 && sizeof(RmSaxpyParams) == 8, "parameter block layout");
}

static void test_saxpy_and_probe() {
  std::vector<float> x(1000), y(1000), want(1000);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>(i % 17) - 8.0f;
    y[i] = static_cast<float>(i % 5);
    want[i] = 2.5f * x[i] + y[i];   // exact in float for these values
  }
  RmSaxpyParams p{static_cast<std::uint32_t>(x.size()), 2.5f};
  std::vector<float> got = y;
  for (std::uint32_t i = 0; i < 1010; ++i) rm_saxpy(x.data(), got.data(), p, i);   // 10 threads past the end must be ignored
  RM_CHECK(got == want);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float a = 1.0f + std::ldexp(1.0f, -13), c = -(1.0f + std::ldexp(1.0f, -12));
  const std::vector<float> in{nan, a, a, c};
  std::vector<float> out(2, -1.0f);
  rm_math_probe(in.data(), out.data(), 1);   // thread 1 does nothing
  RM_CHECK(out[0] == -1.0f);
  rm_math_probe(in.data(), out.data(), 0);
  RM_CHECK(out[0] == 1.0f);
  RM_CHECK_MSG(out[1] == 0.0f, "CPU emulation must round the product before the add (strict FP flags)");
}

static void test_far_rows_vs_cpu() {
  Rng r(31337);
  const Box boxes[] = {
      Box::orthogonal({0, 0, 0}, {9, 9, 9}, {true, true, true}),
      Box::from_lammps({0, 0, 0}, {7.134, 7.0, 7.1}, {1.2, 0.8, 1.0}, {true, true, true}),
      Box::orthogonal({-12, -12, -12}, {12, 12, 12}, {false, false, false}),
      Box::orthogonal({0, 0, 0}, {1.3, 30, 30}, {true, false, false}),
  };
  const NeighborCutoffs cuts[] = {{10.0, 5.0, 7.5}, {6.0, 3.5, 4.0}};
  std::size_t total_entries = 0, total_band = 0;
  for (const Box& b : boxes)
    for (const auto& cut : cuts) {
      const AtomSet a = make_system(b, b.periodic[1] ? 40 : 25, r, 0.7, cut.required_shell());
      a.validate(b);
      const FarList cpu = build_far_list(a, cut);
      const DeviceListInput in = make_device_list_input(a, b, cut, 1e-3);
      FarRowsF32 rows = emulate_far_rows(in, 4096);
      const RowComparison cmp = compare_rows_to_far_list(rows, cpu, a, cut, in.list_margin);
      RM_CHECK_MSG(cmp.ok(), "emulated kernel rows violate the superset/bounded contract");
      RM_CHECK(cmp.cpu_entries == cpu.entries());
      {   // the double precision far list rebuilt from the device rows is IDENTICAL to the host-built one
        const FarList from_rows = far_list_from_rows(a, cut, rows);
        RM_CHECK(from_rows.row_start == cpu.row_start && from_rows.nbr == cpu.nbr && from_rows.dist == cpu.dist && from_rows.dvec == cpu.dvec);
      }
      total_entries += cmp.device_entries;
      total_band += cmp.band_extra;
      // determinism: identical inputs -> identical bytes
      const FarRowsF32 again = emulate_far_rows(in, 4096);
      RM_CHECK(again.nbr == rows.nbr && again.count == rows.count && again.r2 == rows.r2);
    }
  RM_CHECK(total_entries > 20000);
  std::printf("  emulated rows: %zu entries compared, %zu legal band pairs\n", total_entries, total_band);

  // the contract check itself must be able to fail
  {
    const Box b = Box::orthogonal({0, 0, 0}, {9, 9, 9}, {true, true, true});
    const NeighborCutoffs cut{6.0, 3.5, 4.0};
    const AtomSet a = make_system(b, 30, r, 1.0, cut.required_shell());
    const FarList cpu = build_far_list(a, cut);
    const DeviceListInput in = make_device_list_input(a, b, cut, 1e-3);
    FarRowsF32 rows = emulate_far_rows(in, 4096);
    RM_CHECK(compare_rows_to_far_list(rows, cpu, a, cut, in.list_margin).ok());
    FarRowsF32 lost = rows;                                     // drop one entry
    std::size_t row = 0;
    while (lost.count[row] == 0) ++row;
    --lost.count[row];
    RM_CHECK(compare_rows_to_far_list(lost, cpu, a, cut, in.list_margin).missing >= 1);
    FarRowsF32 far = rows;                                      // add an entry far outside the cutoff
    std::size_t row2 = 0;
    while (row2 + 1 < far.count.size() && far.count[row2] >= 4096) ++row2;
    far.nbr[row2 * far.cap + far.count[row2]] = static_cast<std::int32_t>(a.nall() - 1);
    far.r2[row2 * far.cap + far.count[row2]] = 1.0f;
    ++far.count[row2];
    const RowComparison fc = compare_rows_to_far_list(far, cpu, a, cut, in.list_margin);
    RM_CHECK(!fc.ok());
    FarRowsF32 badr2 = rows;
    badr2.r2[row * badr2.cap] += 1.0f;
    RM_CHECK(compare_rows_to_far_list(badr2, cpu, a, cut, in.list_margin).bad_r2 >= 1);
    FarRowsF32 dup = rows;
    std::size_t row3 = 0;
    while (dup.count[row3] < 2) ++row3;
    dup.nbr[row3 * dup.cap + 1] = dup.nbr[row3 * dup.cap];
    RM_CHECK(compare_rows_to_far_list(dup, cpu, a, cut, in.list_margin).structural >= 1);
  }
}

static void test_shared_systems() {
  // the same geometries the Apple-machine tool uses (tools/m3_systems.hpp), through the emulated kernel
  std::size_t n = 0;
  for (const auto& sys : m3::systems(true)) {
    const FarList cpu = build_far_list(sys.atoms, sys.cut);
    const DeviceListInput in = make_device_list_input(sys.atoms, sys.box, sys.cut, 1e-3);
    unsigned launches = 0;
    const FarRowsF32 rows = build_far_rows_with_growth([&](std::uint32_t cap) { return emulate_far_rows(in, cap); }, 16, in.nall, 4, &launches);
    const RowComparison c = compare_rows_to_far_list(rows, cpu, sys.atoms, sys.cut, in.list_margin);
    RM_CHECK_MSG(c.ok(), sys.name);
    RM_CHECK_MSG(c.cpu_entries > 0, sys.name);
    std::printf("  %-32s nall %6zu  cpu entries %9zu  device %9zu  band %2zu  launches %u\n", sys.name.c_str(), sys.atoms.nall(), c.cpu_entries, c.device_entries, c.band_extra, launches);
    ++n;
  }
  RM_CHECK(n == 7);
}

static void test_growth() {
  Rng r(5);
  const Box b = Box::orthogonal({0, 0, 0}, {9, 9, 9}, {true, true, true});
  const NeighborCutoffs cut{6.0, 3.5, 4.0};
  const AtomSet a = make_system(b, 60, r, 1.0, cut.required_shell());
  const FarList cpu = build_far_list(a, cut);
  const DeviceListInput in = make_device_list_input(a, b, cut, 1e-3);
  unsigned launches = 0;
  std::vector<std::uint32_t> caps;
  const FarRowsF32 rows = build_far_rows_with_growth([&](std::uint32_t cap) { caps.push_back(cap); return emulate_far_rows(in, cap); }, 4, in.nall, 4, &launches);
  RM_CHECK(launches == 2 && caps.size() == 2 && caps[0] == 4 && caps[1] >= rows.max_count() && caps[1] % 8 == 0);
  RM_CHECK(compare_rows_to_far_list(rows, cpu, a, cut, in.list_margin).ok());
  // a large enough first guess needs one launch
  const FarRowsF32 one = build_far_rows_with_growth([&](std::uint32_t cap) { return emulate_far_rows(in, cap); }, 4096, in.nall, 4, &launches);
  RM_CHECK(launches == 1 && one.cap == 4096);
  // an overflowing kernel that never stops growing is reported
  RM_EXPECT_THROW(build_far_rows_with_growth([&](std::uint32_t cap) { FarRowsF32 f = emulate_far_rows(in, cap); f.count[0] = cap + 1; return f; }, 8, in.nall, 3, nullptr), SystemError);
  RM_EXPECT_THROW(build_far_rows_with_growth([&](std::uint32_t cap) { return emulate_far_rows(in, cap); }, 0xFFFFFFF0u, in.nall, 3, nullptr), SystemError);
  // truncated rows still carry the true count, never more than cap entries are written
  const FarRowsF32 trunc = emulate_far_rows(in, 3);
  RM_CHECK(trunc.max_count() > 3);
  for (std::uint32_t i = 0; i < trunc.nall; ++i)
    for (std::uint32_t k = std::min<std::uint32_t>(trunc.count[i], 3); k < 3; ++k) RM_CHECK(trunc.nbr[i * 3 + k] == -1);
}

static void test_reduction_kernels() {
  Rng r(77);
  for (std::uint32_t n : {0u, 1u, 2u, 255u, 256u, 257u, 1000u, 4097u, 100000u})
    for (std::uint32_t chunk : {1u, 7u, 64u, 256u, 5000u}) {
      std::vector<float> v(n);
      for (auto& e : v) e = static_cast<float>(r.u() - 0.5) * 1.0e4f;
      const std::uint32_t nchunk = n == 0 ? 0 : (n + chunk - 1) / chunk;
      std::vector<float> part(nchunk > 0 ? nchunk : 1, -7.0f), out(1, -7.0f);
      RmReduceParams p{n, chunk};
      for (std::uint32_t c = 0; c < nchunk + 3; ++c) rm_partial_sums(v.data(), part.data(), p, c);   // extra threads are ignored
      rm_sum_partials(part.data(), out.data(), p, 1);                                               // only thread 0 acts
      RM_CHECK(out[0] == -7.0f);
      rm_sum_partials(part.data(), out.data(), p, 0);
      const std::vector<float> want = fixed_order_partials_f32(v, chunk);
      RM_CHECK(want.size() == nchunk);
      for (std::uint32_t c = 0; c < nchunk; ++c) RM_CHECK(part[c] == want[c]);
      RM_CHECK(out[0] == fixed_order_sum_f32(v, chunk));
    }
}

// Nonbonded kernels emulated vs the CPU-64 reference engine (needs the bundled force field; skipped without REAXMETAL_FFIELD_DIR)
static void emulate_nonbonded(const NonbondedDeviceInput& in, NonbondedDeviceOutput& out) {
  const std::uint32_t nall = in.list.nall, nlocal = in.list.nlocal, cap = in.rows->cap;
  std::vector<float> pf(static_cast<std::size_t>(nlocal) * cap * 3, 0.0f);
  out.grad.assign(3 * static_cast<std::size_t>(nall), 0.0f);
  out.row_e.assign(2 * static_cast<std::size_t>(nlocal), 0.0f);
  out.row_count.assign(nlocal, 0);
  RmNbParams p{};
  p.nlocal = nlocal; p.cap = cap; p.ntypes = in.ntypes; p.vdw_type = in.vdw_type; p.lg = in.lg; p.p_vdW1 = in.p_vdW1; p.swa = in.swa; p.swb = in.swb;
  for (std::uint32_t i = 0; i < nlocal; ++i)
    rm_nb_pairs(in.list.x.data(), in.type.data(), in.tag.data(), in.q.data(), in.pair_table.data(), in.rows->nbr.data(), in.rows->count.data(), pf.data(), out.row_e.data(), in.list.x_lo.data(), reinterpret_cast<std::uint32_t*>(out.row_count.data()), p, i);
  RmNbGatherParams g{nall, nlocal, cap};
  for (std::uint32_t k = 0; k < nall; ++k)
    rm_nb_gather(pf.data(), in.rows->count.data(), in.column.start.data(), in.column.items.data(), out.grad.data(), g, k);
}

static void test_nonbonded_vs_cpu64() {
#ifdef REAXMETAL_FFIELD_DIR
  const ForceField ff = read_force_field_file(std::string(REAXMETAL_FFIELD_DIR) + "/ffield.reax.cho");
  const int tC = ff.match_element("C").at(0), tH = ff.match_element("H").at(0), tO = ff.match_element("O").at(0);
  Rng r(2468);
  // three cases: large periodic cell, non-periodic cluster, and a periodic cell smaller than the cutoff (self-image pairs, tie-break rule)
  for (const int mode : {0, 1, 2}) {
    const bool periodic = mode != 1;
    const Box box = mode == 0 ? Box::orthogonal({0, 0, 0}, {11.0, 11.5, 12.0}, {true, true, true})
                              : mode == 1 ? Box::orthogonal({0, 0, 0}, {11.0, 11.5, 12.0}, {false, false, false}) : Box::orthogonal({0, 0, 0}, {6.3, 6.1, 6.7}, {true, true, true});
    const std::size_t n = mode == 2 ? 24 : 70;
    std::vector<double> x;
    std::vector<int> type;
    std::vector<std::int64_t> tag;
    std::vector<double> q;
    for (std::size_t i = 0; i < n; ++i) {
      const Vec3 c = box.to_cartesian({r.u(), r.u(), r.u()});
      x.insert(x.end(), c.begin(), c.end());
      type.push_back(i % 3 == 0 ? tC : (i % 3 == 1 ? tH : tO));
      tag.push_back(static_cast<std::int64_t>(i) + 1);
      q.push_back(0.6 * r.u() - 0.3);
    }
    NeighborCutoffs cut;
    cut.nonb = ff.file_control().nonb_cut;
    ExpandOptions eo; eo.shell = cut.required_shell();
    const AtomSet a = expand_images(box, x, type, tag, eo);
    const FarList f = build_far_list(a, cut);
    const NonbondedResult ref = compute_nonbonded_core(ff, cut, a, f, q);
    NonbondedDeviceOutput dev;
    const NonbondedDeviceInput in = make_nonbonded_device_input(ff, cut, a, box, q, {}, [&](const DeviceListInput& l) {
      return build_far_rows_with_growth([&](std::uint32_t cap) { return emulate_far_rows(l, cap); }, 16, l.nall);
    });
    emulate_nonbonded(in, dev);
    const NonbondedResult got = finish_nonbonded(ff, a, q, dev);
    const double scale = 1.0 + std::fabs(ref.e[EnergyTerm::VdW]);
    RM_CHECK_MSG(std::fabs(got.e[EnergyTerm::VdW] - ref.e[EnergyTerm::VdW]) < 1e-5 * scale, "emulated e_vdW");
    RM_CHECK_MSG(std::fabs(got.e[EnergyTerm::Coulomb] - ref.e[EnergyTerm::Coulomb]) < 1e-5 * (1.0 + std::fabs(ref.e[EnergyTerm::Coulomb])), "emulated e_ele");
    RM_CHECK(got.e[EnergyTerm::Polarization] == ref.e[EnergyTerm::Polarization]);
    double worst = 0, fmax = 0;
    for (std::size_t k = 0; k < ref.grad.size(); ++k) { worst = std::fmax(worst, std::fabs(got.grad[k] - ref.grad[k])); fmax = std::fmax(fmax, std::fabs(ref.grad[k])); }
    RM_CHECK_MSG(worst < 1e-4 * (1.0 + fmax), "emulated gradient vs CPU-64");
    // deterministic: second run is bitwise identical
    NonbondedDeviceOutput dev2;
    emulate_nonbonded(in, dev2);
    RM_CHECK(dev2.grad == dev.grad && dev2.row_e == dev.row_e);
    (void)periodic;
    std::printf("  nonbonded emulation (%s): nall %zu, pairs %zu, |dE_vdW| %.2e, max |dgrad| %.2e (max |grad| %.1f)\n", mode == 0 ? "periodic" : (mode == 1 ? "cluster" : "small cell"), a.nall(), ref.pairs,
                std::fabs(got.e[EnergyTerm::VdW] - ref.e[EnergyTerm::VdW]), worst, fmax);
  }
#else
  std::puts("  nonbonded emulation skipped (REAXMETAL_FFIELD_DIR not set)");
#endif
}

// ---- the bonded pipeline: the SAME host orchestration run against a CPU backend that executes the M6 kernels as C++ --------------------------
struct EmuBondedBackend final : BondedBackend {
  const BondedDeviceInput* in = nullptr;
  std::vector<float> wf, gp;
  std::vector<std::int32_t> wi;
  RmBParams p{};
  void setup(const BondedDeviceInput& input, const BondedLayout& L) override {
    in = &input;
    wf.assign(L.wf_size, 0.0f);
    wi.assign(L.wi_size, 0);
    p = RmBParams{};
    p.N = L.N; p.nlocal = L.nlocal; p.B = L.B; p.H = L.H; p.NB = static_cast<std::uint32_t>(L.NB); p.ntypes = input.ntypes;
    p.ncx = input.list.grid.ncell[0]; p.ncy = input.list.grid.ncell[1]; p.ncz = input.list.grid.ncell[2];
    p.o_atom = static_cast<std::uint32_t>(L.o_atom); p.o_iatom = static_cast<std::uint32_t>(L.o_iatom); p.o_tkl = static_cast<std::uint32_t>(L.o_tkl);
    p.o_tfl = static_cast<std::uint32_t>(L.o_tfl); p.o_hi = static_cast<std::uint32_t>(L.o_hi); p.o_hf = static_cast<std::uint32_t>(L.o_hf);
    p.enobonds = input.enobonds;
    p.bond_cut = input.bond_cut; p.bo_cut = input.bo_cut; p.thb_cut = input.thb_cut; p.thb_cutsq = input.thb_cutsq; p.hbond_cut = input.hbond_cut;
  }
  template <class K>
  void each(std::uint32_t threads, K kernel) {
    for (std::uint32_t t = 0; t < threads; ++t)
      kernel(in->list.x.data(), in->type.data(), in->tag.data(), in->list.grid.atom_cell.data(), in->list.grid.cell_start.data(), in->list.grid.cell_items.data(),
             in->sb_f.data(), in->sb_i.data(), in->tb_f.data(), in->tb_i.data(), in->thb_idx.data(), in->thb_sets.data(), in->fb_f.data(), in->fb_has.data(), in->hb_f.data(),
             in->gp.data(), wf.data(), wi.data(), in->list.x_lo.data(), p, t);
  }
  void run(std::span<const BondedStep> steps) override {
    for (const BondedStep& st : steps) {
      const std::string k = st.kernel;
      if (k == "rm_b_build") each(st.threads, rm_b_build);
      else if (k == "rm_h_build") each(st.threads, rm_h_build);
      else if (k == "rm_b_prime") each(st.threads, rm_b_prime);
      else if (k == "rm_b_correct") each(st.threads, rm_b_correct);
      else if (k == "rm_b_atom") each(st.threads, rm_b_atom);
      else if (k == "rm_b_valence") each(st.threads, rm_b_valence);
      else if (k == "rm_b_torsion") each(st.threads, rm_b_torsion);
      else if (k == "rm_b_hbond") each(st.threads, rm_b_hbond);
      else if (k == "rm_b_hbgather") each(st.threads, rm_b_hbgather);
      else if (k == "rm_b_cdgather") each(st.threads, rm_b_cdgather);
      else if (k == "rm_b_dbond") each(st.threads, rm_b_dbond);
      else if (k == "rm_b_force") each(st.threads, rm_b_force);
      else RM_CHECK_MSG(false, "unknown kernel " + k);
    }
  }
  void read_float(std::size_t o, std::size_t n, float* dst) override { std::copy(wf.begin() + static_cast<std::ptrdiff_t>(o), wf.begin() + static_cast<std::ptrdiff_t>(o + n), dst); }
  void read_int(std::size_t o, std::size_t n, std::int32_t* dst) override { std::copy(wi.begin() + static_cast<std::ptrdiff_t>(o), wi.begin() + static_cast<std::ptrdiff_t>(o + n), dst); }
};

static void test_bonded_pipeline_vs_cpu64() {
#ifdef REAXMETAL_FFIELD_DIR
  const ForceField ff = read_force_field_file(std::string(REAXMETAL_FFIELD_DIR) + "/ffield.reax.cho");
  const int tC = ff.match_element("C").at(0), tH = ff.match_element("H").at(0), tO = ff.match_element("O").at(0);
  Rng r(97531);
  for (const int mode : {0, 1}) {
    const Box box = mode == 0 ? Box::orthogonal({0, 0, 0}, {13.0, 13.5, 14.0}, {true, true, true}) : Box::orthogonal({0, 0, 0}, {13.0, 13.5, 14.0}, {false, false, false});
    // small molecules on a loose grid: C4H6O-like fragments with perturbed bond lengths
    std::vector<double> x;
    std::vector<int> type;
    std::vector<std::int64_t> tag;
    const double bl = 1.5;
    for (int m = 0; m < 6; ++m) {
      const Vec3 o{2.5 + 3.5 * (m % 3), 3.0 + 4.0 * (m / 3), 4.0};
      const Vec3 off[8] = {{0, 0, 0}, {bl, 0, 0}, {-0.6, 0.9, 0.5}, {-0.6, -0.9, 0.5}, {bl + 0.6, 0.9, -0.5}, {bl + 0.6, -0.9, -0.5}, {-0.2, 0.1, -1.1}, {bl + 0.9, 0.0, 0.9}};
      const int ty[8] = {tC, tC, tH, tH, tH, tH, tO, tH};
      for (int a = 0; a < 8; ++a) {
        for (int c = 0; c < 3; ++c) x.push_back(o[static_cast<std::size_t>(c)] + off[a][static_cast<std::size_t>(c)] + 0.12 * (r.u() - 0.5));
        type.push_back(ty[a]);
        tag.push_back(static_cast<std::int64_t>(type.size()));
      }
    }
    NeighborCutoffs cut;
    cut.nonb = ff.file_control().nonb_cut;
    ExpandOptions eo; eo.shell = cut.required_shell();
    const AtomSet a = expand_images(box, x, type, tag, eo);
    const FarList f = build_far_list(a, cut);
    ControlParams ctl;
    const BondedResult ref = compute_bonded_core(ff, ctl, a, f);
    const BondedDeviceInput in = make_bonded_device_input(ff, ctl, a, box, {});
    EmuBondedBackend be;
    const BondedDeviceOutput dev = run_bonded_pipeline(be, in, 4, 4);   // tiny initial capacities: exercises grow-and-retry
    const BondedResult got = finish_bonded(dev, a.nall());
    RM_CHECK_MSG(dev.attempts >= 2, "capacity growth must have been exercised");
    double worst_e = 0.0;
    for (const EnergyTerm t : {EnergyTerm::Bond, EnergyTerm::LonePair, EnergyTerm::Over, EnergyTerm::Under, EnergyTerm::Valence, EnergyTerm::Penalty,
                               EnergyTerm::Coalition, EnergyTerm::Torsion, EnergyTerm::Conjugation, EnergyTerm::HBond}) {
      const double d = std::fabs(got.e[t] - ref.e[t]) / (1.0 + std::fabs(ref.e[t]));
      worst_e = std::fmax(worst_e, d);
      RM_CHECK_MSG(d < 2e-4, "bonded term energy");
    }
    double worst_g = 0.0, gmax = 0.0;
    for (std::size_t k = 0; k < ref.grad.size(); ++k) { worst_g = std::fmax(worst_g, std::fabs(got.grad[k] - ref.grad[k])); gmax = std::fmax(gmax, std::fabs(ref.grad[k])); }
    RM_CHECK_MSG(worst_g < 2e-3 * (1.0 + gmax), "bonded gradient vs CPU-64");
    EmuBondedBackend be2;
    const BondedDeviceOutput dev2 = run_bonded_pipeline(be2, in, 4, 4);
    RM_CHECK(dev2.grad == dev.grad);
    std::printf("  bonded pipeline emulation (%s): nall %zu, caps B=%u H=%u after %u attempts, worst term rel %.2e, max |dgrad| %.2e (max |grad| %.1f)\n", mode == 0 ? "periodic" : "cluster",
                a.nall(), dev.bond_cap, dev.hbond_cap, dev.attempts, worst_e, worst_g, gmax);
  }
#else
  std::puts("  bonded pipeline emulation skipped (REAXMETAL_FFIELD_DIR not set)");
#endif
}

// The double-single EEM kernels (rm_qeq_h, rm_cg_res_df) against a double precision reference on a random cluster: the matrix entries must agree to
// ~1e-11 relative (single precision would give 1e-7), the residual to the 1e-7 of its float output.
void test_qeq_double_single() {
  Rng r(20261009);
  const std::uint32_t N = 60;
  const double swa = 0.0, swb = 10.0, c144 = 14.4;
  std::vector<double> xd(3 * N);
  for (auto& v : xd) v = 14.0 * r.u();
  std::vector<float> xh(3 * N), xl(3 * N);
  for (std::size_t k = 0; k < xd.size(); ++k) { xh[k] = static_cast<float>(xd[k]); xl[k] = static_cast<float>(xd[k] - static_cast<double>(xh[k])); }
  const std::uint32_t cap = N;
  FarRowsF32 rows;
  rows.nall = N; rows.cap = cap; rows.count.assign(N, 0); rows.nbr.assign(static_cast<std::size_t>(N) * cap, -1); rows.r2.assign(static_cast<std::size_t>(N) * cap, 0.0f);
  for (std::uint32_t i = 0; i < N; ++i)
    for (std::uint32_t j = i + 1; j < N; ++j) rows.nbr[static_cast<std::size_t>(i) * cap + rows.count[i]++] = static_cast<std::int32_t>(j);   // all pairs; the kernel applies r <= swb
  const ColumnIndex col = build_column_index(rows, N);
  const double gamma = 0.8, gi = std::pow(gamma * gamma, -1.5), eta = 8.5;
  auto split = [](double v, float& h, float& l) { h = static_cast<float>(v); l = static_cast<float>(v - static_cast<double>(h)); };
  float shld_h, shld_l, eta_h, eta_l;
  split(gi, shld_h, shld_l); split(eta, eta_h, eta_l);
  RmQeqDfParams dp{};
  dp.nlocal = N; dp.cap = cap; dp.ntypes = 1; dp.swb = static_cast<float>(swb);
  float swa_h; split(swa, swa_h, dp.swa_lo); dp.swa_hi = swa_h;
  split(swb - swa, dp.d_hi, dp.d_lo); split(c144, dp.c_hi, dp.c_lo);
  std::vector<std::int32_t> type(N, 0);
  std::vector<float> hv(static_cast<std::size_t>(N) * cap, 0.0f), hvl(hv.size(), 0.0f);
  for (std::uint32_t i = 0; i < N; ++i) rm_qeq_h(xh.data(), xl.data(), type.data(), &shld_h, &shld_l, rows.nbr.data(), rows.count.data(), hv.data(), hvl.data(), dp, i);
  const terms::TaperCoeffs tap = terms::taper_coeffs(swa, swb);
  double worst = 0;
  std::size_t checked = 0;
  std::vector<double> Href(static_cast<std::size_t>(N) * N, 0.0);
  for (std::uint32_t i = 0; i < N; ++i)
    for (std::uint32_t e = 0; e < rows.count[i]; ++e) {
      const std::uint32_t j = static_cast<std::uint32_t>(rows.nbr[static_cast<std::size_t>(i) * cap + e]);
      const double dx = xd[3 * j] - xd[3 * i], dy = xd[3 * j + 1] - xd[3 * i + 1], dz = xd[3 * j + 2] - xd[3 * i + 2];
      const double rr = std::sqrt(dx * dx + dy * dy + dz * dz);
      double ref = 0.0;
      if (rr <= swb) {
        double Tap, dTap;
        terms::taper_horner<double>(tap.c, rr, Tap, dTap);
        ref = Tap * c144 / std::cbrt(rr * rr * rr + gi);
      }
      const double got = static_cast<double>(hv[static_cast<std::size_t>(i) * cap + e]) + static_cast<double>(hvl[static_cast<std::size_t>(i) * cap + e]);
      if (rr <= swb - 1e-3) { worst = std::fmax(worst, std::fabs(got - ref)); ++checked; }   // absolute error in eV (float arithmetic gives ~1e-8 .. 1e-6 here)
      Href[static_cast<std::size_t>(i) * N + j] = Href[static_cast<std::size_t>(j) * N + i] = ref;
    }
  RM_CHECK(checked > 500);
  RM_CHECK_MSG(worst < 5e-11, "double-single H entries differ from double by more than 5e-11 eV");
  // residual of a random x for two systems
  const std::uint32_t n2 = 2 * N;
  std::vector<double> xs(n2), bs(n2);
  for (auto& v : xs) v = r.u() - 0.5;
  for (auto& v : bs) v = 2.0 * r.u() - 1.0;
  std::vector<float> xsh(n2), xsl(n2), bsh(n2), bsl(n2), eh(N, eta_h), el(N, eta_l), rout(n2, 0.0f);
  for (std::uint32_t k = 0; k < n2; ++k) { split(xs[k], xsh[k], xsl[k]); split(bs[k], bsh[k], bsl[k]); }
  std::vector<std::int32_t> owner(N);
  for (std::uint32_t i = 0; i < N; ++i) owner[i] = static_cast<std::int32_t>(i);
  RmCgParams cp{};
  cp.nlocal = N; cp.cap = cap;
  for (std::uint32_t g = 0; g < n2; ++g)
    rm_cg_res_df(hv.data(), hvl.data(), rows.nbr.data(), rows.count.data(), owner.data(), col.start.data(), col.items.data(), xsh.data(), xsl.data(), eh.data(), el.data(),
                 bsh.data(), bsl.data(), rout.data(), cp, g);
  double rworst = 0;
  for (std::uint32_t sys = 0; sys < 2; ++sys)
    for (std::uint32_t i = 0; i < N; ++i) {
      double y = eta * xs[sys * N + i];
      for (std::uint32_t j = 0; j < N; ++j) y += Href[static_cast<std::size_t>(i) * N + j] * xs[sys * N + j];
      const double rref = bs[sys * N + i] - y;
      rworst = std::fmax(rworst, std::fabs(static_cast<double>(rout[sys * N + i]) - rref) / (std::fabs(bs[sys * N + i]) + std::fabs(y)));
    }
  RM_CHECK_MSG(rworst < 2e-7, "double-single residual differs from double by more than 2e-7 of |b| + |A x| (the float output rounding is 6e-8)");
  std::printf("  double-single EEM: %zu matrix entries, worst absolute error %.2e eV (float would be ~1e-8..1e-6); residual worst %.2e of |b|+|Ax|\n", checked, worst, rworst);
}

int main() {
  test_shader_source_lints();
  test_saxpy_and_probe();
  test_far_rows_vs_cpu();
  test_shared_systems();
  test_growth();
  test_reduction_kernels();
  test_nonbonded_vs_cpu64();
  test_bonded_pipeline_vs_cpu64();
  test_qeq_double_single();
  if (rmtest::failures() != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", rmtest::failures());
    return 1;
  }
  std::puts("metal_emulation: all checks passed (EMULATED on the CPU; not Metal)");
  return 0;
}
