// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// A FAKE implementation of reaxmetal::mtl::Context that executes the Metal kernels as C++ on the CPU (tests/metal_shim). Linked with
// tools/metal_check.cpp it lets the on-device check tool run end to end on Linux, which tests the TOOL (its expectations, its control
// flow, its comparisons) before it is ever run on a GPU. The device name says EMULATED; a pass here is not a Metal result.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "reaxmetal/metal_backend.hpp"

#include "metal_shim.hpp"
#include "reaxmetal_m3_types.h"
#include "reaxmetal_m3.metal"
#include "metal_shim_end.hpp"

namespace reaxmetal::mtl {

struct Context::Impl {
  std::vector<KernelInfo> kernels;
  std::string log;
  double gpu_seconds = 0.0;
};

bool compiled_with_metal() noexcept { return true; }   // the fake claims a backend so that the tool runs; see the device name

Context::Context() : impl_(new Impl) {
  for (const auto& n : kernel_names()) impl_->kernels.push_back({n, 1024, 32, 0});
}
Context::~Context() = default;

DeviceInfo Context::device_info() const {
  DeviceInfo i;
  i.name = "EMULATED CPU (tests/metal_check_emulated.cpp) - NOT A GPU";
  i.os_version = "n/a";
  i.unified_memory = true;
  i.families = "none";
  i.max_threads_per_threadgroup = 1024;
  return i;
}
const std::vector<KernelInfo>& Context::kernels() const { return impl_->kernels; }
const std::string& Context::compile_log() const { return impl_->log; }
double Context::compile_seconds() const { return 0.0; }
double Context::last_gpu_seconds() const { return impl_->gpu_seconds > 0 ? impl_->gpu_seconds : 1e-9; }

namespace {
struct Timer {
  double& out;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  explicit Timer(double& o) : out(o) {}
  ~Timer() { out = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
}  // namespace

std::vector<float> Context::saxpy(float a, std::span<const float> x, std::span<const float> y) {
  Timer t(impl_->gpu_seconds);
  if (x.size() != y.size()) throw MetalError("saxpy: x and y differ in size");
  std::vector<float> out(y.begin(), y.end());
  RmSaxpyParams p{static_cast<std::uint32_t>(x.size()), a};
  for (std::uint32_t i = 0; i < x.size(); ++i) rm_saxpy(x.data(), out.data(), p, i);
  return out;
}

MathProbe Context::math_probe() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float a = 1.0f + std::ldexp(1.0f, -13), c = -(1.0f + std::ldexp(1.0f, -12));
  const float in[4] = {nan, a, a, c};
  float out[2] = {-1.0f, -1.0f};
  rm_math_probe(in, out, 0);
  return {out[0] == 1.0f, out[1]};
}

FarRowsF32 Context::far_rows(const DeviceListInput& in, std::uint32_t initial_cap, unsigned* launches) {
  Timer t(impl_->gpu_seconds);
  auto launch = [&](std::uint32_t cap) {
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
  };
  return build_far_rows_with_growth(launch, initial_cap, in.nall, 4, launches);
}

std::vector<float> Context::partial_sums(std::span<const float> v, std::uint32_t chunk) {
  Timer t(impl_->gpu_seconds);
  if (chunk == 0) throw MetalError("partial_sums: chunk must be > 0");
  if (v.empty()) return {};
  const std::uint32_t n = static_cast<std::uint32_t>(v.size()), nchunk = (n + chunk - 1) / chunk;
  std::vector<float> part(nchunk);
  RmReduceParams p{n, chunk};
  for (std::uint32_t c = 0; c < nchunk; ++c) rm_partial_sums(v.data(), part.data(), p, c);
  return part;
}

float Context::sum(std::span<const float> v, std::uint32_t chunk) {
  Timer t(impl_->gpu_seconds);
  if (chunk == 0) throw MetalError("sum: chunk must be > 0");
  if (v.empty()) return 0.0f;
  const std::uint32_t n = static_cast<std::uint32_t>(v.size()), nchunk = (n + chunk - 1) / chunk;
  std::vector<float> part(nchunk);
  RmReduceParams p{n, chunk};
  for (std::uint32_t c = 0; c < nchunk; ++c) rm_partial_sums(v.data(), part.data(), p, c);
  float out = 0.0f;
  rm_sum_partials(part.data(), &out, p, 0);
  return out;
}

}  // namespace reaxmetal::mtl
