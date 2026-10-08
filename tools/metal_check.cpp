// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// On-device checks of the M3 Metal backend, run on the Apple machine by tools/mac/step{1,2,3}_*.sh:
//   reaxmetal_metal_check --step 1   bring-up: device, runtime shader compile, trivial kernels, buffer round trip   (MET-1)
//   reaxmetal_metal_check --step 2   device neighbor rows vs the CPU-64 list on shared geometries                    (NBR-1/NBR-2)
//   reaxmetal_metal_check --step 3   fixed-order reductions bitwise vs the CPU twin, run-to-run determinism          (FORCE-2, MET-4)
// Output: one line per check, "[ID] PASS|FAIL|INFO ...", then "STEP n RESULT: PASS|FAIL (k failures)". Exit 0 only on PASS.
// On any platform without the Metal build the tool says so and exits 3 (it never reports PASS without a GPU).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "m3_systems.hpp"
#include "reaxmetal/metal_backend.hpp"
#include "reaxmetal/neighbor.hpp"

using namespace reaxmetal;

namespace {
int g_fail = 0;

void report(const char* id, bool ok, const std::string& msg) {
  std::printf("[%s] %s %s\n", id, ok ? "PASS" : "FAIL", msg.c_str());
  if (!ok) ++g_fail;
}
void info(const char* id, const std::string& msg) { std::printf("[%s] INFO %s\n", id, msg.c_str()); }

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

struct Lcg {   // tiny deterministic generator for reduction inputs
  std::uint64_t s;
  explicit Lcg(std::uint64_t seed) : s(seed) {}
  double u() { s = s * 6364136223846793005ull + 1442695040888963407ull; return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0); }
};

int step1(mtl::Context& ctx) {
  const mtl::DeviceInfo d = ctx.device_info();
  info("MET-1a", fmt("device='%s' os='%s' unified_memory=%d families=[%s] max_threads_per_threadgroup=%u recommended_working_set=%.1f GiB max_buffer=%.1f GiB",
                     d.name.c_str(), d.os_version.c_str(), d.unified_memory ? 1 : 0, d.families.c_str(), d.max_threads_per_threadgroup,
                     static_cast<double>(d.recommended_max_working_set) / 1073741824.0, static_cast<double>(d.max_buffer_length) / 1073741824.0));
  report("MET-1a", !d.name.empty(), "a Metal device exists");
  report("MET-1b", true, fmt("runtime compilation of the shader source succeeded in %.3f s (no Xcode/metal compiler involved)", ctx.compile_seconds()));
  if (!ctx.compile_log().empty()) info("MET-1b", "compiler messages: " + ctx.compile_log());
  for (const auto& k : ctx.kernels())
    info("MET-1b", fmt("kernel %-16s max_threads=%u exec_width=%u static_tg_mem=%u", k.name.c_str(), k.max_threads_per_threadgroup, k.thread_execution_width, k.static_threadgroup_memory));
  report("MET-1b", ctx.kernels().size() == mtl::kernel_names().size(), "every kernel has a compute pipeline");

  // saxpy with values whose result is exact in float, including sizes around thread-group boundaries
  bool ok = true;
  for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{63}, std::size_t{64}, std::size_t{65}, std::size_t{1000}, std::size_t{4194304}}) {
    std::vector<float> x(n), y(n), want(n);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = static_cast<float>(i % 17) - 8.0f;
      y[i] = static_cast<float>(i % 5);
      want[i] = 2.5f * x[i] + y[i];
    }
    const auto got = ctx.saxpy(2.5f, x, y);
    const bool same = got.size() == n && std::memcmp(got.data(), want.data(), n * sizeof(float)) == 0;
    ok = ok && same;
    if (!same) info("MET-1c", fmt("saxpy n=%zu differs from the exact result", n));
    else if (n == 4194304) info("MET-1d", fmt("saxpy n=%zu: GPU time %.3f ms (%.1f GB/s effective)", n, ctx.last_gpu_seconds() * 1e3, 3.0 * 4.0 * static_cast<double>(n) / ctx.last_gpu_seconds() / 1e9));
  }
  report("MET-1c", ok, "saxpy kernel: results bitwise equal to the exact CPU result for n = 1 ... 4194304 (buffer round trip, partial threadgroups)");
  report("MET-1c", ctx.saxpy(1.0f, {}, {}).empty(), "empty input handled");

  const mtl::MathProbe m = ctx.math_probe();
  report("MET-1e", m.nan_detected, "NaN detected on the GPU => safe-math compilation took effect");
  info("MET-1f", fmt("multiply-add probe a*b+c = %.9g (0 = product rounded first; %.9g = fused multiply-add contraction)", static_cast<double>(m.mul_add), std::ldexp(1.0, -26)));
  return g_fail;
}

int step2(mtl::Context& ctx, bool quick) {
  for (const auto& sys : m3::systems(!quick)) {
    const FarList cpu = build_far_list(sys.atoms, sys.cut);
    const DeviceListInput in = make_device_list_input(sys.atoms, sys.box, sys.cut, 1e-3);
    unsigned launches = 0;
    const FarRowsF32 a = ctx.far_rows(in, 16, &launches);
    const double gpu_ms = ctx.last_gpu_seconds() * 1e3;
    const RowComparison c = compare_rows_to_far_list(a, cpu, sys.atoms, sys.cut, in.list_margin);
    report("NBR-1", c.ok(), fmt("%-32s nall=%zu cpu_entries=%zu device_entries=%zu missing=%zu illegal_extra=%zu bad_r2=%zu structural=%zu overflow=%d | band_pairs=%zu launches=%u gpu=%.3f ms",
                               sys.name.c_str(), sys.atoms.nall(), c.cpu_entries, c.device_entries, c.missing, c.illegal_extra, c.bad_r2, c.structural, c.overflow ? 1 : 0, c.band_extra, launches, gpu_ms));
    // run-to-run determinism: identical bytes, including the unused part of the rows
    bool same = true;
    for (int rep = 0; rep < 3; ++rep) {
      const FarRowsF32 b = ctx.far_rows(in, a.cap, nullptr);
      same = same && b.count == a.count && b.nbr == a.nbr && std::memcmp(b.r2.data(), a.r2.data(), a.r2.size() * sizeof(float)) == 0;
    }
    report("FORCE-2", same, fmt("%-32s three further launches produce identical rows (bytes)", sys.name.c_str()));
  }
  // forced overflow: a deliberately tiny first capacity must trigger grow-and-retry and still give correct rows
  {
    const auto systems = m3::systems(false);
    const auto& sys = systems[0];
    const FarList cpu = build_far_list(sys.atoms, sys.cut);
    const DeviceListInput in = make_device_list_input(sys.atoms, sys.box, sys.cut, 1e-3);
    unsigned launches = 0;
    const FarRowsF32 r = ctx.far_rows(in, 2, &launches);
    report("NBR-1", launches == 2 && compare_rows_to_far_list(r, cpu, sys.atoms, sys.cut, in.list_margin).ok(), fmt("overflow with capacity 2 -> %u launches, final rows correct (cap %u)", launches, r.cap));
  }
  return g_fail;
}

int step3(mtl::Context& ctx, bool quick) {
  Lcg rng(2024);
  bool all_sum = true, all_part = true;
  std::size_t cases = 0;
  for (std::uint32_t n : {1u, 2u, 255u, 256u, 257u, 1000u, 4097u, 100000u, 1u << 20}) {
    std::vector<float> v(n);
    for (auto& e : v) e = static_cast<float>(rng.u() - 0.5) * 1.0e4f;
    for (std::uint32_t chunk : {1u, 7u, 64u, 256u, 5000u}) {
      const auto want_part = fixed_order_partials_f32(v, chunk);
      const auto got_part = ctx.partial_sums(v, chunk);
      all_part = all_part && got_part.size() == want_part.size() && std::memcmp(got_part.data(), want_part.data(), want_part.size() * sizeof(float)) == 0;
      const float want = fixed_order_sum_f32(v, chunk), got = ctx.sum(v, chunk);
      all_sum = all_sum && std::memcmp(&want, &got, sizeof(float)) == 0;
      ++cases;
    }
  }
  report("FORCE-2", all_part, fmt("partial sums bitwise equal to the CPU twin in %zu (n, chunk) cases", cases));
  report("FORCE-2", all_sum, fmt("full fixed-order sum bitwise equal to the CPU twin in %zu (n, chunk) cases", cases));

  // order sensitivity is real: the crafted vector below sums to different values in different orders; the GPU must pick OURS
  const std::vector<float> trick{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0e8f, -1.0e8f};   // chunk 5 -> 5, left-to-right -> 8
  const float want_trick = fixed_order_sum_f32(trick, 5);
  float natural = 0.0f;
  for (float e : trick) natural += e;
  report("FORCE-2", ctx.sum(trick, 5) == want_trick && want_trick != natural, fmt("order-sensitive vector: GPU %.9g == canonical %.9g (plain left-to-right would give %.9g)", static_cast<double>(ctx.sum(trick, 5)), static_cast<double>(want_trick), static_cast<double>(natural)));

  // repeatability on a big input and timing
  std::vector<float> big(quick ? (1u << 22) : (1u << 24));
  for (auto& e : big) e = static_cast<float>(rng.u() - 0.5) * 1.0e3f;
  const float first = ctx.sum(big, 256);
  bool rep = true;
  double tmin = 1e30;
  for (int i = 0; i < 50; ++i) {
    const float s = ctx.sum(big, 256);
    rep = rep && std::memcmp(&s, &first, sizeof(float)) == 0;
    tmin = std::min(tmin, ctx.last_gpu_seconds());
  }
  const float want_big = fixed_order_sum_f32(big, 256);
  report("FORCE-2", rep && std::memcmp(&first, &want_big, sizeof(float)) == 0, fmt("n=%zu: 50 launches identical and equal to the CPU twin (%.9g); best GPU time %.3f ms", big.size(), static_cast<double>(first), tmin * 1e3));

  // MET-4: a reduction applied to real data - the sum of all device r2 values of a neighbor-row launch
  const auto systems = m3::systems(false);
  const auto& sys = systems[2];   // diamond
  const DeviceListInput in = make_device_list_input(sys.atoms, sys.box, sys.cut, 1e-3);
  const FarRowsF32 rows = ctx.far_rows(in, 400, nullptr);
  std::vector<float> flat;
  for (std::uint32_t i = 0; i < rows.nall; ++i)
    for (std::uint32_t k = 0; k < rows.count[i]; ++k) flat.push_back(rows.r2[static_cast<std::size_t>(i) * rows.cap + k]);
  const float s = ctx.sum(flat, 128), w = fixed_order_sum_f32(flat, 128);
  report("MET-4", std::memcmp(&s, &w, sizeof s) == 0, fmt("neighbor rows -> reduction pipeline (%zu values): GPU %.9g == CPU twin %.9g", flat.size(), static_cast<double>(s), static_cast<double>(w)));
  return g_fail;
}
}  // namespace

int main(int argc, char** argv) {
  int step = 0;
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--step" && i + 1 < argc) step = std::atoi(argv[++i]);
    else if (a == "--quick") quick = true;
    else { std::fprintf(stderr, "usage: reaxmetal_metal_check --step 1|2|3 [--quick]\n"); return 2; }
  }
  if (step < 1 || step > 3) { std::fprintf(stderr, "usage: reaxmetal_metal_check --step 1|2|3 [--quick]\n"); return 2; }
  if (!mtl::compiled_with_metal()) {
    std::printf("STEP %d RESULT: NOT RUN (this binary was built without the Metal backend: needs an Apple platform and -DREAXMETAL_ENABLE_METAL=ON)\n", step);
    return 3;
  }
  try {
    mtl::Context ctx;
    const int failures = step == 1 ? step1(ctx) : step == 2 ? step2(ctx, quick) : step3(ctx, quick);
    std::printf("STEP %d RESULT: %s (%d failures)\n", step, failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("STEP %d RESULT: FAIL (exception: %s)\n", step, e.what());
    return 1;
  }
}
