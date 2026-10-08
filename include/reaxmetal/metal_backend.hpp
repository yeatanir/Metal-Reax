// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// Metal backend, milestone M3 (Apple platforms). The implementation is Objective-C++ against the system Metal framework
// (src/metal/metal_backend.mm); on every other platform a stub reports "not available". Plain C++ interface, no Apple types.
//
// STATUS (docs/ARCHITECTURE_DECISIONS.md ADR-016 taxonomy): written; NOT compiled and NOT executed anywhere yet. The kernels
// are additionally exercised by CPU emulation (tests/test_metal_emulation.cpp), which is not evidence about Metal.
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "reaxmetal/neighbor.hpp"

namespace reaxmetal::mtl {   // not `metal`: that name is Metal Shading Language's own namespace

class MetalError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// True only in a build that contains the Objective-C++ implementation (REAXMETAL_ENABLE_METAL=ON on an Apple platform).
bool compiled_with_metal() noexcept;
// The complete MSL source handed to the runtime compiler: reaxmetal_m3_types.h + "\n" + reaxmetal_m3.metal. Available everywhere.
std::string shader_source();
// Kernel entry points the runtime library must contain.
std::vector<std::string> kernel_names();

struct DeviceInfo {
  std::string name;
  std::string os_version;
  bool unified_memory = false;
  std::uint64_t recommended_max_working_set = 0;
  std::uint64_t max_buffer_length = 0;
  std::uint32_t max_threads_per_threadgroup = 0;
  std::string families;          // e.g. "Apple7 Apple8 Apple9 Metal3"
};

struct KernelInfo {
  std::string name;
  std::uint32_t max_threads_per_threadgroup = 0;
  std::uint32_t thread_execution_width = 0;
  std::uint32_t static_threadgroup_memory = 0;
};

struct MathProbe {
  bool nan_detected = false;     // false => the compiler assumed no NaNs (fast-math took effect)
  float mul_add = 0.0f;          // 0 => product rounded before the add; 2^-26 => fused multiply-add
};

class Context {
 public:
  Context();                     // opens the default device, compiles shader_source() at run time; throws MetalError
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  DeviceInfo device_info() const;
  const std::vector<KernelInfo>& kernels() const;
  const std::string& compile_log() const;        // compiler messages (warnings) of the successful compilation
  double compile_seconds() const;
  double last_gpu_seconds() const;               // GPU start->end of the last command buffer

  // MET-1
  std::vector<float> saxpy(float a, std::span<const float> x, std::span<const float> y);
  MathProbe math_probe();
  // NBR-1: rows over owned+ghost atoms; grows the row capacity and retries when a row overflows
  FarRowsF32 far_rows(const DeviceListInput& in, std::uint32_t initial_cap = 128, unsigned* launches = nullptr);
  // FORCE-2
  std::vector<float> partial_sums(std::span<const float> v, std::uint32_t chunk);
  float sum(std::span<const float> v, std::uint32_t chunk);   // partial_sums + rm_sum_partials in one command buffer

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace reaxmetal::mtl
