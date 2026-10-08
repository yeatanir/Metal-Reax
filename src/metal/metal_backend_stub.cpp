// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Non-Apple (or Metal-disabled) builds: the backend does not exist; every entry point says so explicitly (rule 4).
#include "reaxmetal/metal_backend.hpp"

namespace reaxmetal::mtl {

struct Context::Impl {};
namespace {
[[noreturn]] void unavailable() { throw MetalError("the Metal backend is not available in this build (needs an Apple platform and -DREAXMETAL_ENABLE_METAL=ON)"); }
}  // namespace

bool compiled_with_metal() noexcept { return false; }
Context::Context() { unavailable(); }
Context::~Context() = default;
DeviceInfo Context::device_info() const { unavailable(); }
const std::vector<KernelInfo>& Context::kernels() const { unavailable(); }
const std::string& Context::compile_log() const { unavailable(); }
double Context::compile_seconds() const { unavailable(); }
double Context::last_gpu_seconds() const { unavailable(); }
std::vector<float> Context::saxpy(float, std::span<const float>, std::span<const float>) { unavailable(); }
MathProbe Context::math_probe() { unavailable(); }
FarRowsF32 Context::far_rows(const DeviceListInput&, std::uint32_t, unsigned*) { unavailable(); }
NonbondedDeviceOutput Context::nonbonded(const NonbondedDeviceInput&) { unavailable(); }
BondedDeviceOutput Context::bonded(const BondedDeviceInput&) { unavailable(); }
std::vector<float> Context::partial_sums(std::span<const float>, std::uint32_t) { unavailable(); }
float Context::sum(std::span<const float>, std::uint32_t) { unavailable(); }

}  // namespace reaxmetal::mtl
