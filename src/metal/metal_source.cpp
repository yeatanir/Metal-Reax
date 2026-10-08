// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Embeds the MSL sources (assembled by CMake into reaxmetal_m3_source.inc) so that a runtime-compiling backend and the Linux
// tests see byte-identical text.
#include "reaxmetal/metal_backend.hpp"

namespace reaxmetal::mtl {

namespace {
constexpr const char kTypes[] =
#include "reaxmetal_m3_types.inc"
    ;
constexpr const char kKernels[] =
#include "reaxmetal_m3_kernels.inc"
    ;
}  // namespace

std::string shader_source() { return std::string(kTypes) + "\n" + std::string(kKernels); }

std::vector<std::string> kernel_names() { return {"rm_saxpy", "rm_math_probe", "rm_far_rows", "rm_partial_sums", "rm_sum_partials"}; }

}  // namespace reaxmetal::mtl
