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
constexpr const char kTypes4[] =
#include "reaxmetal_m4_types.inc"
    ;
constexpr const char kTerms[] =
#include "reaxmetal_terms.inc"
    ;
constexpr const char kKernels4[] =
#include "reaxmetal_m4_kernels.inc"
    ;
}  // namespace

// order: M3 types, M4 types (+ math macros), M3 kernels (they open with <metal_stdlib>), terms.hpp, M4 kernels
std::string shader_source() {
  return std::string(kTypes) + "\n" + std::string(kTypes4) + "\n" + std::string(kKernels) + "\n" + std::string(kTerms) + "\n" + std::string(kKernels4);
}

std::vector<std::string> kernel_names() { return {"rm_saxpy", "rm_math_probe", "rm_far_rows", "rm_partial_sums", "rm_sum_partials", "rm_nb_pairs", "rm_nb_gather"}; }

}  // namespace reaxmetal::mtl
