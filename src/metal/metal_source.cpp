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
constexpr const char kTypes6[] =
#include "reaxmetal_m6_types.inc"
    ;
constexpr const char kKernels6[] =
#include "reaxmetal_m6_kernels.inc"
    ;
constexpr const char kKernels4[] =
#include "reaxmetal_m4_kernels.inc"
    ;
}  // namespace

// order: M3 types, M4 types (+ math macros), M6 types, M3 kernels (they open with <metal_stdlib>), terms.hpp, M4 kernels, M6 kernels
std::string shader_source() {
  return std::string(kTypes) + "\n" + std::string(kTypes4) + "\n" + std::string(kTypes6) + "\n" + std::string(kKernels) + "\n" + std::string(kTerms) + "\n" +
         std::string(kKernels4) + "\n" + std::string(kKernels6);
}

std::vector<std::string> kernel_names() { return {"rm_saxpy", "rm_math_probe", "rm_far_rows", "rm_partial_sums", "rm_sum_partials", "rm_nb_pairs", "rm_nb_gather", "rm_qeq_h", "rm_qeq_mv", "rm_cg_res_df", "rm_cg_mv", "rm_cg_start", "rm_cg_dot", "rm_cg_scalar", "rm_cg_update", "rm_cg_dir", "rm_b_build", "rm_h_build", "rm_b_prime", "rm_b_correct", "rm_b_atom", "rm_b_valence", "rm_b_torsion", "rm_b_hbond", "rm_b_hbgather", "rm_b_cdgather", "rm_b_dbond", "rm_b_force"}; }

}  // namespace reaxmetal::mtl
