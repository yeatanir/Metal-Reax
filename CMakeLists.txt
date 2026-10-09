# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
cmake_minimum_required(VERSION 3.24)
project(ReaxMetal VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(REAXMETAL_BUILD_LAMMPS_PLUGIN "Build the LAMMPS plugin(s); needs REAXMETAL_LAMMPS_SOURCE_DIR (pinned LAMMPS src/)" OFF)
set(REAXMETAL_LAMMPS_SOURCE_DIR "" CACHE PATH "src/ directory of the pinned LAMMPS tree (tools/fetch_lammps.sh --full)")
set(REAXMETAL_FFIELD_DIR "" CACHE PATH "directory with the bundled ffield.reax.* files (potentials/ of the pinned LAMMPS tree); enables the ffield_tables test")
set(REAXMETAL_LAMMPS_PREFIX "" CACHE PATH "install prefix of the stock pinned LAMMPS build (tools/build_lammps_reference.sh); enables the lammps_probe CTest")
# shared library of the stock pinned LAMMPS build (used by the in-LAMMPS tests that drive it through its C library interface)
set(RM_LIBLAMMPS "")
foreach(_cand "${REAXMETAL_LAMMPS_PREFIX}/lib/liblammps.so" "${REAXMETAL_LAMMPS_PREFIX}/lib/liblammps.dylib")
  if(REAXMETAL_LAMMPS_PREFIX AND EXISTS "${_cand}" AND NOT RM_LIBLAMMPS)
    set(RM_LIBLAMMPS "${_cand}")
  endif()
endforeach()
option(REAXMETAL_BUILD_TESTS "Build the CTest suite" ON)
# Metal is only available on Apple platforms. REAXMETAL_ENABLE_METAL=ON adds the Objective-C++ backend (src/metal/metal_backend.mm,
# system Metal.framework, shaders compiled at run time - no Xcode/metal compiler needed, ADR-008/ADR-023). It has never been built
# anywhere yet (docs/VALIDATION.md MET-1). On other platforms the option is refused rather than silently building a stub.
option(REAXMETAL_ENABLE_METAL "Build the native Metal backend (Apple only)" OFF)
if(REAXMETAL_ENABLE_METAL AND NOT APPLE)
  message(FATAL_ERROR "REAXMETAL_ENABLE_METAL requires an Apple platform.")
endif()

include(cmake/ReaxMetalFlags.cmake)

add_library(reaxmetal_core STATIC
  src/core/capabilities.cpp
  src/core/compat_flags.cpp
  src/core/energy_terms.cpp
  src/core/pins.cpp
  src/core/sha256.cpp
  src/core/system.cpp
  src/cpu/bonded.cpp
  src/cpu/nonbonded.cpp
  src/neighbor/neighbor.cpp
  src/neighbor/nonbonded_device.cpp
  src/neighbor/qeq_device.cpp
  src/neighbor/bonded_device.cpp
  src/io/ffield_parser.cpp
  src/io/control_parser.cpp
  src/io/pair_settings.cpp
  src/io/table_dump.cpp)
set_target_properties(reaxmetal_core PROPERTIES POSITION_INDEPENDENT_CODE ON)  # linked into the LAMMPS plugin (a shared module)
target_include_directories(reaxmetal_core PUBLIC include)
target_include_directories(reaxmetal_core PRIVATE ${CMAKE_SOURCE_DIR}/src/metal/shaders)
target_compile_features(reaxmetal_core PUBLIC cxx_std_20)
reaxmetal_apply_warnings(reaxmetal_core)
# Reference-quality FP: no contraction, no fast-math (docs/NUMERICAL_POLICY.md section 2).
reaxmetal_apply_strict_fp(reaxmetal_core)

# ---- Metal backend library: shader sources (always), Objective-C++ implementation (Apple + REAXMETAL_ENABLE_METAL) or a stub ----
include(cmake/EmbedText.cmake)
set(RM_GEN ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${RM_GEN})
reaxmetal_embed_text(${CMAKE_SOURCE_DIR}/src/metal/shaders/reaxmetal_m3_types.h ${RM_GEN}/reaxmetal_m3_types.inc)
reaxmetal_embed_text(${CMAKE_SOURCE_DIR}/src/metal/shaders/reaxmetal_m3.metal ${RM_GEN}/reaxmetal_m3_kernels.inc)
reaxmetal_embed_text(${CMAKE_SOURCE_DIR}/src/metal/shaders/reaxmetal_m4_types.h ${RM_GEN}/reaxmetal_m4_types.inc)
reaxmetal_embed_text(${CMAKE_SOURCE_DIR}/include/reaxmetal/terms.hpp ${RM_GEN}/reaxmetal_terms.inc)
reaxmetal_embed_text(${CMAKE_SOURCE_DIR}/src/metal/shaders/reaxmetal_m4.metal ${RM_GEN}/reaxmetal_m4_kernels.inc)
reaxmetal_embed_text(${CMAKE_SOURCE_DIR}/src/metal/shaders/reaxmetal_m6_types.h ${RM_GEN}/reaxmetal_m6_types.inc)
reaxmetal_embed_text(${CMAKE_SOURCE_DIR}/src/metal/shaders/reaxmetal_m6.metal ${RM_GEN}/reaxmetal_m6_kernels.inc)
if(REAXMETAL_ENABLE_METAL)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 20)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  add_library(reaxmetal_metal STATIC src/metal/metal_source.cpp src/metal/metal_backend.mm)
  target_compile_options(reaxmetal_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  find_library(RM_METAL_FRAMEWORK Metal REQUIRED)
  find_library(RM_FOUNDATION_FRAMEWORK Foundation REQUIRED)
  target_link_libraries(reaxmetal_metal PUBLIC reaxmetal_core ${RM_METAL_FRAMEWORK} ${RM_FOUNDATION_FRAMEWORK})
else()
  add_library(reaxmetal_metal STATIC src/metal/metal_source.cpp src/metal/metal_backend_stub.cpp)
  target_link_libraries(reaxmetal_metal PUBLIC reaxmetal_core)
endif()
target_include_directories(reaxmetal_metal PRIVATE ${RM_GEN} ${CMAKE_SOURCE_DIR}/src/metal/shaders)
target_include_directories(reaxmetal_metal PUBLIC include)
reaxmetal_apply_warnings(reaxmetal_metal)
reaxmetal_apply_strict_fp(reaxmetal_metal)

add_executable(reaxmetal_ffield_dump tools/ffield_dump.cpp)
target_link_libraries(reaxmetal_ffield_dump PRIVATE reaxmetal_core)
reaxmetal_apply_warnings(reaxmetal_ffield_dump)

add_executable(reaxmetal_metal_check tools/metal_check.cpp)
target_link_libraries(reaxmetal_metal_check PRIVATE reaxmetal_metal)
target_include_directories(reaxmetal_metal_check PRIVATE tools)
reaxmetal_apply_warnings(reaxmetal_metal_check)
reaxmetal_apply_strict_fp(reaxmetal_metal_check)

add_executable(reaxmetal_neighbor_tool tools/neighbor_tool.cpp)
target_link_libraries(reaxmetal_neighbor_tool PRIVATE reaxmetal_core)
reaxmetal_apply_warnings(reaxmetal_neighbor_tool)

add_executable(reaxmetal_energy_tool tools/energy_tool.cpp)
target_link_libraries(reaxmetal_energy_tool PRIVATE reaxmetal_metal)
reaxmetal_apply_warnings(reaxmetal_energy_tool)

if(REAXMETAL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(REAXMETAL_BUILD_LAMMPS_PLUGIN)
  add_subdirectory(plugin)
endif()
