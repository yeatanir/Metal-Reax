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
option(REAXMETAL_BUILD_TESTS "Build the CTest suite" ON)
# Metal is only available on Apple platforms. The Metal backend does not exist yet (M3);
# the option is reserved so that nothing can silently configure a half-built GPU backend.
option(REAXMETAL_ENABLE_METAL "Build the native Metal backend (Apple only; arrives in M3)" OFF)
if(REAXMETAL_ENABLE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "REAXMETAL_ENABLE_METAL requires an Apple platform.")
  endif()
  message(FATAL_ERROR "REAXMETAL_ENABLE_METAL: the Metal backend is not implemented yet (milestone M3).")
endif()

include(cmake/ReaxMetalFlags.cmake)

add_library(reaxmetal_core STATIC
  src/core/capabilities.cpp
  src/core/compat_flags.cpp
  src/core/energy_terms.cpp
  src/core/pins.cpp
  src/core/sha256.cpp
  src/io/ffield_parser.cpp
  src/io/control_parser.cpp
  src/io/pair_settings.cpp
  src/io/table_dump.cpp)
set_target_properties(reaxmetal_core PROPERTIES POSITION_INDEPENDENT_CODE ON)  # linked into the LAMMPS plugin (a shared module)
target_include_directories(reaxmetal_core PUBLIC include)
target_compile_features(reaxmetal_core PUBLIC cxx_std_20)
reaxmetal_apply_warnings(reaxmetal_core)
# Reference-quality FP: no contraction, no fast-math (docs/NUMERICAL_POLICY.md section 2).
reaxmetal_apply_strict_fp(reaxmetal_core)

add_executable(reaxmetal_ffield_dump tools/ffield_dump.cpp)
target_link_libraries(reaxmetal_ffield_dump PRIVATE reaxmetal_core)
reaxmetal_apply_warnings(reaxmetal_ffield_dump)

if(REAXMETAL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(REAXMETAL_BUILD_LAMMPS_PLUGIN)
  add_subdirectory(plugin)
endif()
