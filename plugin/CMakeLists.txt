# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
# LAMMPS plugin(s) for ReaxMetal. Built only on request (needs a pinned LAMMPS source tree and a matching build):
#   cmake -S . -B build -DREAXMETAL_BUILD_LAMMPS_PLUGIN=ON -DREAXMETAL_LAMMPS_SOURCE_DIR=<lammps>/src
# Compile settings MUST match the LAMMPS build the plugin is loaded into (serial vs MPI, integer sizes,
# LAMMPS_EXCEPTIONS) -- see doc/src/Developer_plugins.rst "Compiling plugins".
if(NOT REAXMETAL_LAMMPS_SOURCE_DIR OR NOT EXISTS "${REAXMETAL_LAMMPS_SOURCE_DIR}/pair.h")
  message(FATAL_ERROR "Set REAXMETAL_LAMMPS_SOURCE_DIR to the 'src' directory of the pinned LAMMPS tree")
endif()
option(REAXMETAL_LAMMPS_MPI "The target LAMMPS was built with MPI" OFF)
option(REAXMETAL_LAMMPS_STD_FORMAT "The target LAMMPS was built as C++20 and uses std::format for fmt (see adapter/lammps_fmt_abi.h)" OFF)

add_library(reaxmetal_probe_plugin MODULE probe/probe_plugin.cpp probe/pair_reaxff_metal_probe.cpp)
set_target_properties(reaxmetal_probe_plugin PROPERTIES PREFIX "" SUFFIX ".so" OUTPUT_NAME "reaxmetalprobeplugin"
                      CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(reaxmetal_probe_plugin PRIVATE "${REAXMETAL_LAMMPS_SOURCE_DIR}")
target_compile_definitions(reaxmetal_probe_plugin PRIVATE LAMMPS_EXCEPTIONS)
if(NOT REAXMETAL_LAMMPS_MPI)
  target_include_directories(reaxmetal_probe_plugin PRIVATE "${REAXMETAL_LAMMPS_SOURCE_DIR}/STUBS")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_target_properties(reaxmetal_probe_plugin PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif(UNIX)
  set_target_properties(reaxmetal_probe_plugin PROPERTIES LINK_FLAGS "-rdynamic")
endif()

# ---- Adapter A1 (M2): real parse + extract + host checks, no physics. Uses the C++20 core library (parser).
add_library(reaxmetal_adapter_plugin MODULE adapter/adapter_plugin.cpp adapter/pair_reaxff_metal.cpp)
set_target_properties(reaxmetal_adapter_plugin PROPERTIES PREFIX "" SUFFIX ".so" OUTPUT_NAME "reaxmetaladapterplugin"
                      CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(reaxmetal_adapter_plugin PRIVATE "${REAXMETAL_LAMMPS_SOURCE_DIR}")
target_compile_definitions(reaxmetal_adapter_plugin PRIVATE LAMMPS_EXCEPTIONS)
if(NOT REAXMETAL_LAMMPS_STD_FORMAT)
  target_compile_options(reaxmetal_adapter_plugin PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/adapter/lammps_fmt_abi.h")
endif()
target_link_libraries(reaxmetal_adapter_plugin PRIVATE reaxmetal_core)
if(NOT REAXMETAL_LAMMPS_MPI)
  target_include_directories(reaxmetal_adapter_plugin PRIVATE "${REAXMETAL_LAMMPS_SOURCE_DIR}/STUBS")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_target_properties(reaxmetal_adapter_plugin PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif(UNIX)
  set_target_properties(reaxmetal_adapter_plugin PROPERTIES LINK_FLAGS "-rdynamic")
endif()

# In-LAMMPS integration probe (tests/lammps/run_probe.py): runs the stock pinned LAMMPS with the probe plugin.
if(REAXMETAL_LAMMPS_PREFIX AND EXISTS "${REAXMETAL_LAMMPS_PREFIX}/bin/lmp" AND REAXMETAL_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME lammps_probe COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_probe.py
           --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_probe_plugin>
           --potentials ${REAXMETAL_LAMMPS_SOURCE_DIR}/../potentials --work ${CMAKE_BINARY_DIR}/probe-work)
  set_tests_properties(lammps_probe PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${REAXMETAL_LAMMPS_PREFIX}/lib;DYLD_LIBRARY_PATH=${REAXMETAL_LAMMPS_PREFIX}/lib"
    FAIL_REGULAR_EXPRESSION "FAILED;UNEXPECTED;SUMMARY failures=[1-9]")
endif()

# Adapter A1 differential test (tests/lammps/run_a1.py): extract() parity, accept/reject parity, host checks and the
# explicit compute() refusal, against the stock pinned LAMMPS shared library. Needs a shared-library LAMMPS build
# with LAMMPS_EXCEPTIONS (tools/build_reference_matrix.sh stock-gcc) and REAXMETAL_FFIELD_DIR.
if(REAXMETAL_LAMMPS_PREFIX AND REAXMETAL_FFIELD_DIR AND EXISTS "${REAXMETAL_LAMMPS_PREFIX}/lib/liblammps.so" AND REAXMETAL_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME lammps_a1 COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_a1.py
           --lib ${REAXMETAL_LAMMPS_PREFIX}/lib/liblammps.so --plugin $<TARGET_FILE:reaxmetal_adapter_plugin>
           --ffield-dir ${REAXMETAL_FFIELD_DIR})
  set_tests_properties(lammps_a1 PROPERTIES
    ENVIRONMENT "LAMMPS_POTENTIALS=${REAXMETAL_FFIELD_DIR};LD_LIBRARY_PATH=${REAXMETAL_LAMMPS_PREFIX}/lib"
    FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
endif()
