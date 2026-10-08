# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
function(reaxmetal_add_test name)
  add_executable(test_${name} test_${name}.cpp)
  target_link_libraries(test_${name} PRIVATE reaxmetal_core)
  target_compile_definitions(test_${name} PRIVATE REAXMETAL_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  reaxmetal_apply_warnings(test_${name})
  add_test(NAME ${name} COMMAND test_${name})
endfunction()

reaxmetal_add_test(pins)
reaxmetal_add_test(capabilities)
reaxmetal_add_test(energy_terms)
reaxmetal_add_test(docs_sync)
reaxmetal_add_test(license_headers)
reaxmetal_add_test(compat_flags)
reaxmetal_add_test(ffield_parser)
reaxmetal_add_test(pair_settings)
reaxmetal_add_test(neighbor)
reaxmetal_add_test(terms)

# The Metal kernels executed as C++ (EMULATION: evidence about kernel logic, not about Metal; see the file header)
add_executable(test_metal_emulation test_metal_emulation.cpp)
target_link_libraries(test_metal_emulation PRIVATE reaxmetal_metal)
target_include_directories(test_metal_emulation PRIVATE ${PROJECT_SOURCE_DIR}/src/metal/shaders ${CMAKE_CURRENT_SOURCE_DIR}/metal_shim ${PROJECT_SOURCE_DIR}/tools)
target_compile_definitions(test_metal_emulation PRIVATE REAXMETAL_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
reaxmetal_apply_warnings(test_metal_emulation)
reaxmetal_apply_strict_fp(test_metal_emulation)
add_test(NAME metal_emulation COMMAND test_metal_emulation)
# The Apple-machine check tool, run end to end against a FAKE backend that executes the kernels on the CPU: tests the tool itself
add_executable(metal_check_emulated metal_check_emulated.cpp ${PROJECT_SOURCE_DIR}/tools/metal_check.cpp ${PROJECT_SOURCE_DIR}/src/metal/metal_source.cpp)
target_link_libraries(metal_check_emulated PRIVATE reaxmetal_core)
target_include_directories(metal_check_emulated PRIVATE ${PROJECT_SOURCE_DIR}/src/metal/shaders ${CMAKE_CURRENT_SOURCE_DIR}/metal_shim ${PROJECT_SOURCE_DIR}/tools ${CMAKE_BINARY_DIR}/generated)
reaxmetal_apply_warnings(metal_check_emulated)
reaxmetal_apply_strict_fp(metal_check_emulated)
foreach(_s 1 2 3)
  add_test(NAME metal_check_emulated_step${_s} COMMAND metal_check_emulated --step ${_s} --quick)
  set_tests_properties(metal_check_emulated_step${_s} PROPERTIES PASS_REGULAR_EXPRESSION "STEP ${_s} RESULT: PASS" FAIL_REGULAR_EXPRESSION "FAIL")
endforeach()

# Syntax/type check of the Objective-C++ backend against hand-written stub headers (no Apple SDK needed). Weak evidence, see the script.
find_program(RM_CLANGXX NAMES clang++ clang++-18 clang++-17 clang++-16 clang++-15)
if(RM_CLANGXX AND NOT APPLE)
  add_test(NAME objcxx_syntax COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_objcxx_syntax.sh ${RM_CLANGXX})
  set_tests_properties(objcxx_syntax PROPERTIES SKIP_RETURN_CODE 77)
endif()

# ---- M1: frozen artefacts and the reference-harness self-tests (no LAMMPS needed) --------------------------------------
add_test(NAME tolerances_frozen
  COMMAND ${CMAKE_COMMAND} -DMANIFEST=${PROJECT_SOURCE_DIR}/tolerances/TOLERANCES.sha256 -DBASE=${PROJECT_SOURCE_DIR}/tolerances
          -DEXPECT_MIN=1 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_hash_file.cmake)
add_test(NAME fixtures_frozen
  COMMAND ${CMAKE_COMMAND} -DMANIFEST=${PROJECT_SOURCE_DIR}/tests/fixtures/FIXTURES.sha256 -DBASE=${PROJECT_SOURCE_DIR}/tests/fixtures
          -DEXPECT_MIN=100 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_hash_file.cmake)
add_test(NAME patch_frozen
  COMMAND ${CMAKE_COMMAND} -DMANIFEST=${PROJECT_SOURCE_DIR}/third_party/lammps/patches/PATCHES.sha256 -DBASE=${PROJECT_SOURCE_DIR}/third_party/lammps/patches
          -DEXPECT_MIN=1 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_hash_file.cmake)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy" RESULT_VARIABLE _np_rc OUTPUT_QUIET ERROR_QUIET)
  if(_np_rc EQUAL 0)
    add_test(NAME reaxref_selftest COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/python/test_reaxref.py)
  else()
    message(STATUS "numpy not found: reaxref_selftest not registered")
  endif()
endif()

# PARSE-1 against the 11 bundled force fields: needs the files from the pinned LAMMPS tree (not committed, ADR-017)
if(NOT REAXMETAL_FFIELD_DIR AND REAXMETAL_LAMMPS_SOURCE_DIR AND EXISTS "${REAXMETAL_LAMMPS_SOURCE_DIR}/../potentials/ffield.reax.cho")
  set(REAXMETAL_FFIELD_DIR "${REAXMETAL_LAMMPS_SOURCE_DIR}/../potentials")
endif()
if(REAXMETAL_FFIELD_DIR AND EXISTS "${REAXMETAL_FFIELD_DIR}/ffield.reax.cho")
  add_executable(test_ffield_tables test_ffield_tables.cpp)
  target_link_libraries(test_ffield_tables PRIVATE reaxmetal_core)
  reaxmetal_apply_warnings(test_ffield_tables)
  add_test(NAME ffield_tables COMMAND test_ffield_tables ${REAXMETAL_FFIELD_DIR} ${PROJECT_SOURCE_DIR}/tests/fixtures/ffield_tables.sha256)
  # NBR-3: engine pair enumeration vs the interaction tallies LAMMPS recorded in M1 (needs the ffields for nonb_cut) and
  # NBR-2: engine ghost set vs LAMMPS' ghost set (needs the stock LAMMPS shared library)
  if(Python3_FOUND)
    add_test(NAME neighbor_fixtures COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/python/test_neighbor_fixtures.py
             --tool $<TARGET_FILE:reaxmetal_neighbor_tool> --ffield-dir ${REAXMETAL_FFIELD_DIR})
    set_tests_properties(neighbor_fixtures PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    # BOND-1 (M4 phase A): CPU-64 bond/lone-pair/over/under energies vs the pinned-LAMMPS values of M1 + finite-difference check of their gradient
    add_test(NAME bonded_fixtures COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/python/test_bonded_fixtures.py
             --tool $<TARGET_FILE:reaxmetal_energy_tool> --ffield-dir ${REAXMETAL_FFIELD_DIR} --fd)
    set_tests_properties(bonded_fixtures PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    add_test(NAME nonbonded_fixtures COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/python/test_nonbonded_fixtures.py
             --tool $<TARGET_FILE:reaxmetal_energy_tool> --ffield-dir ${REAXMETAL_FFIELD_DIR} --fd)
    set_tests_properties(nonbonded_fixtures PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    if(RM_LIBLAMMPS)
      add_test(NAME lammps_ghosts COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/lammps/run_ghosts.py
               --lib ${RM_LIBLAMMPS} --tool $<TARGET_FILE:reaxmetal_neighbor_tool> --ffield-dir ${REAXMETAL_FFIELD_DIR})
      set_tests_properties(lammps_ghosts PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL"
        ENVIRONMENT "LD_LIBRARY_PATH=${REAXMETAL_LAMMPS_PREFIX}/lib;DYLD_LIBRARY_PATH=${REAXMETAL_LAMMPS_PREFIX}/lib")
    endif()
  endif()
else()
  message(STATUS "REAXMETAL_FFIELD_DIR not set: ffield_tables (PARSE-1), neighbor_fixtures and lammps_ghosts not registered")
endif()
