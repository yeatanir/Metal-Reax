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
else()
  message(STATUS "REAXMETAL_FFIELD_DIR not set: ffield_tables (PARSE-1) not registered")
endif()
