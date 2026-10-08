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
