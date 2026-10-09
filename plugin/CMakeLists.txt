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
if(REAXMETAL_LAMMPS_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)   # only for the mpi.h that the LAMMPS headers include
endif()
option(REAXMETAL_LAMMPS_STD_FORMAT "The target LAMMPS was built as C++20 and uses std::format for fmt (see adapter/lammps_fmt_abi.h)" OFF)

add_library(reaxmetal_probe_plugin MODULE probe/probe_plugin.cpp probe/pair_reaxff_metal_probe.cpp)
set_target_properties(reaxmetal_probe_plugin PROPERTIES PREFIX "" SUFFIX ".so" OUTPUT_NAME "reaxmetalprobeplugin"
                      CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(reaxmetal_probe_plugin PRIVATE "${REAXMETAL_LAMMPS_SOURCE_DIR}")
target_compile_definitions(reaxmetal_probe_plugin PRIVATE LAMMPS_EXCEPTIONS)
if(NOT REAXMETAL_LAMMPS_MPI)
  target_include_directories(reaxmetal_probe_plugin PRIVATE "${REAXMETAL_LAMMPS_SOURCE_DIR}/STUBS")
else()
  target_include_directories(reaxmetal_probe_plugin PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_target_properties(reaxmetal_probe_plugin PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif(UNIX)
  set_target_properties(reaxmetal_probe_plugin PROPERTIES LINK_FLAGS "-rdynamic")
endif()

# ---- Adapter A1 (M2): real parse + extract + host checks, no physics. Uses the C++20 core library (parser).
add_library(reaxmetal_adapter_plugin MODULE adapter/adapter_plugin.cpp adapter/pair_reaxff_metal.cpp adapter/fix_qeq_reaxff_metal.cpp adapter/fix_qtpie_reaxff_metal.cpp)
set_target_properties(reaxmetal_adapter_plugin PROPERTIES PREFIX "" SUFFIX ".so" OUTPUT_NAME "reaxmetaladapterplugin"
                      CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(reaxmetal_adapter_plugin PRIVATE "${REAXMETAL_LAMMPS_SOURCE_DIR}")
target_compile_definitions(reaxmetal_adapter_plugin PRIVATE LAMMPS_EXCEPTIONS)
if(NOT REAXMETAL_LAMMPS_STD_FORMAT)
  target_compile_options(reaxmetal_adapter_plugin PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/adapter/lammps_fmt_abi.h")
endif()
target_link_libraries(reaxmetal_adapter_plugin PRIVATE reaxmetal_metal)   # reaxmetal_metal = the Metal backend (REAXMETAL_ENABLE_METAL=ON) or its 'not available' stub; pulls reaxmetal_core
if(NOT REAXMETAL_LAMMPS_MPI)
  target_include_directories(reaxmetal_adapter_plugin PRIVATE "${REAXMETAL_LAMMPS_SOURCE_DIR}/STUBS")
else()
  target_include_directories(reaxmetal_adapter_plugin PRIVATE ${MPI_CXX_INCLUDE_DIRS})
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

# Adapter differential tests (tests/lammps/run_a1.py, run_a2.py): extract()/accept-reject parity and host checks (A1), and the ghost-native
# host view + far-list self-check inside LAMMPS against the 58 fixtures (A2). They drive the stock pinned LAMMPS shared library through its
# C interface, so they need a build with LAMMPS_EXCEPTIONS (tools/build_lammps_reference.sh) and REAXMETAL_FFIELD_DIR.
if(RM_LIBLAMMPS AND REAXMETAL_FFIELD_DIR AND REAXMETAL_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  foreach(_t a1 a2)
    add_test(NAME lammps_${_t} COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_${_t}.py
             --lib ${RM_LIBLAMMPS} --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR})
    set_tests_properties(lammps_${_t} PROPERTIES
      ENVIRONMENT "LAMMPS_POTENTIALS=${REAXMETAL_FFIELD_DIR};LD_LIBRARY_PATH=${REAXMETAL_LAMMPS_PREFIX}/lib;DYLD_LIBRARY_PATH=${REAXMETAL_LAMMPS_PREFIX}/lib"
      FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
  endforeach()
endif()

# INT-2 (M4/M6): in-LAMMPS A/B of stock `pair_style reaxff` vs `reaxff/metal backend cpu64` on the 58 fixtures: 14 energy slots, PE, charges, forces, pressure.
if(REAXMETAL_LAMMPS_PREFIX AND EXISTS "${REAXMETAL_LAMMPS_PREFIX}/bin/lmp" AND REAXMETAL_FFIELD_DIR AND REAXMETAL_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME lammps_int2 COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_int2.py
           --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend cpu64)
  set_tests_properties(lammps_int2 PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
  # PERATOM-1: per-atom energy and virial (compute pe/atom, stress/atom) equal stock on the 58 fixtures
  add_test(NAME lammps_peratom COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_peratom.py
           --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend cpu64)
  set_tests_properties(lammps_peratom PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
  if(REAXMETAL_LAMMPS_MPI)
    # multi-rank A/B against stock run with the same number of ranks (needs mpirun on PATH)
    foreach(_np 2 4)
      add_test(NAME lammps_int2_mpi${_np}_cpu64 COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_int2.py
               --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend cpu64 --np ${_np})
      set_tests_properties(lammps_int2_mpi${_np}_cpu64 PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
      if(REAXMETAL_ENABLE_METAL)
        add_test(NAME lammps_int2_mpi${_np}_metal COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_int2.py
                 --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend metal --gpu-qeq --np ${_np})
        set_tests_properties(lammps_int2_mpi${_np}_metal PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
      endif()
    endforeach()
  endif()
  # VAR-1: vdW type 2 and `enobonds no`, which no bundled fixture exercises
  add_test(NAME lammps_variants COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_variants.py
           --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend cpu64)
  set_tests_properties(lammps_variants PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
  # EEM-1: strict mode and charge verification of fix qeq/reaxff/metal
  add_test(NAME lammps_eem COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_eem.py
           --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} $<$<BOOL:${REAXMETAL_ENABLE_METAL}>:--metal>)
  set_tests_properties(lammps_eem PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
  # CHG-1: every ReaxFF charge model of LAMMPS (qeq, qeq/rel, qtpie, acks2, qeq/shielded, parameter file, group subset), with and without fix efield
  if(EXISTS "${REAXMETAL_LAMMPS_SOURCE_DIR}/../examples/reaxff/water/in.water.qeq")
    add_test(NAME lammps_charge_models COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_water_models.py
             --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --examples ${REAXMETAL_LAMMPS_SOURCE_DIR}/../examples/reaxff/water --backend cpu64)
    set_tests_properties(lammps_charge_models PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    if(REAXMETAL_ENABLE_METAL)
      add_test(NAME lammps_charge_models_metal COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_water_models.py
               --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --examples ${REAXMETAL_LAMMPS_SOURCE_DIR}/../examples/reaxff/water --backend metal --gpu-qeq)
      set_tests_properties(lammps_charge_models_metal PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    endif()
  endif()
  # ANA-1: fix reaxff/bonds, fix reaxff/species, compute reaxff/atom, compute SPEC/ATOM work unmodified with reaxff/metal
  if(EXISTS "${REAXMETAL_LAMMPS_SOURCE_DIR}/../examples/reaxff/CHO/data.CHO")
    add_test(NAME lammps_analysis COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_analysis.py
             --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --example ${REAXMETAL_LAMMPS_SOURCE_DIR}/../examples/reaxff/CHO --backend cpu64)
    set_tests_properties(lammps_analysis PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    if(REAXMETAL_ENABLE_METAL)
      add_test(NAME lammps_analysis_metal COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_analysis.py
               --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --example ${REAXMETAL_LAMMPS_SOURCE_DIR}/../examples/reaxff/CHO --backend metal)
      set_tests_properties(lammps_analysis_metal PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    endif()
  endif()
  # HYB-1: pair_style hybrid/overlay with reaxff/metal (charge-implicit ReaxFF + tabulated ZBL), stock vs plugin
  if(EXISTS "${REAXMETAL_LAMMPS_SOURCE_DIR}/../examples/reaxff/ci-reaxFF/in.ci-reax.CH")
    add_test(NAME lammps_hybrid COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_hybrid.py
             --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --example ${REAXMETAL_LAMMPS_SOURCE_DIR}/../examples/reaxff/ci-reaxFF --backend cpu64)
    set_tests_properties(lammps_hybrid PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    if(REAXMETAL_ENABLE_METAL)
      add_test(NAME lammps_hybrid_metal COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_hybrid.py
               --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --example ${REAXMETAL_LAMMPS_SOURCE_DIR}/../examples/reaxff/ci-reaxFF --backend metal)
      set_tests_properties(lammps_hybrid_metal PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    endif()
  endif()
  add_test(NAME lammps_min1_cpu64 COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_min.py
           --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend cpu64)
  set_tests_properties(lammps_min1_cpu64 PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
  if(REAXMETAL_ENABLE_METAL)
    # INT-2 with the Metal backend (FP32 on the GPU): judged by the owner-set C3 criteria
    add_test(NAME lammps_int2_metal COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_int2.py
             --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend metal)
    set_tests_properties(lammps_int2_metal PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    # the same with fix qeq/reaxff/metal (EEM matrix and matvec on the GPU): charges are compared with the stock fix's under the C3 charge limit
    add_test(NAME lammps_int2_metal_qeq COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_int2.py
             --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend metal --gpu-qeq)
    set_tests_properties(lammps_int2_metal_qeq PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    add_test(NAME lammps_variants_metal COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_variants.py
             --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend metal)
    set_tests_properties(lammps_variants_metal PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    add_test(NAME lammps_peratom_metal COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_peratom.py
             --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend metal)
    set_tests_properties(lammps_peratom_metal PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    # the same with the GPU-resident EEM solve (double-single residual + FP32 device CG): charges must equal the stock fix's to ~1e-10
    add_test(NAME lammps_int2_metal_qeq_resident COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_int2.py
             --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend metal --gpu-qeq --qeq-extra resident)
    set_tests_properties(lammps_int2_metal_qeq_resident PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    # mixed mode: bond-order terms in double on the host, non-bonded + GPU QEq on the device
    add_test(NAME lammps_int2_metal_mixed COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_int2.py
             --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} "--backend=metal bonded cpu64" --gpu-qeq --qeq-extra resident)
    set_tests_properties(lammps_int2_metal_mixed PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    # DENSE-1: dense random carbon (packed, minimised, hot): nearly collinear triples that broke the float acos angle; forces and energy vs stock
    add_test(NAME lammps_dense_carbon COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_dense_carbon.py
             --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield ${REAXMETAL_FFIELD_DIR}/ffield.reax.cho --n 600 --seeds 2)
    set_tests_properties(lammps_dense_carbon PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    # MIN-1: LAMMPS `minimize` reaches the stock minimum (energies, displacement) on the perturbed fixtures, with the GPU charge fix
    add_test(NAME lammps_min1_metal COMMAND ${Python3_EXECUTABLE} -I ${PROJECT_SOURCE_DIR}/tests/lammps/run_min.py
             --lmp ${REAXMETAL_LAMMPS_PREFIX}/bin/lmp --plugin $<TARGET_FILE:reaxmetal_adapter_plugin> --ffield-dir ${REAXMETAL_FFIELD_DIR} --backend metal --gpu-qeq)
    set_tests_properties(lammps_min1_metal PROPERTIES FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
  endif()
endif()
