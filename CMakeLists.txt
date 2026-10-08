cmake_minimum_required(VERSION 3.24)
project(ReaxMetal VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

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
  src/core/energy_terms.cpp
  src/core/pins.cpp)
target_include_directories(reaxmetal_core PUBLIC include)
target_compile_features(reaxmetal_core PUBLIC cxx_std_20)
reaxmetal_apply_warnings(reaxmetal_core)
# Reference-quality FP: no contraction, no fast-math (docs/NUMERICAL_POLICY.md section 2).
reaxmetal_apply_strict_fp(reaxmetal_core)

if(REAXMETAL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
