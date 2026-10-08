# Compiler flag helpers. Kept in one place so NUMERICAL_POLICY.md can point at a single source of truth.

function(reaxmetal_apply_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  endif()
endfunction()

# Strict floating point for the CPU FP64 reference: forbid FMA contraction and fast-math so results do
# not depend on the optimisation level. (LAMMPS itself is *not* built this way, so bitwise equality with
# LAMMPS is not expected -- see NUMERICAL_POLICY.md section 2.)
function(reaxmetal_apply_strict_fp target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${target} PRIVATE -ffp-contract=off -fno-fast-math)
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /fp:strict)
  endif()
endfunction()
