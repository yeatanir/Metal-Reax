// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Force-included (-include) into the adapter when the target LAMMPS was built as C++17 (its default).
//
// LAMMPS' bundled fmt/format.h (src/fmt/format.h:40) switches fmt::format_args to std::format_args when <version> reports
// __cpp_lib_format (clang, GCC 14+). That changes the mangled signature of Error::_all/_one/_warning, so a C++20 plugin
// compiled with such a compiler fails to load into a C++17 LAMMPS ("undefined symbol ...Error8_warning...basic_format_args...").
// Including <version> first and withdrawing the macro makes the plugin select the same (fmt::) branch as that LAMMPS.
// If LAMMPS itself was built with C++20 and std::format, configure with -DREAXMETAL_LAMMPS_STD_FORMAT=ON instead.
#pragma once
#if __has_include(<version>)
#include <version>
#endif
#ifdef __cpp_lib_format
#undef __cpp_lib_format
#endif
