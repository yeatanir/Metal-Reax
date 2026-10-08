// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// CPU emulation shim: lets the .metal sources of src/metal/shaders be compiled as ordinary C++ and executed one "thread" at
// a time. It checks the *logic* (indexing, bounds, ordering, arithmetic order) of kernels written in the subset of MSL used
// there. It does NOT check Metal semantics: address-space rules, the Metal compiler's own diagnostics, threadgroup behaviour,
// GPU float behaviour (denormals, fused multiply-add). Everything it proves is "emulated", never "ran on Metal".
//
// Usage (one translation unit):  #include <all std headers first>  #include "metal_shim.hpp"  #include "<types.h>"  #include "<kernels.metal>"
//                                #include "metal_shim_end.hpp"
#pragma once
#include <cmath>
#include <cstdint>

using uint = std::uint32_t;
using uchar = std::uint8_t;
using ushort = std::uint16_t;
namespace metal {}
// Metal's math functions are global; make the C++ library's float overloads visible under the same unqualified names
using std::acos;
using std::atan2;
using std::cos;
using std::exp;
using std::fabs;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;

// MSL qualifiers and attributes disappear; `[[buffer(0)]]` becomes the legal empty attribute `[[ ]]`.
#define kernel
#define device
#define constant
#define threadgroup
#define thread
#define buffer(n)
#define thread_position_in_grid
