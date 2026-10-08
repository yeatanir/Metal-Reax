// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
#include <string>
#include <string_view>
namespace reaxmetal {
// FIPS 180-4 SHA-256, lowercase hex. Used to hash canonical table dumps (PARSE-1 goldens).
std::string sha256_hex(std::string_view data);
}  // namespace reaxmetal
