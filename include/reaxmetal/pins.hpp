// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
#include <string_view>

namespace reaxmetal {

// Upstream reference this engine is validated against. These constants are cross-checked against
// third_party/lammps/PIN.txt by the `pins` test; edit both together and log the change.
inline constexpr std::string_view kLammpsRepo = "https://github.com/lammps/lammps.git";
inline constexpr std::string_view kLammpsTag = "stable_30Sep2026";
inline constexpr std::string_view kLammpsTagObject = "752e990c8d4c30bb134f6c1b630a2ab056cbc087";
inline constexpr std::string_view kLammpsCommit = "8de817dd79bfe4525d5d39246a212d833e6dee07";
inline constexpr std::string_view kLammpsLicense = "GPL-2.0-only";

}  // namespace reaxmetal
