// SPDX-License-Identifier: GPL-2.0-only
// Intentionally tiny: the pins are header constants; this TU keeps the library non-header-only so the
// build graph (and strict-FP flags) are exercised from M0 onward.
#include "reaxmetal/pins.hpp"

namespace reaxmetal {
static_assert(kLammpsCommit.size() == 40, "commit must be a full SHA-1");
static_assert(kLammpsTagObject.size() == 40, "tag object must be a full SHA-1");
}  // namespace reaxmetal
