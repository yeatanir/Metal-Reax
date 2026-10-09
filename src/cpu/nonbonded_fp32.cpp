// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// CPU-32 twin of the nonbonded terms: nonbonded.cpp compiled with Real = float (FP32-1).
#define RM_REAL float
#define RM_REAL_FLOAT 1
#define RM_NONBONDED_NAME compute_nonbonded_core_fp32
#include "nonbonded.cpp"
