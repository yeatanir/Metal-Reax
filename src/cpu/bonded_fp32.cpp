// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// CPU-32 twin of the bonded terms: bonded.cpp compiled with Real = float (calibrates the FP32 error envelope, FP32-1).
#define RM_REAL float
#define RM_REAL_FLOAT 1
#define RM_BONDED_NAME compute_bonded_core_fp32
#include "bonded.cpp"
