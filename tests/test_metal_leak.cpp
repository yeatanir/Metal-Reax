// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Regression: Metal entry points called from a plain C++ loop (no autorelease pool, like a LAMMPS run) must not retain their buffers. Before the
// pool was added, every call leaked all of its buffers: a 648-atom MD run grew by gigabytes per thousand steps and the process was killed.
#include <cstdio>
#include <vector>

#include <mach/mach.h>

#include "reaxmetal/metal_backend.hpp"

static double rss_mb() {
  mach_task_basic_info i;
  mach_msg_type_number_t c = MACH_TASK_BASIC_INFO_COUNT;
  task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&i), &c);
  return static_cast<double>(i.resident_size) / 1e6;
}

int main() {
  reaxmetal::mtl::Context ctx;
  const std::vector<float> x(1 << 20, 1.0f), y(1 << 20, 2.0f);
  ctx.saxpy(2.0f, x, y);
  const double before = rss_mb();
  for (int i = 0; i < 200; ++i) ctx.saxpy(2.0f, x, y);   // 200 x 8 MB of buffers: a leak shows as > 1.6 GB
  const double grew = rss_mb() - before;
  std::printf("RSS grew by %.1f MB over 200 calls\n", grew);
  if (grew > 200.0) { std::printf("FAIL: Metal buffers are leaking\n"); return 1; }
  std::printf("PASS\n");
  return 0;
}
