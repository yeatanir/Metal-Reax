/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan
 * Registration of pair_style reaxff/metal (adapter A1). Follows doc/src/Developer_plugins.rst of the pinned LAMMPS. */
#include "lammpsplugin.h"
#include "version.h"
#include "pair_reaxff_metal.h"

using namespace LAMMPS_NS;

static Pair *creator(LAMMPS *lmp) { return new PairReaxFFMetal(lmp); }

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;
  lammpsplugin_t plugin;
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "reaxff/metal";
  plugin.info = "ReaxMetal adapter A1 (parse, extract, host checks; no force backend yet)";
  plugin.author = "ReaxMetal project";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
