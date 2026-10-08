/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan
 * SPIKE / SCAFFOLDING (M0.5). Registration follows doc/src/Developer_plugins.rst (pinned LAMMPS). */
#include "lammpsplugin.h"
#include "version.h"
#include "pair_reaxff_metal_probe.h"

using namespace LAMMPS_NS;

static Pair *probecreator(LAMMPS *lmp) { return new PairReaxFFMetalProbe(lmp); }

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;
  lammpsplugin_t plugin;
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "reaxff/metal";
  plugin.info = "ReaxMetal host-contract PROBE (no physics) v0";
  plugin.author = "ReaxMetal project";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &probecreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
