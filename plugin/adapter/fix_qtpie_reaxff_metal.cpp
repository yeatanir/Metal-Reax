/* SPDX-License-Identifier: GPL-2.0-only
 * SPDX-FileCopyrightText: 2026 Anirban Phukan */
#include "fix_qtpie_reaxff_metal.h"

#include "neighbor.h"
#include "neigh_request.h"

using namespace LAMMPS_NS;

void FixQtpieReaxFFMetal::init()
{
  FixQtpieReaxFF::init();                                              // requests the plain half newton-off list as id 0
  neighbor->add_request(this, NeighConst::REQ_NEWTON_OFF | NeighConst::REQ_GHOST)->set_id(1);   // id 1: the list with ghost rows that QTPIE needs
}

void FixQtpieReaxFFMetal::init_list(int id, NeighList *ptr)
{
  if (id == 1) list = ptr;                                             // ignore the id 0 list
}

void FixQEqRelReaxFFMetal::init()
{
  FixQEqRelReaxFF::init();
  neighbor->add_request(this, NeighConst::REQ_NEWTON_OFF | NeighConst::REQ_GHOST)->set_id(1);
}

void FixQEqRelReaxFFMetal::init_list(int id, NeighList *ptr)
{
  if (id == 1) list = ptr;
}
