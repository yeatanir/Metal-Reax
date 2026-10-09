// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// Decision census (FP32-1 / NUMERICAL_POLICY 5.3): per owned atom, how many discrete decisions of each kind a backend took. Hard thresholds of the force
// field (bond-order cut-off, three-body and four-body gates, hydrogen-bond gate, non-bonded cut-off) are where single precision can take a different
// decision than double precision; comparing these counts atom by atom between CPU-64, the CPU-32 twin and Metal lists every such flip.
#include <cstdint>
#include <vector>

namespace reaxmetal {

struct DecisionCensus {
  std::vector<std::int32_t> bonds;        // bonds of the owned atom (bond-order cut-off decision)
  std::vector<std::int32_t> angle_sets;   // (i, k) bond pairs of the owned centre j that pass the three-body gates (thb_cut, thb_cutsq)
  std::vector<std::int32_t> torsions;     // (i, k, l) chains of the owned central atom j that pass the four-body gates
  std::vector<std::int32_t> hbonds;       // donor-hydrogen-acceptor triples of the owned hydrogen j that are evaluated (BO >= 0.01, parameters set)
  std::vector<std::int32_t> nonbonded;    // counted non-bonded pairs of the owned row i (cut-off decision)
  std::vector<std::int32_t> sbo_region;   // branch of the valence-angle SBO function of the owned atom: 0 SBO <= 0, 1 (0,1], 2 (1,2), 3 >= 2
  std::vector<std::int32_t> lp_trunc;     // (int)(Delta_e / 2) of the owned atom: the truncation that makes the lone-pair number jump at integer Delta_e / 2
};

}  // namespace reaxmetal
