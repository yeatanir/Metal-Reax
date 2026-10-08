// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// pair_style reaxff/metal argument parsing (syntax of pinned LAMMPS pair_style reaxff) and capability gating.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "reaxmetal/capabilities.hpp"
#include "reaxmetal/pair_settings.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

static PairSettings parse(std::vector<std::string> a) { return parse_pair_style_args(a); }

int main() {
  {
    const PairSettings p = parse({"NULL"});
    RM_CHECK(p.control_file.empty() && p.checkqeq && !p.lgvdw && p.enobonds && p.tabulate == 0 && p.notices.empty());
    RM_CHECK(p.control.bond_cut == 5.0 && p.control.hbond_cut == 7.5 && p.control.thb_cut == 0.001 && p.control.thb_cutsq == 0.00001);
  }
  {
    const PairSettings p = parse({"NULL", "checkqeq", "no", "enobonds", "off", "lgvdw", "true", "tabulate", "0"});
    RM_CHECK(!p.checkqeq && !p.enobonds && p.lgvdw && p.tabulate == 0);
    const PairSettings q = parse({"NULL", "safezone", "1.5", "mincap", "100", "minhbonds", "30", "list/blocking", "yes"});
    RM_CHECK(q.safezone == 1.5 && q.mincap == 100 && q.minhbonds == 30 && q.list_blocking && q.notices.size() == 4);   // Ignored options leave a notice each
  }
  // Deferred: spline tabulation changes the numbers (opt.tabulate)
  RM_EXPECT_THROW(parse({"NULL", "tabulate", "1"}), UnsupportedFeatureError);
  RM_EXPECT_THROW(parse({"NULL", "tabulate", "25"}), UnsupportedFeatureError);
  // illegal syntax
  RM_EXPECT_THROW(parse({}), FfieldError);
  RM_EXPECT_THROW(parse({"NULL", "checkqeq"}), FfieldError);
  RM_EXPECT_THROW(parse({"NULL", "checkqeq", "maybe"}), FfieldError);
  RM_EXPECT_THROW(parse({"NULL", "frobnicate", "yes"}), FfieldError);
  RM_EXPECT_THROW(parse({"NULL", "safezone", "-1"}), FfieldError);
  RM_EXPECT_THROW(parse({"NULL", "mincap", "-3"}), FfieldError);
  RM_EXPECT_THROW(parse({"NULL", "mincap", "2.5"}), FfieldError);
  RM_EXPECT_THROW(parse({"NULL", "tabulate", "-1"}), FfieldError);
  RM_EXPECT_THROW(parse({"/nonexistent/control.reax"}), FfieldError);
  // control file: cutoffs, inactive keyword notice, and a Deferred tabulation request
  const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/reaxmetal_test_control.txt";
  {
    std::ofstream o(path);
    o << "nbrhood_cutoff 4.0\nhbond_cutoff 6.0\nthb_cutoff 0.002\nsimulation_name x\n";
  }
  {
    const PairSettings p = parse({path, "lgvdw", "yes"});
    RM_CHECK(p.control_file == path && p.control.bond_cut == 4.0 && p.control.hbond_cut == 6.0 && p.control.thb_cut == 0.002 && p.lgvdw);
    RM_CHECK(p.notices.size() == 1 && p.notices[0].find("simulation_name") != std::string::npos);
  }
  {
    std::ofstream o(path);
    o << "tabulate_long_range 10\n";
  }
  RM_EXPECT_THROW(parse({path}), UnsupportedFeatureError);
  std::remove(path.c_str());
  return rmtest::finish("pair_settings");
}
