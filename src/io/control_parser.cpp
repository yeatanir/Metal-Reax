// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Independent reader for the ReaxFF control file (behaviour of pinned LAMMPS src/REAXFF/reaxff_control.cpp; ENGINE_SPEC 2.6).
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_set>

#include "reaxmetal/forcefield.hpp"
#include "text_reader.hpp"

namespace reaxmetal {

ControlParams parse_control_text(std::string_view text, std::string_view name, std::vector<std::string>* warnings) {
  // keywords the reference accepts and ignores (reaxff_control.cpp:36-47)
  static const std::unordered_set<std::string> inactive = {
      "ensemble_type", "nsteps", "dt", "proc_by_dim", "random_vel", "restart_format", "restart_freq", "reposition_atoms",
      "restrict_bonds", "remove_CoM_vel", "debug_level", "reneighbor", "vlist_buffer", "ghost_cutoff", "qeq_freq", "q_err",
      "ilu_refactor", "ilu_droptol", "temp_init", "temp_final", "t_mass", "t_mode", "t_rate", "t_freq", "pressure", "p_mass",
      "pt_mass", "compress", "press_mode", "geo_format", "traj_compress", "traj_method", "molecular_analysis", "ignore",
      "dipole_anal", "freq_dipole_anal", "diffusion_coef", "freq_diffusion_coef", "restrict_type", "traj_title",
      "simulation_name", "energy_update_freq", "atom_info", "atom_velocities", "atom_forces", "bond_info", "angle_info"};
  ControlParams c;  // defaults: reaxff_control.cpp:73-79
  detail::Reader rd(text, name);
  try {
    while (true) {
      detail::Tokens v = rd.next_values();  // blank lines skipped; EOF ends the file
      const std::string keyword(v.next_string());
      if (!v.has_next()) throw FfieldError(rd.where() + ": no value(s) for control parameter: " + keyword);
      if (inactive.count(keyword)) {
        if (warnings) warnings->push_back("Ignoring inactive control parameter: " + keyword);
      } else if (keyword == "nbrhood_cutoff") {
        c.bond_cut = v.next_double();
      } else if (keyword == "bond_graph_cutoff") {
        c.bg_cut = v.next_double();
      } else if (keyword == "thb_cutoff") {
        c.thb_cut = v.next_double();
      } else if (keyword == "thb_cutoff_sq") {
        c.thb_cutsq = v.next_double();
      } else if (keyword == "hbond_cutoff") {
        c.hbond_cut = v.next_double();
      } else if (keyword == "tabulate_long_range") {
        c.tabulate = v.next_int();
      } else if (keyword == "write_freq") {
        if (v.next_int() > 0 && warnings)
          warnings->push_back("Support for writing native trajectories has been removed after LAMMPS version 8 April 2021");
      } else {
        throw FfieldError(rd.where() + ": unknown parameter " + keyword + " in control file");
      }
    }
  } catch (detail::EofReached&) {
  }
  return c;
}

ControlParams read_control_file(const std::string& path, std::vector<std::string>* warnings) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw FfieldError("cannot open ReaxFF control file " + path);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return parse_control_text(text, path, warnings);
}

}  // namespace reaxmetal
