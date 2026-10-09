// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Command-line driver for the CPU-64 bonded core (used by tests/python/test_bonded_fixtures.py):
//   reaxmetal_energy_tool --ffield FILE [--lgvdw] [--noenobonds] [--bond R] [--hbond R] [--shell R] [--grad] CASE.txt
// CASE.txt: same format as reaxmetal_neighbor_tool ("origin/a/b/c/periodic/atoms N" then "tag type x y z").
// Output, one item per line: "e_bond v", "e_lp v", "e_ov v", "e_un v", "bonds n" and with --grad "grad tag gx gy gz" (dE/dx folded onto owners).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "reaxmetal/bonded.hpp"
#include "reaxmetal/bonded_device.hpp"
#include "reaxmetal/metal_backend.hpp"
#include "reaxmetal/nonbonded.hpp"
#include "reaxmetal/nonbonded_device.hpp"

using namespace reaxmetal;

int main(int argc, char** argv) {
  std::string ffield, casefile;
  ControlParams ctl;
  BondedOptions bo;
  double shell = -1.0;
  bool grad = false, lgvdw = false;
  std::string backend = "cpu64";    // nonbonded backend: cpu64 | metal
  std::string grad_terms = "all";  // gradient output: bonded | nonbonded | all
  std::vector<double> q;
  std::vector<std::string> elements;  // LAMMPS type t (1-based) <-> element elements[t-1], mapped through the ffield (pair_coeff rule)
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "missing value after %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
    if (a == "--ffield") ffield = val();
    else if (a == "--bond") ctl.bond_cut = std::stod(val());
    else if (a == "--hbond") ctl.hbond_cut = std::stod(val());
    else if (a == "--shell") shell = std::stod(val());
    else if (a == "--grad") grad = true;
    else if (a == "--grad-terms") grad_terms = val();
    else if (a == "--backend") backend = val();
    else if (a == "--lgvdw") lgvdw = true;
    else if (a == "--noenobonds") bo.enobonds = false;
    else if (a == "--elements") { std::istringstream es(val()); std::string e; while (std::getline(es, e, ',')) elements.push_back(e); }
    else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    else casefile = a;
  }
  try {
    if (ffield.empty() || casefile.empty()) { std::fprintf(stderr, "usage: reaxmetal_energy_tool --ffield F [--lgvdw] [--noenobonds] [--grad] CASE.txt\n"); return 2; }
    FfieldOptions fo; fo.lgvdw = lgvdw;
    const ForceField ff = read_force_field_file(ffield, fo);
    NeighborCutoffs cut;
    cut.nonb = ff.file_control().nonb_cut; cut.bond = ctl.bond_cut; cut.hbond = ctl.hbond_cut;
    std::ifstream in(casefile);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", casefile.c_str()); return 2; }
    Box box;
    std::vector<double> x;
    std::vector<int> type;
    std::vector<std::int64_t> tag;
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream ss(line);
      std::string key;
      if (!(ss >> key)) continue;
      if (key == "origin") ss >> box.origin[0] >> box.origin[1] >> box.origin[2];
      else if (key == "a" || key == "b" || key == "c") { auto& v = box.vectors[static_cast<std::size_t>(key[0] - 'a')]; ss >> v[0] >> v[1] >> v[2]; }
      else if (key == "periodic") { int p[3]; ss >> p[0] >> p[1] >> p[2]; for (std::size_t d = 0; d < 3; ++d) box.periodic[d] = p[d] != 0; }
      else if (key == "atoms") {
        std::size_t natoms = 0; ss >> natoms;
        for (std::size_t k = 0; k < natoms; ++k) {
          std::int64_t t; int ty; double r[3];
          if (!(in >> t >> ty >> r[0] >> r[1] >> r[2])) { std::fprintf(stderr, "truncated atom list\n"); return 2; }
          if (!elements.empty()) {
            if (ty < 1 || static_cast<std::size_t>(ty) > elements.size()) { std::fprintf(stderr, "type %d outside --elements\n", ty); return 2; }
            const auto m = ff.match_element(elements[static_cast<std::size_t>(ty - 1)]);
            if (m.size() != 1) { std::fprintf(stderr, "element %s not unique in ffield\n", elements[static_cast<std::size_t>(ty - 1)].c_str()); return 2; }
            ty = m[0];
          }
          tag.push_back(t); type.push_back(ty); x.insert(x.end(), r, r + 3);
        }
        // optional "charges" section after the atom list: one value per owned atom
        std::string tok;
        if (in >> tok && tok == "charges") { double qv; while (q.size() < natoms && (in >> qv)) q.push_back(qv); }
        break;
      }
    }
    ExpandOptions eo;
    eo.shell = shell > 0 ? shell : cut.required_shell();
    const AtomSet a = expand_images(box, x, type, tag, eo);
    a.validate(box);
    const FarList f = build_far_list(a, cut);
    double bonded_gpu_ms = -1;
    BondedResult r;
    if (backend == "metal") {
      mtl::Context ctx;
      const auto bin = make_bonded_device_input(ff, ctl, a, box, bo);
      const auto bout = ctx.bonded(bin);
      bonded_gpu_ms = 1e3 * ctx.last_gpu_seconds();
      r = finish_bonded(bout, a.nall());
      r.stats.bonds = bout.bond_cap;   // (not the bond count: the capacity that was needed)
    } else {
      r = backend == "cpu32" ? compute_bonded_core_fp32(ff, ctl, a, f, bo) : compute_bonded_core(ff, ctl, a, f, bo);
    }
    if (bonded_gpu_ms >= 0) std::printf("bonded_gpu_ms %.6g\nbond_cap %zu\n", bonded_gpu_ms, r.stats.bonds);
    std::printf("e_bond %.17g\ne_lp %.17g\ne_ov %.17g\ne_un %.17g\nbonds %zu\n", r.e[EnergyTerm::Bond], r.e[EnergyTerm::LonePair],
                r.e[EnergyTerm::Over], r.e[EnergyTerm::Under], r.stats.bonds);
    std::printf("e_ang %.17g\ne_pen %.17g\ne_coa %.17g\ne_tor %.17g\ne_con %.17g\ne_hb %.17g\n", r.e[EnergyTerm::Valence], r.e[EnergyTerm::Penalty],
                r.e[EnergyTerm::Coalition], r.e[EnergyTerm::Torsion], r.e[EnergyTerm::Conjugation], r.e[EnergyTerm::HBond]);
    NonbondedResult nb;
    if (q.size() == a.nlocal) {
      NonbondedOptions no; no.lgvdw = lgvdw;
      double gpu_ms = -1;
      if (backend == "metal") {
        mtl::Context ctx;
        const auto din = make_nonbonded_device_input(ff, cut, a, box, q, no, [&](const DeviceListInput& l) { return ctx.far_rows(l); });
        const auto dout = ctx.nonbonded(din);
        gpu_ms = 1e3 * ctx.last_gpu_seconds();
        nb = finish_nonbonded(ff, a, q, dout);
      } else {
        nb = backend == "cpu32" ? compute_nonbonded_core_fp32(ff, cut, a, f, q, no) : compute_nonbonded_core(ff, cut, a, f, q, no);
      }
      if (gpu_ms >= 0) std::printf("gpu_ms %.6g\n", gpu_ms);
      std::printf("e_vdW %.17g\ne_ele %.17g\ne_pol %.17g\npairs %zu\n", nb.e[EnergyTerm::VdW], nb.e[EnergyTerm::Coulomb], nb.e[EnergyTerm::Polarization], nb.pairs);
    }
    if (grad) {
      std::vector<double> g(3 * a.nlocal, 0.0);
      for (std::size_t i = 0; i < a.nall(); ++i)
        for (std::size_t c = 0; c < 3; ++c) g[3 * static_cast<std::size_t>(a.owner[i]) + c] += (grad_terms == "nonbonded" ? 0.0 : r.grad[3 * i + c]) + (nb.grad.empty() || grad_terms == "bonded" ? 0.0 : nb.grad[3 * i + c]);
      for (std::size_t i = 0; i < a.nlocal; ++i) std::printf("grad %lld %.17g %.17g %.17g\n", static_cast<long long>(a.tag[i]), g[3 * i], g[3 * i + 1], g[3 * i + 2]);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
