// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// eReaxFF engine against the exact external reference (plan tasks 3.3 / 3.4, final check): all energy terms and the total gradient of a
// non-periodic eReaxFF case with fixed charges (MOLCHARGE), as printed by the rebuilt legacy executable at full precision.
// Usage: test_ereaxff_reference <ffield> <geo> <electron taper> <ereac_export.out> [tolerance] [stage_export.out|-] [periodic|np] [skip-list]
// The expected energy terms (TERMS line, kcal/mol: eb ea elp emol ev+epen ecoa ehb et eco ew ep ech efi eelect) and the gradient are read from the export
// (STEP 0). An optional comma-separated skip list (e.g. torsion,torsion_conj) names terms that are printed but not checked (known stock-vs-legacy residuals).
// 'periodic' takes the cell from the CRYSTX line of the geo file (orthorhombic, periodic in all three directions) instead of a free cluster.
// The optional stage export has 'STAGE name' blocks of cumulative 'SGRAD i gx gy gz' lines (after electron, lonpar, covbon, ovcor, valang, torang, hbond, nonbon);
// consecutive differences give the per-routine gradients, compared here with the electron, bonded and nonbonded gradients of the engine.
// Files are not stored in this repository (parameter / geometry / reference output of the Jamil-target system).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "reaxmetal/bonded.hpp"
#include "reaxmetal/electron.hpp"
#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/nonbonded.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: %s ffield geo taper ereac_export.out [tol]\n", argv[0]); return 2; }
  FfieldOptions fo; fo.ereaxff = true; fo.electron_taper = std::atof(argv[3]);
  const double tol = argc > 5 ? std::atof(argv[5]) : 1e-9;
  const ForceField ff = read_force_field_file(argv[1], fo);
  std::vector<double> x, q; std::vector<int> type; std::vector<std::int64_t> tag; double cell[3] = {0, 0, 0}, ang[3] = {90, 90, 90};
  const bool periodic = argc > 7 && std::string(argv[7]) == "periodic";
  const std::string skip = argc > 8 ? std::string(",") + argv[8] + "," : std::string();
  {
    std::ifstream in(argv[2]); std::string line; std::vector<std::array<double, 3>> groups;
    while (std::getline(in, line)) {
      std::istringstream ls(line); std::string rec; ls >> rec;
      if (rec == "HETATM") { int idx; std::string name, sym; double px, py, pz; ls >> idx >> name >> sym >> px >> py >> pz;
        const auto m = ff.match_element(name); RM_CHECK(m.size() == 1); if (m.size() != 1) return 1;
        type.push_back(m[0]); tag.push_back(idx); x.insert(x.end(), {px, py, pz}); }
      else if (rec == "CRYSTX") { ls >> cell[0] >> cell[1] >> cell[2] >> ang[0] >> ang[1] >> ang[2]; }
      else if (rec == "MOLCHARGE") { double a, b, c; ls >> a >> b >> c; groups.push_back({a, b, c}); }
    }
    q.assign(type.size(), 0.0);
    for (const auto& g : groups) for (int i = static_cast<int>(g[0]); i <= static_cast<int>(g[1]); ++i) q[static_cast<std::size_t>(i - 1)] = g[2] / (g[1] - g[0] + 1.0);
  }
  if (periodic) RM_CHECK(cell[0] > 0 && cell[1] > 0 && cell[2] > 0);
  Box box = Box::orthogonal({-100, -100, -100}, {200, 200, 200}, {false, false, false});
  if (periodic) {
    // legacy cell (reac.f:45-48, 508-531): pi = 3.14159265d0 (truncated), lattice vectors (tm11,tm21,tm31), (0,tm22,tm32), (0,0,tm33). A nominal 90 degree
    // cell therefore has off-axis components of ~2e-8 A, which changes periodic energies at the 1e-10 relative level.
    const double pil = 3.14159265, dgrrdn = pil / 180.0;
    const double sa = std::sin(ang[0] * dgrrdn), ca = std::cos(ang[0] * dgrrdn), sb = std::sin(ang[1] * dgrrdn), cb = std::cos(ang[1] * dgrrdn);
    double cphi = (std::cos(ang[2] * dgrrdn) - ca * cb) / (sa * sb); if (cphi > 1.0) cphi = 1.0;
    const double sphi = std::sqrt(1.0 - cphi * cphi);
    box.origin = {0, 0, 0};
    box.vectors = {{{cell[0] * sb * sphi, cell[0] * sb * cphi, cell[0] * cb}, {0.0, cell[1] * sa, cell[1] * ca}, {0.0, 0.0, cell[2]}}};
    box.periodic = {true, true, true};
  }
  ExpandOptions eo; eo.shell = 12.0; eo.wrap_owned = false;
  const AtomSet atoms = expand_images(box, x, type, tag, eo);
  NeighborCutoffs cut;
  const FarList far = build_far_list(atoms, cut);
  ControlParams ctl = ff.file_control(); ctl.bond_cut = 5.0; ctl.hbond_cut = 7.5; ctl.thb_cut = 0.001; ctl.thb_cutsq = 0.00001;
  const BondedResult b = compute_bonded_core(ff, ctl, atoms, far);
  NonbondedOptions nopt; nopt.coulomb_constant = electron_terms::kCoulombKcal; nopt.legacy_coulomb_exponent = true;   // legacy c1c, see nonbonded.hpp
  const NonbondedResult nb = compute_nonbonded_core(ff, cut, atoms, far, q, nopt);
  const ElectronResult el = compute_electron_core(ff, cut, atoms, far);
  // expected terms from the export (STEP 0)
  std::vector<double> terms14; double estrc_ref = 0;
  { std::ifstream in(argv[4]); std::string line; int step = -1;
    while (std::getline(in, line)) { std::istringstream ls(line); std::string k; ls >> k;
      if (k == "STEP") { ls >> step; if (step > 0) break; }
      else if (step == 0 && k == "TERMS") { double v; while (ls >> v) terms14.push_back(v); }
      else if (step == 0 && k == "ESTRC") ls >> estrc_ref; } }
  RM_CHECK(terms14.size() == 14);
  if (terms14.size() != 14) return rmtest::finish("ereaxff_reference");
  struct Ref { const char* name; double got, want; };
  const Ref refs[] = {
      {"bond", b.e[EnergyTerm::Bond], terms14[0]},
      {"over+under", b.e[EnergyTerm::Over] + b.e[EnergyTerm::Under], terms14[1]},
      {"lone_pair", b.e[EnergyTerm::LonePair], terms14[2]},
      {"valence+penalty", b.e[EnergyTerm::Valence] + b.e[EnergyTerm::Penalty], terms14[4]},
      {"angle_conj", b.e[EnergyTerm::Coalition], terms14[5]},
      {"torsion", b.e[EnergyTerm::Torsion], terms14[7]},
      {"torsion_conj", b.e[EnergyTerm::Conjugation], terms14[8]},
      {"hbond", b.e[EnergyTerm::HBond], terms14[6]},
      {"vdw", nb.e[EnergyTerm::VdW], terms14[9]},
      {"coulomb", nb.e[EnergyTerm::Coulomb], terms14[10]},
      {"electron", el.e_electron, terms14[13]},
  };
  double worst = 0, tot = 0, tot_want = 0;
  for (const Ref& r : refs) {
    std::printf("%-16s %22.12f %22.12f  diff %11.3e  rel %9.2e\n", r.name, r.got, r.want, r.got - r.want, r.want != 0 ? (r.got - r.want) / r.want : 0.0);
    worst = std::fmax(worst, std::fabs(r.got - r.want)); tot += r.got; tot_want += r.want;
    if (skip.find(std::string(",") + r.name + ",") != std::string::npos) { std::printf("  (%s not checked: listed in the skip list)\n", r.name); continue; }
    RM_CHECK_MSG(std::fabs(r.got - r.want) <= tol * std::fmax(1.0, std::fabs(r.want)), r.name);
  }
  std::printf("%-16s %22.12f %22.12f  diff %11.3e (ESTRC of the export %.12f)\n", "total", tot, tot_want, tot - tot_want, estrc_ref);
  // total gradient against the STEP 0 GRAD lines
  std::vector<double> ref;
  {
    std::ifstream in(argv[4]); std::string line; bool step0 = false;
    while (std::getline(in, line)) {
      std::istringstream ls(line); std::string k; ls >> k;
      if (k == "STEP") { int s; ls >> s; step0 = (s == 0); if (!step0 && !ref.empty()) break; }
      else if (step0 && k == "GRAD") { int i; double gx, gy, gz; ls >> i >> gx >> gy >> gz; ref.insert(ref.end(), {gx, gy, gz}); }
    }
  }
  RM_CHECK(ref.size() == 3 * type.size());
  if (ref.size() == 3 * type.size()) {
    std::vector<double> gfold(3 * type.size(), 0.0);
    for (std::size_t k = 0; k < atoms.nall(); ++k) for (std::size_t c = 0; c < 3; ++c)
      gfold[3 * static_cast<std::size_t>(atoms.owner[k]) + c] += b.grad[3 * k + c] + nb.grad[3 * k + c] + el.grad[3 * k + c];
    double gmax = 0, gerr = 0; std::size_t worst_atom = 0;
    for (std::size_t i = 0; i < type.size(); ++i) for (std::size_t c = 0; c < 3; ++c) {
      const double g = gfold[3 * i + c];
      const double e = std::fabs(g - ref[3 * i + c]);
      if (e > gerr) { gerr = e; worst_atom = i; }
      gmax = std::fmax(gmax, std::fabs(ref[3 * i + c]));
    }
    std::printf("gradient: max |g_ref| = %.4f, max |dg| = %.3e kcal/mol/A (atom %zu, %s)\n", gmax, gerr, worst_atom + 1, ff.single(type[worst_atom]).name.c_str());
    RM_CHECK_MSG(gerr <= tol * std::fmax(1.0, gmax) * 1e3, "total gradient");
  }
  if (argc > 6 && std::string(argv[6]) != "-") {
    std::map<std::string, std::vector<double>> stage;
    { std::ifstream in(argv[6]); std::string line, cur; bool step0 = true;
      while (std::getline(in, line)) { std::istringstream ls(line); std::string k; ls >> k;
        if (k == "STEP") { int st; ls >> st; step0 = (st == 0); }
        else if (step0 && k == "STAGE") { ls >> cur; stage[cur].clear(); }
        else if (step0 && k == "SGRAD") { int i; double gx, gy, gz; ls >> i >> gx >> gy >> gz; stage[cur].insert(stage[cur].end(), {gx, gy, gz}); } } }
    const std::size_t n3 = 3 * type.size();
    for (const char* nm : {"electron", "hbond", "nonbon"}) if (stage[nm].size() != n3) { std::printf("stage %s missing\n", nm); RM_CHECK(false); return rmtest::finish("ereaxff_reference"); }
    auto cmp = [&](const char* what, const std::vector<double>& mine, const std::vector<double>& want) {
      double mx = 0, er = 0; for (std::size_t k = 0; k < n3; ++k) { mx = std::fmax(mx, std::fabs(want[k])); er = std::fmax(er, std::fabs(mine[k] - want[k])); }
      std::printf("stage gradient %-9s max |ref| %9.4f  max |diff| %.3e  (rel %.1e)\n", what, mx, er, er / std::fmax(mx, 1e-300));
      RM_CHECK_MSG(er <= tol * 1e3 * std::fmax(1.0, mx), what);
    };
    std::vector<double> ref_b(n3), ref_n(n3);
    for (std::size_t k = 0; k < n3; ++k) { ref_b[k] = stage["hbond"][k] - stage["electron"][k]; ref_n[k] = stage["nonbon"][k] - stage["hbond"][k]; }
    std::vector<double> my_b(n3), my_n(n3);
    for (std::size_t k = 0; k < n3; ++k) { my_b[k] = b.grad[k]; my_n[k] = nb.grad[k]; }
    cmp("electron", el.grad, stage["electron"]);
    cmp("bonded", my_b, ref_b);
    cmp("nonbonded", my_n, ref_n);
  }
  return rmtest::finish("ereaxff_reference");
}
