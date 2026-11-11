// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// eReaxFF modified bonded terms in the CPU-64 engine (plan tasks 3.3 / 3.4): electron sharing (corrected valencies A, B), lone pair with
// stlpc, over/under-coordination with the pxel2*e2*S factor. Central finite differences of the energies against the analytic gradient,
// on small clusters and in a periodic cell, with knock-out variants that isolate each term. SYNTHETIC force field (invented numbers).
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <string>

#include "reaxmetal/bonded.hpp"
#include "reaxmetal/forcefield.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

namespace {
std::string num(double v) { std::ostringstream o; o.precision(17); o << v; return o.str(); }

struct Knobs {
  double de = 1.0;        // scales the bond dissociation energies (0 removes the bond energy)
  double ovun1 = 0.45;    // p_ovun1 (bond column 8): 0 removes sum_ovun1 and with it the over-coordination energy
  double valp1 = 38.0;    // C under-coordination strength; < 0 switches the under-coordination energy off
  double plp2 = 10.0;     // C lone-pair strength; 0 removes the lone-pair energy
  double pxel2 = 0.9;     // atom line 2 col 5 of C
  double pxel1 = 0.15;    // bond line 2 col 8
  double vincr = 3.0;     // sharing width of C
  double gp5 = 0.0;       // C2 correction strength (0 = inactive)
};

std::string make_ff(const Knobs& k) {
  std::ostringstream o;
  o << "synthetic eReaxFF\n39 ! n\n";
  for (int i = 0; i < 39; ++i) {
    double v = 1.0;
    switch (i) {
      case 0: v = 50.0; break; case 1: v = 9.5; break; case 2: v = 1.67; break; case 3: v = 1.72; break; case 4: v = 6.87; break;
      case 5: v = k.gp5; break; case 6: v = 1.06; break; case 7: v = 4.6; break; case 8: v = 12.1; break; case 9: v = 13.3; break;
      case 10: v = -34.5; break; case 11: v = 0.0; break; case 12: v = 10.0; break; case 13: v = 2.9; break; case 14: v = 33.9; break;
      case 15: v = 6.09; break; case 16: v = 1.06; break; case 17: v = 2.04; break; case 29: v = 0.1; break; case 31: v = 0.7; break;
      case 32: v = 50.0; break; case 35: v = 1.0; break; case 37: v = 5.0; break;
      default: break;
    }
    o << " " << num(v) << " !p" << i << "\n";
  }
  o << "2 ! atoms\n c\n c\n c\n";
  o << " C 1.35 4.0 12.011 2.05 0.11 0.35 1.17 4.0\n  9.0 1.5 4.0 " << num(k.valp1) << " " << num(k.pxel2) << " 5.3 4.5 3.0\n  1.15 " << num(k.plp2)
    << " " << num(k.vincr) << " 5.4 19.3 13.5 3.2 22.0\n  -4.1 4.9 0.0027 4.0 3.0 0.0 0.0 0.0\n";
  o << " El -0.1 0.0 1.0 1.3 0.005 0.25 -0.1 0.0\n  2.0 4.0 4.0 0.0 0.0 1.0 1.0 0.0\n  -0.1 0.0 -2.37 6.5 8.6 0.24 2.0 3.5\n  -5.0 3.19 0.16 6.3 2.58 0.0 0.0 0.0\n";
  o << "1 ! bonds\n c\n  1 1 " << num(156.0 * k.de) << " " << num(100.0 * k.de) << " " << num(80.0 * k.de) << " 0.55 -0.30 1.0 25.0 " << num(k.ovun1)
    << "\n  3.9 -0.16 8.2 1.0 -0.058 6.8 1.0 " << num(k.pxel1) << "\n";
  o << "0\n0\n0\n0\n";
  return o.str();
}

struct Sys { Box box; std::vector<double> x; std::vector<int> type; };   // type 0 = C, 1 = El

AtomSet atoms_of(const Sys& s) {
  std::vector<std::int64_t> tag(s.type.size());
  for (std::size_t i = 0; i < tag.size(); ++i) tag[i] = static_cast<std::int64_t>(i) + 1;
  ExpandOptions o; o.shell = 12.0; o.wrap_owned = false;
  return expand_images(s.box, s.x, s.type, tag, o);
}

BondedResult run(const ForceField& ff, const Sys& s, AtomSet* out = nullptr) {
  NeighborCutoffs cut;
  AtomSet a = atoms_of(s);
  const FarList far = build_far_list(a, cut);
  ControlParams ctl = ff.file_control(); ctl.bond_cut = 5.0; ctl.hbond_cut = 7.5; ctl.thb_cut = 0.001;
  BondedResult r = compute_bonded_core(ff, ctl, a, far);
  if (out) *out = a;
  return r;
}
double energy(const BondedResult& r) {
  return r.e[EnergyTerm::Bond] + r.e[EnergyTerm::LonePair] + r.e[EnergyTerm::Over] + r.e[EnergyTerm::Under];
}
std::vector<double> folded(const BondedResult& r, const AtomSet& a) {
  std::vector<double> g(3 * a.nlocal, 0.0);
  for (std::size_t i = 0; i < a.nall(); ++i) for (std::size_t c = 0; c < 3; ++c) g[3 * static_cast<std::size_t>(a.owner[i]) + c] += r.grad[3 * i + c];
  return g;
}
double max_fd_error(const ForceField& ff, Sys s, double h, double* scale = nullptr) {
  AtomSet a;
  const std::vector<double> g = folded(run(ff, s, &a), a);
  double worst = 0, mx = 0;
  for (std::size_t k = 0; k < s.x.size(); ++k) {
    const double x0 = s.x[k];
    s.x[k] = x0 + h; const double ep = energy(run(ff, s));
    s.x[k] = x0 - h; const double em = energy(run(ff, s));
    s.x[k] = x0;
    worst = std::max(worst, std::fabs((ep - em) / (2 * h) - g[k]));
    mx = std::max(mx, std::fabs(g[k]));
  }
  if (scale) *scale = mx;
  return worst;
}

// carbon ring (+ a few carbons) with electrons close enough that exp(-vincr r^2) is O(0.1)
Sys make_sys(std::mt19937_64& rng, bool periodic, int nel) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Sys s;
  const double L = 7.4;
  s.box = periodic ? Box::orthogonal({0, 0, 0}, {L, L, L}, {true, true, true}) : Box::orthogonal({-50, -50, -50}, {50, 50, 50}, {false, false, false});
  const double cx = periodic ? 0.3 : 0.0, cy = periodic ? 0.2 : 0.0, cz = periodic ? 0.5 * L : 0.0;
  for (int i = 0; i < 6; ++i) {   // ring of 6 C, plus 2 attached C
    const double ang = 2 * 3.14159265358979 * i / 6.0;
    s.x.push_back(cx + 1.42 * std::cos(ang) + 0.08 * u(rng)); s.x.push_back(cy + 1.42 * std::sin(ang) + 0.08 * u(rng)); s.x.push_back(cz + 0.08 * u(rng));
    s.type.push_back(0);
  }
  s.x.push_back(cx + 2.8 + 0.1 * u(rng)); s.x.push_back(cy + 0.1 * u(rng)); s.x.push_back(cz + 0.1 * u(rng)); s.type.push_back(0);
  s.x.push_back(cx - 2.7 + 0.1 * u(rng)); s.x.push_back(cy + 0.9 + 0.1 * u(rng)); s.x.push_back(cz + 0.1 * u(rng)); s.type.push_back(0);
  for (int e = 0; e < nel; ++e) {  // electrons 0.35-0.9 A from a carbon
    const std::size_t host = static_cast<std::size_t>(e) % 8;
    double d[3] = {u(rng), u(rng), u(rng)};
    const double nrm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), rad = 0.35 + 0.55 * std::fabs(u(rng));
    for (std::size_t c = 0; c < 3; ++c) s.x.push_back(s.x[3 * host + c] + rad * d[c] / nrm);
    s.type.push_back(1);
  }
  return s;
}
}  // namespace

int main() {
  FfieldOptions fo; fo.ereaxff = true; fo.electron_taper = 5.0;
  struct Case { const char* name; Knobs k; };
  Knobs full;
  Knobs lp_only; lp_only.de = 0; lp_only.valp1 = -1; lp_only.ovun1 = 0.0;                         // E = lone pair only
  Knobs lp_nopx; lp_nopx = lp_only; lp_nopx.pxel2 = 0.0;
  Knobs under_only; under_only.de = 0; under_only.plp2 = 0.0; under_only.ovun1 = 0.0;           // E = under-coordination only
  Knobs over_only; over_only.plp2 = 0.0; over_only.valp1 = -1;                                    // E = bond + over-coordination
  Knobs over_nopx = over_only; over_nopx.pxel2 = 0.0;
  Knobs no_share = full; no_share.vincr = 0.0; no_share.pxel2 = 0.0;                             // no electron influence at all
  Knobs c2 = full; c2.gp5 = 54.7;
  const Case cases[] = {{"lone pair only (A,B channels)", lp_only}, {"lone pair only, pxel2=0", lp_nopx}, {"under only", under_only},
                        {"over only (+bond)", over_only}, {"over only, pxel2=0", over_nopx}, {"all terms", full}, {"all terms + C2 correction", c2}};
  std::mt19937_64 rng(2024);
  for (const Case& cs : cases) {
    const ForceField ff = parse_force_field(make_ff(cs.k), "syn", fo);
    for (int periodic = 0; periodic < 2; ++periodic) {
      double worst = 0, scale = 0;
      for (int trial = 0; trial < 3; ++trial) {
        const Sys s = make_sys(rng, periodic == 1, 4);
        double sc = 0;
        worst = std::max(worst, max_fd_error(ff, s, 1e-6, &sc));
        scale = std::max(scale, sc);
      }
      std::printf("%-32s %-8s max |FD - analytic| = %.3e kcal/mol/A (max |g| = %.2f)\n", cs.name, periodic ? "periodic" : "cluster", worst, scale);
      RM_CHECK_MSG(worst < 1e-5, std::string(cs.name) + (periodic ? " periodic" : " cluster"));
    }
  }
  // sanity: electrons really change the energy (otherwise the test would be vacuous)
  {
    const ForceField ff = parse_force_field(make_ff(full), "syn", fo);
    std::mt19937_64 r2(5);
    Sys s = make_sys(r2, false, 4);
    Sys s0 = s; s0.x.resize(3 * 8); s0.type.resize(8);
    const double with = energy(run(ff, s)), without = energy(run(ff, s0));
    std::printf("electrons change the bonded energy by %.4f kcal/mol\n", with - without);
    RM_CHECK(std::fabs(with - without) > 1e-3);
  }
  return rmtest::finish("ereaxff_bonded");
}
