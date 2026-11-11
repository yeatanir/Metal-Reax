// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// eReaxFF electron pair term (plan task 3.2): closed-form single pair, taper behaviour, finite-difference forces in a free cluster and in a small
// periodic cell (ghost gradients folded onto owners), translation invariance and zero net force. SYNTHETIC force field (invented numbers).
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <string>

#include "reaxmetal/electron.hpp"
#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/terms_host.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

namespace {
std::string num(double v) { std::ostringstream o; o.precision(17); o << v; return o.str(); }

std::string make_ff() {  // C and El; electron columns chosen to give O(1) kcal/mol energies at ~1-3 A
  std::ostringstream o;
  o << "synthetic\n39 ! n\n";
  for (int i = 0; i < 39; ++i) { double v = 1.0 + 0.5 * i; if (i == 5) v = 0; if (i == 11) v = 0; if (i == 12) v = 10; if (i == 29) v = 0.5; if (i == 35) v = 1; if (i == 37) v = 5.0; o << " " << num(v) << " !p\n"; }
  o << "2 ! atoms\n c\n c\n c\n";
  o << " C 1.5 4.0 12.0 1.75 0.25 0.9 1.25 4.0\n  9.0 2.0 4.0 20.0 0.8 5.5 4.5 0.0\n  1.25 0.5 181.0 0.1 0.2 0.3 0.4 0.9\n  0.7 2.5 0.35 4.0 3.0 0.0 0.0 0.0\n";
  o << " El -0.1 0.0 1.0 1.3 0.005 0.25 -0.1 0.0\n  2.0 2.0 4.0 0.0 0.11 1.0002 1.0009 10.0\n  -0.1 0.0 -2.37 6.4918 8.5961 0.2368 2.0 0.7\n  -5.0 3.1873 0.28 6.2998 2.5791 0.0 0.0 0.0\n";
  o << "1 ! bonds\n c\n  1 1 78.4 115.4 68.2 0.58 -0.29 1.0 35.0 0.51\n  3.856 -0.164 8.2326 1.0 -0.0585 6.7997 1.0 0.1481\n";
  o << "0\n0\n0\n0\n";
  return o.str();
}

struct Sys {
  Box box;
  std::vector<double> x;   // owned positions, 3n
  std::vector<int> type;   // 0 = C, 1 = El
};

AtomSet atoms_of(const Sys& s, double shell) {
  std::vector<std::int64_t> tag(s.type.size());
  for (std::size_t i = 0; i < tag.size(); ++i) tag[i] = static_cast<std::int64_t>(i) + 1;
  ExpandOptions o;
  o.shell = shell;
  o.wrap_owned = false;
  return expand_images(s.box, s.x, s.type, tag, o);
}

ElectronResult run(const ForceField& ff, const Sys& s, AtomSet* out_atoms = nullptr) {
  NeighborCutoffs cut;  // nonb 10
  AtomSet a = atoms_of(s, 12.0);
  const FarList far = build_far_list(a, cut);
  ElectronResult r = compute_electron_core(ff, cut, a, far);
  if (out_atoms) *out_atoms = a;
  return r;
}

// owner-folded gradient (ghost contributions added to their owner)
std::vector<double> folded(const ElectronResult& r, const AtomSet& a) {
  std::vector<double> g(3 * a.nlocal, 0.0);
  for (std::size_t i = 0; i < a.nall(); ++i) {
    const std::size_t o = static_cast<std::size_t>(a.owner[i]);
    for (std::size_t c = 0; c < 3; ++c) g[3 * o + c] += r.grad[3 * i + c];
  }
  return g;
}

double max_fd_error(const ForceField& ff, Sys s, double h) {
  AtomSet a;
  const std::vector<double> g = folded(run(ff, s, &a), a);
  double worst = 0;
  for (std::size_t k = 0; k < s.x.size(); ++k) {
    const double x0 = s.x[k];
    s.x[k] = x0 + h; const double ep = run(ff, s).e_electron;
    s.x[k] = x0 - h; const double em = run(ff, s).e_electron;
    s.x[k] = x0;
    worst = std::max(worst, std::fabs((ep - em) / (2 * h) - g[k]));
  }
  return worst;
}
}  // namespace

int main() {
  FfieldOptions fo; fo.ereaxff = true; fo.electron_taper = 5.0;
  const ForceField ff = parse_force_field(make_ff(), "syn", fo);
  const SingleBody& C = ff.single(0); const SingleBody& E = ff.single(1);

  // (1) one core-El pair and one El-El pair against the closed forms (written independently here from the formulas)
  {
    const double r = 1.3, pi = 3.14159265358979323846;
    Sys s{Box::orthogonal({-50, -50, -50}, {50, 50, 50}, {false, false, false}), {0, 0, 0, r, 0, 0}, {0, 1}};
    const ElectronResult res = run(ff, s);
    const double c = electron_terms::kCoulombKcal;
    const auto tap = terms::taper_coeffs(0.0, 5.0);
    double Tap, dT; terms::taper_horner<double>(tap.c, r, Tap, dT);
    const double want = -c * C.betael * 6.0 * std::erf(std::sqrt(2 * C.alphag) * r) / r * Tap;
    RM_CHECK(std::fabs(res.e_core_el - want) < 1e-12 * std::fabs(want));
    RM_CHECK(res.pairs == 1 && res.e_el_el == 0.0);
    Sys s2{s.box, {0, 0, 0, r, 0, 0}, {1, 1}};
    const ElectronResult r2 = run(ff, s2);
    const double want2 = c * E.betael * std::erf(std::sqrt(E.alphag) * r) / r * Tap;
    RM_CHECK(std::fabs(r2.e_el_el - want2) < 1e-12 * std::fabs(want2));
    RM_CHECK(r2.e_core_el == 0.0);
    (void)pi;
    // C-C: no electron term at all
    Sys s3{s.box, {0, 0, 0, r, 0, 0}, {0, 0}};
    RM_CHECK(run(ff, s3).pairs == 0 && run(ff, s3).e_electron == 0.0);
  }
  // (2) beyond the electron taper the term is exactly zero, at the taper it vanishes continuously
  {
    Sys far_s{Box::orthogonal({-50, -50, -50}, {50, 50, 50}, {false, false, false}), {0, 0, 0, 5.0000001, 0, 0}, {0, 1}};
    RM_CHECK(run(ff, far_s).e_electron == 0.0);
    far_s.x[3] = 4.9999;
    RM_CHECK(std::fabs(run(ff, far_s).e_electron) < 1e-9);
  }
  // (3) finite-difference forces: free cluster and small periodic cell, random configurations
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int periodic = 0; periodic < 2; ++periodic) {
    double worst = 0;
    for (int trial = 0; trial < 4; ++trial) {
      Sys s;
      const double L = periodic ? 7.5 : 12.0;
      s.box = Box::orthogonal({0, 0, 0}, {L, L, L}, {periodic == 1, periodic == 1, periodic == 1});
      const int n = 9;
      for (int i = 0; i < n; ++i) { s.type.push_back(i % 3 == 0 ? 1 : 0); for (int c = 0; c < 3; ++c) s.x.push_back(0.2 * L + 0.6 * L * u(rng)); }
      if (periodic) for (auto& v : s.x) v = L * u(rng);
      worst = std::max(worst, max_fd_error(ff, s, 1e-5));
      // zero net force and translation invariance
      AtomSet a; const auto g = folded(run(ff, s, &a), a);
      double fx[3] = {0, 0, 0};
      for (std::size_t i = 0; i < s.type.size(); ++i) for (std::size_t c = 0; c < 3; ++c) fx[c] += g[3 * i + c];
      RM_CHECK(std::fabs(fx[0]) + std::fabs(fx[1]) + std::fabs(fx[2]) < 1e-9);
      Sys t = s; for (std::size_t k = 0; k < t.x.size(); k += 3) t.x[k] += 0.37;
      RM_CHECK(std::fabs(run(ff, t).e_electron - run(ff, s).e_electron) < 1e-9);
    }
    std::printf("%s: max |FD - analytic gradient| = %.3e kcal/mol/A\n", periodic ? "periodic" : "cluster ", worst);
    RM_CHECK_MSG(worst < 1e-6, "finite-difference gradient");
  }
  // (4) refusals
  {
    const ForceField stock = parse_force_field(make_ff(), "syn");
    Sys s{Box::orthogonal({-50, -50, -50}, {50, 50, 50}, {false, false, false}), {0, 0, 0, 1, 0, 0}, {0, 1}};
    NeighborCutoffs cut; AtomSet a = atoms_of(s, 12.0); const FarList far = build_far_list(a, cut);
    RM_EXPECT_THROW(compute_electron_core(stock, cut, a, far), SystemError);  // not parsed in eReaxFF mode
    NeighborCutoffs shortc; shortc.nonb = 4.0;
    RM_EXPECT_THROW(compute_electron_core(ff, shortc, a, far), SystemError);  // far list shorter than the taper
  }
  return rmtest::finish("electron");
}
