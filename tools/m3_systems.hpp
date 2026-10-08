// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Deterministic geometries shared by the Metal check tool (Apple machine) and the Linux emulation test, so that both exercise
// the same systems. Header-only, test support only.
#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "reaxmetal/neighbor.hpp"

namespace reaxmetal::m3 {

struct NamedSystem {
  std::string name;
  Box box;
  AtomSet atoms;
  NeighborCutoffs cut;
};

namespace detail {
struct Rng {  // deterministic across standard libraries (std::uniform_real_distribution is not)
  std::mt19937_64 g;
  explicit Rng(std::uint64_t s) : g(s) {}
  double u() { return static_cast<double>(g() >> 11) * (1.0 / 9007199254740992.0); }
};

inline NamedSystem make(std::string name, const Box& box, std::vector<double> x, const NeighborCutoffs& cut) {
  const std::size_t n = x.size() / 3;
  std::vector<int> type(n, 0);
  std::vector<std::int64_t> tag(n);
  for (std::size_t i = 0; i < n; ++i) tag[i] = static_cast<std::int64_t>(i) + 1;
  ExpandOptions opt;
  opt.shell = cut.required_shell();
  NamedSystem s{std::move(name), box, expand_images(box, x, type, tag, opt), cut};
  s.atoms.validate(box);
  return s;
}

inline std::vector<double> random_in_box(const Box& box, std::size_t n, Rng& r) {
  std::vector<double> x;
  for (std::size_t i = 0; i < n; ++i) {
    const Vec3 c = box.to_cartesian({r.u(), r.u(), r.u()});
    x.insert(x.end(), c.begin(), c.end());
  }
  return x;
}
}  // namespace detail

// `large` adds a ~20k-atom system for timing.
inline std::vector<NamedSystem> systems(bool large) {
  using namespace detail;
  std::vector<NamedSystem> out;
  const NeighborCutoffs reax{10.0, 5.0, 7.5};
  {
    Rng r(1);
    const Box b = Box::orthogonal({0, 0, 0}, {14, 14, 14}, {true, true, true});
    out.push_back(make("random_cubic_periodic_300", b, random_in_box(b, 300, r), reax));
  }
  {
    Rng r(2);
    const Box b = Box::from_lammps({0, 0, 0}, {13.0, 12.0, 11.0}, {3.0, -2.5, 1.5}, {true, true, true});
    out.push_back(make("random_triclinic_periodic_200", b, random_in_box(b, 200, r), reax));
  }
  {
    std::vector<double> x;   // diamond cubic, a = 3.567 A, 3x3x3 cells, 216 atoms
    const double a = 3.567;
    const double basis[8][3] = {{0, 0, 0}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0}, {.25, .25, .25}, {.25, .75, .75}, {.75, .25, .75}, {.75, .75, .25}};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
          for (const auto& bb : basis) {
            x.push_back((i + bb[0]) * a);
            x.push_back((j + bb[1]) * a);
            x.push_back((k + bb[2]) * a);
          }
    const Box b = Box::orthogonal({0, 0, 0}, {3 * a, 3 * a, 3 * a}, {true, true, true});
    out.push_back(make("diamond_3x3x3_periodic_216", b, std::move(x), reax));
  }
  {
    Rng r(4);
    std::vector<double> x;   // non-periodic cluster inside a sphere of radius 8
    while (x.size() < 3 * 120) {
      const double px = (r.u() - 0.5) * 16, py = (r.u() - 0.5) * 16, pz = (r.u() - 0.5) * 16;
      if (px * px + py * py + pz * pz <= 64.0) { x.push_back(px); x.push_back(py); x.push_back(pz); }
    }
    const Box b = Box::orthogonal({-20, -20, -20}, {20, 20, 20}, {false, false, false});
    out.push_back(make("cluster_nonperiodic_120", b, std::move(x), reax));
  }
  {
    const Box b = Box::orthogonal({0, 0, 0}, {1.3, 30, 30}, {true, false, false});
    out.push_back(make("chain_period_1.30_1atom", b, {0.2, 15, 15}, reax));
  }
  {
    Rng r(6);
    const Box b = Box::orthogonal({0, 0, 0}, {7.54, 7.54, 7.54}, {true, true, true});   // cell smaller than the cutoff (many image shells)
    out.push_back(make("small_cell_7.54_64", b, random_in_box(b, 64, r), reax));
  }
  if (large) {
    Rng r(7);
    const Box b = Box::orthogonal({0, 0, 0}, {40, 40, 40}, {true, true, true});
    out.push_back(make("random_cubic_periodic_6000", b, random_in_box(b, 6000, r), reax));
  }
  return out;
}

}  // namespace reaxmetal::m3
