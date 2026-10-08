// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// NBR-1 (CPU part): box geometry, image expander, cell grid, far list vs brute force, owner-computes counting, reductions.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <tuple>

#include "reaxmetal/neighbor.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

namespace {
struct Rng {  // deterministic across standard libraries (std::uniform_real_distribution is not)
  std::mt19937_64 g;
  explicit Rng(std::uint64_t s) : g(s) {}
  double u() { return static_cast<double>(g() >> 11) * (1.0 / 9007199254740992.0); }
};

AtomSet random_system(const Box& box, std::size_t n, Rng& r, double spread = 1.0) {
  std::vector<double> x;
  std::vector<int> type(n, 0);
  std::vector<std::int64_t> tag;
  for (std::size_t i = 0; i < n; ++i) {
    Vec3 f{r.u() * spread, r.u() * spread, r.u() * spread};
    const Vec3 c = box.to_cartesian(f);
    x.insert(x.end(), c.begin(), c.end());
    tag.push_back(static_cast<std::int64_t>(i) + 1);
  }
  ExpandOptions opt;
  opt.shell = 9.0;
  return expand_images(box, x, type, tag, opt);
}

bool same_list(const FarList& a, const FarList& b) {
  return a.row_start == b.row_start && a.nbr == b.nbr && a.dist == b.dist && a.dvec == b.dvec;
}
}  // namespace

static void test_box() {
  const Box b = Box::from_lammps({1, 2, 3}, {4, 7, 9}, {0.5, -0.25, 1.0}, {true, true, true});
  RM_CHECK(std::fabs(b.volume() - 3.0 * 5.0 * 6.0) < 1e-12);
  const auto h = b.heights();
  RM_CHECK(std::fabs(h[2] - 6.0) < 1e-12);                    // c has the full z extent, a and b lie in planes of constant z
  RM_CHECK(h[0] < 3.0 && h[0] > 0.0);                         // tilt shortens the height normal to a
  const Vec3 r{2.3, 5.1, 4.4};
  const Vec3 back = b.to_cartesian(b.to_fractional(r));
  for (int c = 0; c < 3; ++c) RM_CHECK(std::fabs(back[static_cast<std::size_t>(c)] - r[static_cast<std::size_t>(c)]) < 1e-12);
  Box bad = b;
  std::swap(bad.vectors[0], bad.vectors[1]);                  // left-handed
  RM_EXPECT_THROW(bad.validate(), SystemError);
  Box nan = b;
  nan.vectors[1][1] = std::nan("");
  RM_EXPECT_THROW(nan.validate(), SystemError);
  RM_EXPECT_THROW(Box::orthogonal({0, 0, 0}, {0, 1, 1}, {true, true, true}), SystemError);
}

static void test_expander_exact_counts() {
  // cubic box L=10, shell 3 -> w = 0.3 along each periodic direction
  const Box b = Box::orthogonal({0, 0, 0}, {10, 10, 10}, {true, true, true});
  const std::vector<int> type{0};
  const std::vector<std::int64_t> tag{7};
  ExpandOptions opt;
  opt.shell = 3.0;
  auto count = [&](Vec3 f) {
    const Vec3 x = b.to_cartesian(f);
    const std::vector<double> xs{x[0], x[1], x[2]};
    return expand_images(b, xs, type, tag, opt).nghost();
  };
  RM_CHECK(count({0.5, 0.5, 0.5}) == 0);          // deep inside: nothing within 3 A of a face
  RM_CHECK(count({0.1, 0.5, 0.5}) == 1);          // image at +1 along x only
  RM_CHECK(count({0.95, 0.5, 0.5}) == 1);         // image at -1 along x only
  RM_CHECK(count({0.1, 0.1, 0.5}) == 3);          // x, y and the xy corner
  RM_CHECK(count({0.1, 0.1, 0.1}) == 7);          // 2^3 - 1
  RM_CHECK(count({0.1, 0.1, 0.1}) == 7);
  // a periodic chain with 1.3 A period and a 12 A shell: images -9..9 minus none -> 18 ghosts
  const Box chain = Box::orthogonal({0, 0, 0}, {1.3, 20, 20}, {true, false, false});
  const std::vector<double> x1{0.4, 10.0, 10.0};
  ExpandOptions o2;
  o2.shell = 12.0;
  const AtomSet a = expand_images(chain, x1, type, tag, o2);
  RM_CHECK(a.nghost() == 18);   // |s*1.3 + (0.4/1.3-ish offset)| : f=0.3077, s in [-9.23-0.31, ...] -> -9..9 without 0
  a.validate(chain);
}

static void test_expander_completeness_and_wrap() {
  Rng r(12345);
  const Box boxes[] = {
      Box::from_lammps({0, 0, 0}, {7.134, 7.0, 7.1}, {1.2, 0.8, 1.0}, {true, true, true}),
      Box::from_lammps({-2, 1, 0.5}, {4, 8, 6}, {-2.5, 1.5, 3.0}, {true, true, false}),
      Box::from_lammps({0, 0, 0}, {3.1, 9, 12}, {0, 0, 0}, {true, false, true}),
  };
  for (const Box& b : boxes) {
    b.validate();
    const std::size_t n = 12;
    std::vector<double> x;
    std::vector<int> type(n, 1);
    std::vector<std::int64_t> tag;
    for (std::size_t i = 0; i < n; ++i) {
      Vec3 f{r.u() * 3.0 - 1.0, r.u() * 3.0 - 1.0, r.u()};  // deliberately outside [0,1): must be wrapped
      for (std::size_t d = 0; d < 3; ++d)
        if (!b.periodic[d]) f[d] = r.u();                   // a non-periodic direction cannot be wrapped
      const Vec3 c = b.to_cartesian(f);
      x.insert(x.end(), c.begin(), c.end());
      tag.push_back(static_cast<std::int64_t>(100 + i));
    }
    ExpandOptions opt;
    opt.shell = 6.5;
    const AtomSet a = expand_images(b, x, type, tag, opt);
    a.validate(b);
    for (std::size_t i = 0; i < n; ++i)                        // owned atoms were wrapped into the primary cell
      for (std::size_t d = 0; d < 3; ++d) {
        const double f = b.to_fractional(a.position(i))[d];
        if (b.periodic[d]) RM_CHECK(f >= -1e-12 && f < 1.0 + 1e-12);
      }
    // completeness: every image (j,s) within `shell` of any owned atom is present; no duplicates
    std::set<std::tuple<int, int, int, int>> have;
    for (std::size_t g = a.nlocal; g < a.nall(); ++g) {
      const bool fresh = have.insert({a.owner[g], a.shift[g][0], a.shift[g][1], a.shift[g][2]}).second;
      RM_CHECK(fresh);
    }
    std::size_t missing = 0;
    for (int s0 = -4; s0 <= 4; ++s0)
      for (int s1 = -4; s1 <= 4; ++s1)
        for (int s2 = -4; s2 <= 4; ++s2) {
          if ((s0 && !b.periodic[0]) || (s1 && !b.periodic[1]) || (s2 && !b.periodic[2])) continue;
          if (!s0 && !s1 && !s2) continue;
          const Vec3 sh = b.lattice_shift({s0, s1, s2});
          for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < n; ++i) {
              const Vec3 xi = a.position(i), xj = a.position(j);
              const double d = std::sqrt(std::pow(xj[0] + sh[0] - xi[0], 2) + std::pow(xj[1] + sh[1] - xi[1], 2) + std::pow(xj[2] + sh[2] - xi[2], 2));
              if (d <= opt.shell && !have.count({static_cast<int>(j), s0, s1, s2})) ++missing;
            }
        }
    RM_CHECK_MSG(missing == 0, "images within the shell missing from the ghost set");
  }
  // errors
  const Box b = Box::orthogonal({0, 0, 0}, {5, 5, 5}, {true, true, true});
  const std::vector<double> x{1, 1, 1};
  const std::vector<int> t{0};
  const std::vector<std::int64_t> g{1};
  ExpandOptions tiny;
  tiny.max_atoms = 5;
  tiny.shell = 20.0;
  RM_EXPECT_THROW(expand_images(b, x, t, g, tiny), SystemError);
  const std::vector<double> xnan{std::nan(""), 1, 1};
  RM_EXPECT_THROW(expand_images(b, xnan, t, g), SystemError);
  RM_EXPECT_THROW(expand_images(b, std::vector<double>{1, 1}, t, g), SystemError);
  AtomSet bad = expand_images(b, x, t, g);
  bad.x[3 * bad.nlocal] += 1e-3;  // corrupt a ghost
  RM_EXPECT_THROW(bad.validate(b), SystemError);
}

static void test_far_list_vs_bruteforce() {
  Rng r(2024);
  const Box boxes[] = {
      Box::orthogonal({0, 0, 0}, {8, 8, 8}, {true, true, true}),
      Box::from_lammps({0, 0, 0}, {7.134, 7.0, 7.1}, {1.2, 0.8, 1.0}, {true, true, true}),
      Box::orthogonal({0, 0, 0}, {25, 25, 25}, {false, false, false}),
      Box::orthogonal({0, 0, 0}, {5.5, 30, 30}, {true, false, false}),
  };
  const NeighborCutoffs cuts[] = {{10.0, 5.0, 7.5}, {6.0, 3.5, 4.0}, {4.0, 4.0, 4.0}};
  for (const Box& b : boxes)
    for (const auto& c : cuts) {
      AtomSet a = random_system(b, 30, r, 0.8);
      a.validate(b);
      NeighborCutoffs cc = c;
      ExpandOptions opt;
      opt.shell = cc.required_shell();
      const std::size_t n = a.nlocal;
      std::vector<double> xo(a.x.begin(), a.x.begin() + static_cast<std::ptrdiff_t>(3 * n));
      std::vector<int> to(a.type.begin(), a.type.begin() + static_cast<std::ptrdiff_t>(n));
      std::vector<std::int64_t> go(a.tag.begin(), a.tag.begin() + static_cast<std::ptrdiff_t>(n));
      const AtomSet e = expand_images(b, xo, to, go, opt);
      const FarList fast = build_far_list(e, cc), slow = build_far_list_bruteforce(e, cc);
      RM_CHECK_MSG(same_list(fast, slow), "cell-grid list differs from brute force");
      RM_CHECK(fast.rows() == e.nall());
      for (std::size_t i = 0; i < fast.rows(); ++i)
        for (std::size_t p = fast.row_start[i]; p < fast.row_start[i + 1]; ++p) {
          RM_CHECK(static_cast<std::size_t>(fast.nbr[p]) > i);
          RM_CHECK(fast.dist[p] <= cc.row_cut(i, e.nlocal));
          if (p > fast.row_start[i]) RM_CHECK(fast.nbr[p] > fast.nbr[p - 1]);
        }
    }
  // owned rows use nonb_cut, ghost rows bond_cut (ENGINE_SPEC Q-08): two atoms 7 A apart, one of them a ghost-only row
  {
    AtomSet a;
    a.nlocal = 1;
    a.x = {0, 0, 0, 7, 0, 0, 14, 0, 0};
    a.type = {0, 0, 0};
    a.tag = {1, 2, 3};
    a.owner = {0, 0, 0};
    a.shift = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
    const FarList f = build_far_list(a, {10.0, 5.0, 7.5});
    RM_CHECK(f.row_start == (std::vector<std::size_t>{0, 1, 1, 1}));   // owned row sees the ghost at 7 A (<= 10); ghost rows only 5 A
    RM_CHECK(f.nbr.size() == 1 && f.nbr[0] == 1);
  }
  // the cutoff test is inclusive (d_sqr <= cutoff_sqr, pair_reaxff.cpp:652), also for pairs exactly on the cutoff
  {
    AtomSet a;
    a.nlocal = 2;
    a.x = {0, 0, 0, 3, 4, 0};
    a.type = {0, 0};
    a.tag = {1, 2};
    a.owner = {0, 1};
    a.shift = {{0, 0, 0}, {0, 0, 0}};
    RM_CHECK(build_far_list(a, {5.0, 3.0, 3.0}).entries() == 1);
    RM_CHECK(build_far_list(a, {4.999999999, 3.0, 3.0}).entries() == 0);
    RM_CHECK(build_far_list_bruteforce(a, {5.0, 3.0, 3.0}).entries() == 1);
  }
  RM_EXPECT_THROW(build_far_list(AtomSet{}, {0.0, 5.0, 7.5}), SystemError);
  RM_CHECK(build_far_list(AtomSet{}, {10.0, 5.0, 7.5}).rows() == 0);
}

static void test_pair_counting_rules() {
  // single atom, cubic periodic box L=3, nonb 5: lattice vectors with 0 < |n| L <= 5 are |n|^2 in {1,2} -> 6 + 12 = 18;
  // each physical self pair appears as +n and -n and is counted once by the coordinate rule -> 9
  const Box b = Box::orthogonal({0, 0, 0}, {3, 3, 3}, {true, true, true});
  const std::vector<double> x{0.3, 0.9, 1.7};
  const std::vector<int> t{0};
  const std::vector<std::int64_t> g{1};
  ExpandOptions opt;
  opt.shell = 10.0;
  const AtomSet a = expand_images(b, x, t, g, opt);
  const NeighborCutoffs cut{5.0, 3.5, 4.0};
  const PairCounts c = count_nonbonded_pairs(a, build_far_list(a, cut), cut);
  RM_CHECK(c.oo == 0 && c.og == 0 && c.self == 9);
  // 1.30 A period chain, nonb 10: 7 images on each side -> 7 self pairs (the M1 reference tally of cho_chain_1atom_period1.30 is vdw.self = 7)
  const Box chain = Box::orthogonal({0, 0, 0}, {1.3, 20, 20}, {true, false, false});
  const std::vector<double> x1{0.2, 10, 10};
  const AtomSet a1 = expand_images(chain, x1, t, g, opt);
  const PairCounts c1 = count_nonbonded_pairs(a1, build_far_list(a1, {10.0, 5.0, 7.5}), {10.0, 5.0, 7.5});
  RM_CHECK(c1.oo == 0 && c1.og == 0 && c1.self == 7);
  // two atoms in a 4 A cell: every physical pair once, whichever of the two orientations the lists contain
  const Box two = Box::orthogonal({0, 0, 0}, {4, 4, 4}, {true, true, true});
  const std::vector<double> xs{0.5, 0.5, 0.5, 2.0, 1.0, 3.5};
  const std::vector<int> ts{0, 0};
  const std::vector<std::int64_t> gs{1, 2};
  const AtomSet a2 = expand_images(two, xs, ts, gs, opt);
  const NeighborCutoffs c2{6.0, 3.0, 3.0};
  const PairCounts k = count_nonbonded_pairs(a2, build_far_list(a2, c2), c2);
  // independent count: lattice vectors n, pair (1,2+n) for tag order 1<2 counted from atom 1 only, owned-owned (n=0) once,
  // self images of each atom counted half -> compare with explicit enumeration
  std::size_t oo = 0, og = 0, self = 0;
  for (int s0 = -4; s0 <= 4; ++s0)
    for (int s1 = -4; s1 <= 4; ++s1)
      for (int s2 = -4; s2 <= 4; ++s2) {
        const Vec3 sh = two.lattice_shift({s0, s1, s2});
        // atom 2 + shift seen from atom 1
        const double d12 = std::sqrt(std::pow(2.0 + sh[0] - 0.5, 2) + std::pow(1.0 + sh[1] - 0.5, 2) + std::pow(3.5 + sh[2] - 0.5, 2));
        if (d12 <= 6.0) (s0 == 0 && s1 == 0 && s2 == 0) ? ++oo : ++og;
        if ((s0 || s1 || s2) && std::sqrt(std::pow(sh[0], 2) + std::pow(sh[1], 2) + std::pow(sh[2], 2)) <= 6.0) self += 2;  // two atoms, each +-n
      }
  self /= 2;   // +n and -n are one physical pair
  RM_CHECK_MSG(k.oo == oo && k.og == og && k.self == self, "pair class counts differ from the explicit enumeration");
  // classification unit cases (reaxff_nonbonded.cpp:104-117)
  const double up[3] = {0, 0, 2}, down[3] = {0, 0, -2}, tie_y[3] = {0, 3, 0}, tie_x[3] = {2, 0, 0}, tie_mx[3] = {-2, 0, 0}, tiny_z[3] = {5, 0, 5e-5};
  RM_CHECK(classify_nonbonded_entry(0, 5, 3, 4, 4, up) == PairClass::SelfImage);
  RM_CHECK(classify_nonbonded_entry(0, 5, 3, 4, 4, down) == PairClass::None);
  RM_CHECK(classify_nonbonded_entry(0, 5, 3, 4, 4, tie_y) == PairClass::SelfImage);
  RM_CHECK(classify_nonbonded_entry(0, 5, 3, 4, 4, tie_x) == PairClass::SelfImage);
  RM_CHECK(classify_nonbonded_entry(0, 5, 3, 4, 4, tie_mx) == PairClass::None);
  RM_CHECK(classify_nonbonded_entry(0, 5, 3, 4, 4, tiny_z) == PairClass::SelfImage);   // |dz| < SMALL falls through to x
  RM_CHECK(classify_nonbonded_entry(0, 5, 3, 1, 2, down) == PairClass::OwnedGhost);
  RM_CHECK(classify_nonbonded_entry(0, 5, 3, 2, 1, up) == PairClass::None);
  RM_CHECK(classify_nonbonded_entry(0, 2, 3, 9, 1, up) == PairClass::OwnedOwned);
}

static void test_cell_grid() {
  Rng r(7);
  std::vector<double> x;
  for (int i = 0; i < 3 * 500; ++i) x.push_back(r.u() * 40.0 - 20.0);
  const CellGrid g = build_cell_grid(x, 6.0);
  RM_CHECK(g.cell_size[0] >= 6.0 && g.cell_size[1] >= 6.0 && g.cell_size[2] >= 6.0);
  RM_CHECK(g.cell_start.back() == 500 && g.cell_items.size() == 500);
  for (std::uint32_t c = 0; c < g.total(); ++c)
    for (std::uint32_t p = g.cell_start[c] + 1; p < g.cell_start[c + 1]; ++p) RM_CHECK(g.cell_items[p - 1] < g.cell_items[p]);
  const CellGrid coarse = build_cell_grid(x, 1.0, 100);          // forced to coarsen
  RM_CHECK(coarse.total() <= 100 && coarse.cell_size[0] >= 1.0);
  RM_EXPECT_THROW(build_cell_grid(x, 0.0), SystemError);
  const CellGrid empty = build_cell_grid(std::vector<double>{}, 3.0);
  RM_CHECK(empty.total() == 1 && empty.cell_items.empty());
  const CellGrid single = build_cell_grid(std::vector<double>{1, 2, 3}, 3.0);
  RM_CHECK(single.total() == 1 && single.cell_items.size() == 1);
}

static void test_device_input() {
  const Box b = Box::orthogonal({-5, -5, -5}, {5, 5, 5}, {true, true, true});
  const std::vector<double> x{-4.9, 0, 0, 4.0, 4.0, 4.0};
  const std::vector<int> t{0, 0};
  const std::vector<std::int64_t> g{1, 2};
  const NeighborCutoffs cut{6.0, 3.0, 3.0};
  ExpandOptions opt;
  opt.shell = 6.0;
  const AtomSet a = expand_images(b, x, t, g, opt);
  const DeviceListInput in = make_device_list_input(a, b, cut, 1e-3);
  RM_CHECK(in.nall == a.nall() && in.nlocal == 2);
  RM_CHECK(in.x.size() == 3 * a.nall());
  RM_CHECK(std::fabs(static_cast<double>(in.x[0]) - 0.1) < 1e-6);              // relative to the box lower corner
  RM_CHECK(in.list_margin >= 1e-3);
  RM_CHECK(std::fabs(static_cast<double>(in.rc2_owned) - 36.012001) < 1e-3);   // (6 + margin)^2
  RM_CHECK(in.grid.cell_size[0] >= 6.0 + 2.0 * in.list_margin - 1e-12);
  RM_EXPECT_THROW(make_device_list_input(a, b, cut, -1.0), SystemError);
  // huge coordinates raise the margin above the request (float resolution)
  AtomSet far = a;
  far.x[0] = 1.0e6;
  RM_CHECK(make_device_list_input(far, b, cut, 1e-3).list_margin > 0.5);
}

static void test_reductions() {
  // values chosen so that the summation order matters in float: 1e8 + 1 - 1e8
  const std::vector<float> v{1.0e8f, 1.0f, -1.0e8f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float seq = 0.0f;
  for (float e : v) seq += e;
  RM_CHECK(fixed_order_sum_f32(v, 8) == seq);               // one chunk == plain sequential order
  RM_CHECK(fixed_order_sum_f32(v, 3) == fixed_order_sum_f32(v, 3));
  const auto p = fixed_order_partials_f32(v, 3);
  RM_CHECK(p.size() == 3);
  RM_CHECK(p[0] == 0.0f);                                   // (1e8 + 1) - 1e8 = 0 in float
  RM_CHECK(p[1] == 3.0f && p[2] == 2.0f);
  RM_CHECK(fixed_order_sum_f32(v, 3) == 5.0f);
  RM_CHECK(fixed_order_sum_f32(std::vector<float>{}, 4) == 0.0f);
  RM_EXPECT_THROW(fixed_order_partials_f32(v, 0), SystemError);
  // bitwise reproducible for any input
  Rng r(99);
  std::vector<float> w(10007);
  for (auto& e : w) e = static_cast<float>(r.u() - 0.5) * 1000.0f;
  RM_CHECK(fixed_order_sum_f32(w, 256) == fixed_order_sum_f32(w, 256));
}

int main() {
  test_box();
  test_expander_exact_counts();
  test_expander_completeness_and_wrap();
  test_far_list_vs_bruteforce();
  test_pair_counting_rules();
  test_cell_grid();
  test_device_input();
  test_reductions();
  if (rmtest::failures() != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", rmtest::failures());
    return 1;
  }
  std::puts("neighbor: all checks passed");
  return 0;
}
