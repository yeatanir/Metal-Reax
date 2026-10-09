// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Image expander, cell grid and CPU-64 far list. Semantics: include/reaxmetal/neighbor.hpp, ENGINE_SPEC 3 / 3.1.
#include "reaxmetal/neighbor.hpp"

#include <algorithm>
#include <thread>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <string>

namespace reaxmetal {

double NeighborCutoffs::required_shell() const noexcept { return std::max({nonb, hbond, 2.0 * bond}); }

// ---- image expander ----------------------------------------------------------------------------------------------
AtomSet expand_images(const Box& box, std::span<const double> x_owned, std::span<const int> type, std::span<const std::int64_t> tag,
                      const ExpandOptions& opt) {
  box.validate();
  const std::size_t n = type.size();
  if (x_owned.size() != 3 * n || tag.size() != n) throw SystemError("expand_images: x/type/tag sizes disagree");
  if (!(opt.shell >= 0.0) || !std::isfinite(opt.shell)) throw SystemError("expand_images: shell must be finite and >= 0");

  AtomSet a;
  a.nlocal = n;
  a.x.assign(x_owned.begin(), x_owned.end());
  a.type.assign(type.begin(), type.end());
  a.tag.assign(tag.begin(), tag.end());
  a.owner.resize(n);
  std::iota(a.owner.begin(), a.owner.end(), 0);
  a.shift.assign(n, {0, 0, 0});

  const auto h = box.heights();
  std::array<double, 3> w{0, 0, 0};
  std::array<int, 3> smax{0, 0, 0};
  for (std::size_t d = 0; d < 3; ++d)
    if (box.periodic[d]) {
      w[d] = opt.shell / h[d];
      smax[d] = static_cast<int>(std::ceil(w[d])) + 1;
    }

  // fractional coordinates of the owned atoms (after the optional wrap)
  std::vector<Vec3> frac(n);
  for (std::size_t i = 0; i < n; ++i) {
    Vec3 r = a.position(i);
    for (std::size_t c = 0; c < 3; ++c)
      if (!std::isfinite(r[c])) throw SystemError("expand_images: non-finite coordinate of atom " + std::to_string(i));
    Vec3 f = box.to_fractional(r);
    if (opt.wrap_owned) {
      std::array<double, 3> k{0, 0, 0};
      for (std::size_t d = 0; d < 3; ++d)
        if (box.periodic[d]) k[d] = std::floor(f[d]);
      for (std::size_t d = 0; d < 3; ++d)
        if (k[d] != 0.0)
          for (std::size_t c = 0; c < 3; ++c) r[c] -= k[d] * box.vectors[d][c];
      f = box.to_fractional(r);
      for (std::size_t c = 0; c < 3; ++c) a.x[3 * i + c] = r[c];
    }
    frac[i] = f;
  }

  for (int s0 = -smax[0]; s0 <= smax[0]; ++s0)
    for (int s1 = -smax[1]; s1 <= smax[1]; ++s1)
      for (int s2 = -smax[2]; s2 <= smax[2]; ++s2) {
        if (s0 == 0 && s1 == 0 && s2 == 0) continue;
        const std::array<int, 3> s{s0, s1, s2};
        const Vec3 sh = box.lattice_shift(s);
        for (std::size_t k = 0; k < n; ++k) {
          bool inside = true;
          for (std::size_t d = 0; d < 3 && inside; ++d) {
            if (!box.periodic[d]) continue;  // only the zero shift exists along a non-periodic direction
            const double g = frac[k][d] + static_cast<double>(s[d]);
            inside = (g >= -w[d]) && (g <= 1.0 + w[d]);
          }
          if (!inside) continue;
          if (a.nall() >= opt.max_atoms) throw SystemError("expand_images: more than max_atoms atoms would be created");
          a.x.push_back(a.x[3 * k] + sh[0]);
          a.x.push_back(a.x[3 * k + 1] + sh[1]);
          a.x.push_back(a.x[3 * k + 2] + sh[2]);
          a.type.push_back(a.type[k]);
          a.tag.push_back(a.tag[k]);
          a.owner.push_back(static_cast<std::int32_t>(k));
          a.shift.push_back({s0, s1, s2});
        }
      }
  return a;
}

// ---- cell grid ---------------------------------------------------------------------------------------------------
void CellGrid::coords(std::uint32_t cell, std::uint32_t c[3]) const noexcept {
  c[0] = cell % ncell[0];
  c[1] = (cell / ncell[0]) % ncell[1];
  c[2] = cell / (ncell[0] * ncell[1]);
}

CellGrid build_cell_grid(std::span<const double> x, double min_cell, std::size_t max_cells) {
  if (!(min_cell > 0.0) || !std::isfinite(min_cell)) throw SystemError("build_cell_grid: min_cell must be positive and finite");
  if (x.size() % 3 != 0) throw SystemError("build_cell_grid: x size is not a multiple of 3");
  const std::size_t n = x.size() / 3;
  CellGrid g;
  Vec3 lo{0, 0, 0}, hi{0, 0, 0};
  if (n > 0) {
    lo = hi = {x[0], x[1], x[2]};
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t c = 0; c < 3; ++c) {
        if (!std::isfinite(x[3 * i + c])) throw SystemError("build_cell_grid: non-finite coordinate");
        lo[c] = std::min(lo[c], x[3 * i + c]);
        hi[c] = std::max(hi[c], x[3 * i + c]);
      }
  }
  g.origin = lo;
  double ext[3];
  for (std::size_t c = 0; c < 3; ++c) {
    ext[c] = hi[c] - lo[c];
    const double k = std::floor(ext[c] / min_cell);
    g.ncell[c] = k < 1.0 ? 1u : (k > 4.0e9 ? 4'000'000'000u : static_cast<std::uint32_t>(k));
  }
  auto total = [&] { return static_cast<double>(g.ncell[0]) * g.ncell[1] * g.ncell[2]; };
  while (total() > static_cast<double>(max_cells)) {
    std::size_t big = 0;
    for (std::size_t c = 1; c < 3; ++c)
      if (g.ncell[c] > g.ncell[big]) big = c;
    g.ncell[big] = (g.ncell[big] + 1) / 2;
  }
  for (std::size_t c = 0; c < 3; ++c) g.cell_size[c] = std::max(min_cell, ext[c] / g.ncell[c]);

  g.atom_cell.resize(n);
  const std::uint32_t nc = g.total();
  g.cell_start.assign(static_cast<std::size_t>(nc) + 1, 0);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t c[3];
    for (std::size_t d = 0; d < 3; ++d) {
      const double t = std::floor((x[3 * i + d] - lo[d]) / g.cell_size[d]);
      c[d] = t < 0.0 ? 0u : std::min<std::uint32_t>(g.ncell[d] - 1, static_cast<std::uint32_t>(t));
    }
    g.atom_cell[i] = (c[2] * g.ncell[1] + c[1]) * g.ncell[0] + c[0];
    ++g.cell_start[g.atom_cell[i] + 1];
  }
  for (std::size_t c = 0; c < nc; ++c) g.cell_start[c + 1] += g.cell_start[c];
  g.cell_items.resize(n);
  std::vector<std::uint32_t> fill(g.cell_start.begin(), g.cell_start.end() - 1);
  for (std::size_t i = 0; i < n; ++i) g.cell_items[fill[g.atom_cell[i]]++] = static_cast<std::uint32_t>(i);
  return g;
}

// ---- CPU-64 far list ---------------------------------------------------------------------------------------------
namespace {
// the reference's arithmetic: get_distance(x[j], x[i]) -> dvec = x_j - x_i; d_sqr = dx^2 + dy^2 + dz^2 (pair_reaxff.cpp:645-660)
inline bool within(const double* xi, const double* xj, double rc2, double dv[3], double& d2) {
  dv[0] = xj[0] - xi[0];
  dv[1] = xj[1] - xi[1];
  dv[2] = xj[2] - xi[2];
  d2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
  return d2 <= rc2;
}

struct RowEntry {
  std::int32_t j;
  double d, dv[3];
};

void append_row(FarList& out, std::vector<RowEntry>& row) {
  std::sort(row.begin(), row.end(), [](const RowEntry& a, const RowEntry& b) { return a.j < b.j; });
  for (const auto& e : row) {
    out.nbr.push_back(e.j);
    out.dist.push_back(e.d);
    out.dvec.insert(out.dvec.end(), e.dv, e.dv + 3);
  }
  out.row_start.push_back(out.nbr.size());
  row.clear();
}

void check_cutoffs(const NeighborCutoffs& cut) {
  if (!(cut.nonb > 0.0) || !(cut.bond > 0.0) || !std::isfinite(cut.nonb) || !std::isfinite(cut.bond))
    throw SystemError("neighbor cutoffs must be positive and finite");
}
}  // namespace

FarList build_far_list(const AtomSet& atoms, const NeighborCutoffs& cut) {
  check_cutoffs(cut);
  const std::size_t n = atoms.nall();
  FarList out;
  out.row_start.push_back(0);
  if (n == 0) return out;
  const CellGrid g = build_cell_grid(atoms.x, std::max(cut.nonb, cut.bond));
  std::vector<RowEntry> row;
  for (std::size_t i = 0; i < n; ++i) {
    const double rc = cut.row_cut(i, atoms.nlocal);
    const double rc2 = rc * rc;
    std::uint32_t c[3];
    g.coords(g.atom_cell[i], c);
    const double* xi = &atoms.x[3 * i];
    for (std::uint32_t cz = c[2] ? c[2] - 1 : 0; cz <= std::min(c[2] + 1, g.ncell[2] - 1); ++cz)
      for (std::uint32_t cy = c[1] ? c[1] - 1 : 0; cy <= std::min(c[1] + 1, g.ncell[1] - 1); ++cy)
        for (std::uint32_t cx = c[0] ? c[0] - 1 : 0; cx <= std::min(c[0] + 1, g.ncell[0] - 1); ++cx) {
          const std::uint32_t cell = (cz * g.ncell[1] + cy) * g.ncell[0] + cx;
          for (std::uint32_t p = g.cell_start[cell]; p < g.cell_start[cell + 1]; ++p) {
            const std::uint32_t j = g.cell_items[p];
            if (j <= i) continue;
            RowEntry e;
            double d2;
            if (!within(xi, &atoms.x[3 * j], rc2, e.dv, d2)) continue;
            e.j = static_cast<std::int32_t>(j);
            e.d = std::sqrt(d2);
            row.push_back(e);
          }
        }
    append_row(out, row);
  }
  return out;
}

FarList far_list_from_rows(const AtomSet& atoms, const NeighborCutoffs& cut, const FarRowsF32& rows) {
  check_cutoffs(cut);
  const std::size_t n = atoms.nall();
  if (rows.nall != n) throw SystemError("far_list_from_rows: rows and atoms differ in size");
  for (std::size_t i = 0; i < n; ++i)
    if (rows.count[i] > rows.cap) throw SystemError("far_list_from_rows: row overflow");
  // Rows are independent: contiguous blocks of rows are filtered and sorted by worker threads into private fragments, which are then concatenated in
  // block order, so the result does not depend on the number of threads.
  const std::size_t nthreads = std::max<std::size_t>(1, std::min<std::size_t>(8, std::min<std::size_t>(std::thread::hardware_concurrency(), n / 4096 + 1)));
  struct Fragment { std::vector<std::size_t> row_len; std::vector<std::int32_t> nbr; std::vector<double> dist, dvec; };
  std::vector<Fragment> frag(nthreads);
  auto work = [&](std::size_t t) {
    const std::size_t lo = n * t / nthreads, hi = n * (t + 1) / nthreads;
    Fragment& f = frag[t];
    f.row_len.reserve(hi - lo);
    std::vector<RowEntry> row;
    for (std::size_t i = lo; i < hi; ++i) {
      const double rc = cut.row_cut(i, atoms.nlocal);
      const double rc2 = rc * rc;
      const double* xi = &atoms.x[3 * i];
      const std::uint32_t cnt = rows.count[i];
      row.clear();
      for (std::uint32_t e = 0; e < cnt; ++e) {
        const std::int32_t j = rows.nbr[i * rows.cap + e];
        RowEntry r;
        double d2;
        if (!within(xi, &atoms.x[3 * static_cast<std::size_t>(j)], rc2, r.dv, d2)) continue;
        r.j = j;
        r.d = std::sqrt(d2);
        row.push_back(r);
      }
      std::sort(row.begin(), row.end(), [](const RowEntry& a, const RowEntry& b) { return a.j < b.j; });
      for (const auto& e : row) {
        f.nbr.push_back(e.j);
        f.dist.push_back(e.d);
        f.dvec.insert(f.dvec.end(), e.dv, e.dv + 3);
      }
      f.row_len.push_back(row.size());
    }
  };
  std::vector<std::thread> pool;
  for (std::size_t t = 1; t < nthreads; ++t) pool.emplace_back(work, t);
  work(0);
  for (auto& th : pool) th.join();
  FarList out;
  std::size_t total = 0;
  for (const auto& f : frag) total += f.nbr.size();
  out.row_start.reserve(n + 1);
  out.row_start.push_back(0);
  out.nbr.reserve(total); out.dist.reserve(total); out.dvec.reserve(3 * total);
  for (const auto& f : frag) {
    for (const std::size_t len : f.row_len) out.row_start.push_back(out.row_start.back() + len);
    out.nbr.insert(out.nbr.end(), f.nbr.begin(), f.nbr.end());
    out.dist.insert(out.dist.end(), f.dist.begin(), f.dist.end());
    out.dvec.insert(out.dvec.end(), f.dvec.begin(), f.dvec.end());
  }
  return out;
}

FarList build_far_list_bruteforce(const AtomSet& atoms, const NeighborCutoffs& cut) {
  check_cutoffs(cut);
  const std::size_t n = atoms.nall();
  FarList out;
  out.row_start.push_back(0);
  std::vector<RowEntry> row;
  for (std::size_t i = 0; i < n; ++i) {
    const double rc = cut.row_cut(i, atoms.nlocal);
    for (std::size_t j = i + 1; j < n; ++j) {
      RowEntry e;
      double d2;
      if (!within(&atoms.x[3 * i], &atoms.x[3 * j], rc * rc, e.dv, d2)) continue;
      e.j = static_cast<std::int32_t>(j);
      e.d = std::sqrt(d2);
      row.push_back(e);
    }
    append_row(out, row);
  }
  return out;
}

// ---- owner-computes rules ----------------------------------------------------------------------------------------
PairClass classify_nonbonded_entry(std::size_t /*i*/, std::size_t j, std::size_t nlocal, std::int64_t tag_i, std::int64_t tag_j,
                                   const double dvec[3]) noexcept {
  constexpr double SMALL = 0.0001;  // reaxff_nonbonded.cpp:84
  if (j < nlocal) return PairClass::OwnedOwned;
  if (tag_i < tag_j) return PairClass::OwnedGhost;
  if (tag_i == tag_j) {
    bool flag = false;
    if (dvec[2] > SMALL) flag = true;
    else if (std::fabs(dvec[2]) < SMALL) {
      if (dvec[1] > SMALL) flag = true;
      else if (std::fabs(dvec[1]) < SMALL && dvec[0] > SMALL) flag = true;
    }
    return flag ? PairClass::SelfImage : PairClass::None;
  }
  return PairClass::None;
}

PairCounts count_nonbonded_pairs(const AtomSet& atoms, const FarList& list, const NeighborCutoffs& cut) {
  PairCounts c;
  for (std::size_t i = 0; i < atoms.nlocal; ++i)
    for (std::size_t p = list.row_start[i]; p < list.row_start[i + 1]; ++p) {
      if (list.dist[p] > cut.nonb) continue;
      const auto j = static_cast<std::size_t>(list.nbr[p]);
      switch (classify_nonbonded_entry(i, j, atoms.nlocal, atoms.tag[i], atoms.tag[j], &list.dvec[3 * p])) {
        case PairClass::OwnedOwned: ++c.oo; break;
        case PairClass::OwnedGhost: ++c.og; break;
        case PairClass::SelfImage: ++c.self; break;
        case PairClass::None: break;
      }
    }
  return c;
}

// ---- device-list input -------------------------------------------------------------------------------------------
DeviceListInput make_device_list_input(const AtomSet& atoms, const Box& box, const NeighborCutoffs& cut, double margin) {
  check_cutoffs(cut);
  if (!(margin >= 0.0)) throw SystemError("make_device_list_input: margin must be >= 0");
  if (atoms.nall() > 0xFFFFFFF0u) throw SystemError("make_device_list_input: too many atoms for 32-bit indices");
  DeviceListInput in;
  in.nall = static_cast<std::uint32_t>(atoms.nall());
  in.nlocal = static_cast<std::uint32_t>(atoms.nlocal);
  in.origin = box.origin;
  in.x.resize(3 * atoms.nall());
  in.x_lo.resize(3 * atoms.nall());
  double maxabs = 0.0;
  for (std::size_t i = 0; i < atoms.nall(); ++i)
    for (std::size_t c = 0; c < 3; ++c) {
      const double r = atoms.x[3 * i + c] - box.origin[c];
      maxabs = std::max(maxabs, std::fabs(r));
      in.x[3 * i + c] = static_cast<float>(r);
      in.x_lo[3 * i + c] = static_cast<float>(r - static_cast<double>(in.x[3 * i + c]));
    }
  // float coordinates carry <= 0.5 ulp = 0.5*FLT_EPSILON*|r| error each; a distance error is below 4*FLT_EPSILON*maxabs.
  in.list_margin = std::max(margin, 8.0 * static_cast<double>(FLT_EPSILON) * maxabs);
  const double m = in.list_margin;
  in.rc2_owned = static_cast<float>((cut.nonb + m) * (cut.nonb + m));
  in.rc2_ghost = static_cast<float>((cut.bond + m) * (cut.bond + m));
  in.grid = build_cell_grid(atoms.x, std::max(cut.nonb, cut.bond) + 2.0 * m);
  return in;
}

std::uint32_t FarRowsF32::max_count() const noexcept {
  std::uint32_t m = 0;
  for (std::uint32_t c : count) m = std::max(m, c);
  return m;
}

FarRowsF32 build_far_rows_with_growth(const std::function<FarRowsF32(std::uint32_t)>& launch, std::uint32_t initial_cap, std::uint32_t nall,
                                      unsigned max_attempts, unsigned* attempts) {
  std::uint32_t cap = initial_cap == 0 ? 1u : initial_cap;
  for (unsigned a = 1; a <= max_attempts; ++a) {
    if (static_cast<std::uint64_t>(nall) * cap > 0xFFFFFFFFull) throw SystemError("far rows: nall * capacity exceeds 32-bit indexing");
    FarRowsF32 rows = launch(cap);
    if (attempts) *attempts = a;
    const std::uint32_t need = rows.max_count();
    if (need <= cap) return rows;
    cap = (need + 7u) / 8u * 8u;
  }
  throw SystemError("far rows: row capacity still too small after " + std::to_string(max_attempts) + " attempts");
}

RowComparison compare_rows_to_far_list(const FarRowsF32& rows, const FarList& cpu, const AtomSet& atoms, const NeighborCutoffs& cut, double margin) {
  RowComparison r;
  const std::size_t n = atoms.nall();
  if (rows.nall != n || rows.count.size() != n || cpu.rows() != n) {
    r.structural = 1;
    return r;
  }
  r.cpu_entries = cpu.entries();
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t cnt = rows.count[i];
    if (cnt > rows.cap) {
      r.overflow = true;
      continue;
    }
    r.device_entries += cnt;
    const double rc = cut.row_cut(i, atoms.nlocal);
    std::vector<std::int32_t> dev(rows.nbr.begin() + static_cast<std::ptrdiff_t>(i * rows.cap), rows.nbr.begin() + static_cast<std::ptrdiff_t>(i * rows.cap + cnt));
    std::vector<std::size_t> pos(dev.size());
    std::iota(pos.begin(), pos.end(), std::size_t{0});
    std::sort(pos.begin(), pos.end(), [&](std::size_t a, std::size_t b) { return dev[a] < dev[b]; });
    std::vector<std::int32_t> sorted;
    for (std::size_t k : pos) sorted.push_back(dev[k]);
    for (std::size_t k = 0; k < sorted.size(); ++k) {
      if (sorted[k] <= static_cast<std::int32_t>(i) || static_cast<std::size_t>(sorted[k]) >= n || (k > 0 && sorted[k] == sorted[k - 1])) ++r.structural;
    }
    // superset check against the CPU row (ascending j)
    std::size_t a = cpu.row_start[i];
    const std::size_t aend = cpu.row_start[i + 1];
    std::size_t b = 0;
    while (a < aend && b < sorted.size()) {
      if (cpu.nbr[a] == sorted[b]) { ++a; ++b; }
      else if (cpu.nbr[a] < sorted[b]) { ++r.missing; ++a; }
      else { ++b; }   // device-only entry: classified below
    }
    r.missing += aend - a;
    // device-only entries: legal only inside the margin band; r2 consistency for every device entry
    for (std::size_t k = 0; k < pos.size(); ++k) {
      const std::int32_t j = dev[k];
      if (j <= static_cast<std::int32_t>(i) || static_cast<std::size_t>(j) >= n) continue;
      const double dx = atoms.x[3 * static_cast<std::size_t>(j)] - atoms.x[3 * i], dy = atoms.x[3 * static_cast<std::size_t>(j) + 1] - atoms.x[3 * i + 1],
                   dz = atoms.x[3 * static_cast<std::size_t>(j) + 2] - atoms.x[3 * i + 2];
      const double d2 = dx * dx + dy * dy + dz * dz, d = std::sqrt(d2);
      const bool in_cpu = std::binary_search(cpu.nbr.begin() + static_cast<std::ptrdiff_t>(cpu.row_start[i]), cpu.nbr.begin() + static_cast<std::ptrdiff_t>(cpu.row_start[i + 1]), j);
      if (!in_cpu) {
        if (d > rc + 2.0 * margin) ++r.illegal_extra;
        else ++r.band_extra;
      }
      const double dev_r2 = static_cast<double>(rows.r2[i * rows.cap + k]);
      if (std::fabs(dev_r2 - d2) > 2.0 * (d + margin) * margin + 1e-6) ++r.bad_r2;
    }
  }
  return r;
}

// ---- deterministic reductions ------------------------------------------------------------------------------------
std::vector<float> fixed_order_partials_f32(std::span<const float> v, std::uint32_t chunk_len) {
  if (chunk_len == 0) throw SystemError("fixed_order_partials_f32: chunk_len must be > 0");
  const std::size_t nchunk = (v.size() + chunk_len - 1) / chunk_len;
  std::vector<float> part(nchunk, 0.0f);
  for (std::size_t c = 0; c < nchunk; ++c) {
    float s = 0.0f;
    const std::size_t end = std::min(v.size(), (c + 1) * chunk_len);
    for (std::size_t e = c * chunk_len; e < end; ++e) s += v[e];
    part[c] = s;
  }
  return part;
}

float fixed_order_sum_f32(std::span<const float> v, std::uint32_t chunk_len) {
  float s = 0.0f;
  for (float p : fixed_order_partials_f32(v, chunk_len)) s += p;
  return s;
}

}  // namespace reaxmetal
