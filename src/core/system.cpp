// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/system.hpp"

#include <cmath>
#include <string>

namespace reaxmetal {
namespace {
Vec3 cross(const Vec3& a, const Vec3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
}  // namespace

Box Box::orthogonal(const Vec3& lo, const Vec3& hi, std::array<bool, 3> periodic) {
  return from_lammps(lo, hi, {0, 0, 0}, periodic);
}

Box Box::from_lammps(const Vec3& lo, const Vec3& hi, const Vec3& tilt, std::array<bool, 3> periodic) {
  Box b;
  b.origin = lo;
  b.vectors[0] = {hi[0] - lo[0], 0, 0};
  b.vectors[1] = {tilt[0], hi[1] - lo[1], 0};
  b.vectors[2] = {tilt[1], tilt[2], hi[2] - lo[2]};
  b.periodic = periodic;
  b.validate();
  return b;
}

double Box::volume() const { return dot(vectors[0], cross(vectors[1], vectors[2])); }

std::array<double, 3> Box::heights() const {
  const double v = std::fabs(volume());
  return {v / norm(cross(vectors[1], vectors[2])), v / norm(cross(vectors[2], vectors[0])), v / norm(cross(vectors[0], vectors[1]))};
}

Vec3 Box::to_fractional(const Vec3& r) const {
  const Vec3 d{r[0] - origin[0], r[1] - origin[1], r[2] - origin[2]};
  const double det = volume();
  return {dot(d, cross(vectors[1], vectors[2])) / det, dot(d, cross(vectors[2], vectors[0])) / det, dot(d, cross(vectors[0], vectors[1])) / det};
}

Vec3 Box::to_cartesian(const Vec3& f) const {
  Vec3 r = origin;
  for (int k = 0; k < 3; ++k)
    for (int c = 0; c < 3; ++c) r[static_cast<std::size_t>(c)] += f[static_cast<std::size_t>(k)] * vectors[static_cast<std::size_t>(k)][static_cast<std::size_t>(c)];
  return r;
}

Vec3 Box::lattice_shift(const std::array<int, 3>& s) const {
  Vec3 r{0, 0, 0};
  for (std::size_t k = 0; k < 3; ++k)
    for (std::size_t c = 0; c < 3; ++c) r[c] += static_cast<double>(s[k]) * vectors[k][c];
  return r;
}

void Box::validate() const {
  for (double o : origin)
    if (!std::isfinite(o)) throw SystemError("box origin is not finite");
  for (const auto& v : vectors)
    for (double c : v)
      if (!std::isfinite(c)) throw SystemError("box vector is not finite");
  const double vol = volume();
  if (!(vol > 0.0)) throw SystemError("box is degenerate or left-handed (volume " + std::to_string(vol) + " <= 0)");
}

void AtomSet::validate(const Box& box, double tol) const {
  const std::size_t n = nall();
  if (nlocal > n) throw SystemError("nlocal > nall");
  if (x.size() != 3 * n || tag.size() != n || owner.size() != n || shift.size() != n) throw SystemError("AtomSet arrays have inconsistent sizes");
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t c = 0; c < 3; ++c)
      if (!std::isfinite(x[3 * i + c])) throw SystemError("non-finite coordinate of atom " + std::to_string(i));
    if (i < nlocal) {
      if (owner[i] != static_cast<std::int32_t>(i) || shift[i] != std::array<std::int32_t, 3>{0, 0, 0})
        throw SystemError("owned atom " + std::to_string(i) + " has an owner/shift");
      continue;
    }
    if (distributed) continue;   // owners live on other ranks; the ghost contract is then LAMMPS' own
    const std::int32_t o = owner[i];
    if (o < 0 || static_cast<std::size_t>(o) >= nlocal) throw SystemError("ghost " + std::to_string(i) + " has an invalid owner");
    const auto os = static_cast<std::size_t>(o);
    if (tag[i] != tag[os] || type[i] != type[os]) throw SystemError("ghost " + std::to_string(i) + " differs from its owner in tag/type");
    const Vec3 sh = box.lattice_shift({shift[i][0], shift[i][1], shift[i][2]});
    for (std::size_t c = 0; c < 3; ++c)
      if (std::fabs(x[3 * i + c] - (x[3 * os + c] + sh[c])) > tol)
        throw SystemError("ghost " + std::to_string(i) + " is not owner + lattice shift (component " + std::to_string(c) + ")");
  }
}

}  // namespace reaxmetal
