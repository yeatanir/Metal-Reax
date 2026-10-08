// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// Simulation box and the *expanded atom set* the engine consumes (ENGINE_SPEC 3, ADR-013 "ghost-native"):
//   atoms [0, nlocal)        owned atoms
//   atoms [nlocal, nall)     ghost atoms, each = an owner atom plus an integer lattice shift
// Periodic images are always a host concern: LAMMPS supplies its own ghosts through the adapter; the standalone harness
// builds the same kind of set with expand_images() (neighbor.hpp). No kernel does minimum-image or shift arithmetic.
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace reaxmetal {

using Vec3 = std::array<double, 3>;

class SystemError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// General parallelepiped: origin + f0*a + f1*b + f2*c, f in [0,1). LAMMPS' restricted triclinic form (a=(lx,0,0),
// b=(xy,ly,0), c=(xz,yz,lz)) is the special case produced by from_lammps(). `periodic[d]` applies to lattice vector d.
struct Box {
  Vec3 origin{0, 0, 0};
  std::array<Vec3, 3> vectors{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  std::array<bool, 3> periodic{false, false, false};

  static Box orthogonal(const Vec3& lo, const Vec3& hi, std::array<bool, 3> periodic);
  // tilt = (xy, xz, yz) as in the LAMMPS data file
  static Box from_lammps(const Vec3& lo, const Vec3& hi, const Vec3& tilt, std::array<bool, 3> periodic);

  double volume() const;                 // signed; validate() requires > 0 (right-handed)
  std::array<double, 3> heights() const; // distance between the two faces spanned by the other two lattice vectors
  Vec3 to_fractional(const Vec3& r) const;
  Vec3 to_cartesian(const Vec3& f) const;
  Vec3 lattice_shift(const std::array<int, 3>& s) const;  // s0*a + s1*b + s2*c
  void validate() const;                 // finite, right-handed, non-degenerate; throws SystemError
};

struct AtomSet {
  std::size_t nlocal = 0;
  std::vector<double> x;              // 3 * nall, Cartesian
  std::vector<int> type;              // force-field type index (>= 0) or -1 (atom not handled by this style)
  std::vector<std::int64_t> tag;      // LAMMPS atom id; ghosts carry their owner's tag
  std::vector<std::int32_t> owner;    // owner[i] == i for owned atoms; the owned index of a ghost
  std::vector<std::array<std::int32_t, 3>> shift;  // lattice shift of a ghost relative to its owner ({0,0,0} if owned)

  std::size_t nall() const noexcept { return type.size(); }
  std::size_t nghost() const noexcept { return type.size() - nlocal; }
  Vec3 position(std::size_t i) const noexcept { return {x[3 * i], x[3 * i + 1], x[3 * i + 2]}; }
  // Throws SystemError unless sizes agree and every ghost is exactly (owner position + shift . box) to `tol` Angstrom
  // (the contract LAMMPS_INTEGRATION.md section 4 measured on every ghost of every step).
  void validate(const Box& box, double tol = 1e-9) const;
};

}  // namespace reaxmetal
