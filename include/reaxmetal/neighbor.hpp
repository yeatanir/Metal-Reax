// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// M3: image expander, cell grid and the "far neighbor list" of the engine (CPU-64 reference implementation).
//
// Far list = what pinned LAMMPS hands to the ReaxFF kernels (pair_reaxff.cpp:629-680, built from a half, newton-off
// neighbor list with ghost neighbors, npair_bin_ghost.cpp):
//   row i (every owned AND ghost atom) holds the atoms j > i (index order) with |x_j - x_i| <= row cutoff,
//   row cutoff = nonb_cut for owned rows, bond_cut for ghost rows (ENGINE_SPEC Q-08).
// Owner-computes counting rules for pair terms are NOT part of the list; they are applied by the consumer
// (nonbonded_pair_counted(), ENGINE_SPEC 3.1).
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "reaxmetal/system.hpp"

namespace reaxmetal {

struct NeighborCutoffs {
  double nonb = 10.0;   // nonb_cut: row cutoff of owned atoms
  double bond = 5.0;    // bond_cut: row cutoff of ghost atoms
  double hbond = 7.5;   // hbond_cut: only enters the required ghost shell
  double row_cut(std::size_t i, std::size_t nlocal) const noexcept { return i < nlocal ? nonb : bond; }
  // Smallest ghost shell for which owned-atom results are those of the reference (ENGINE_SPEC 3, LAMMPS_INTEGRATION C3).
  double required_shell() const noexcept;
};

struct ExpandOptions {
  double shell = 12.0;        // ghost shell width (Angstrom): LAMMPS uses cutforce + skin
  bool wrap_owned = true;     // wrap owned atoms into the primary cell first (LAMMPS does so when it re-neighbors)
  std::size_t max_atoms = 50'000'000;  // refuse (SystemError) to build absurd sets
};

// Owned atoms first (input order, wrapped if requested), then ghosts ordered by (shift s0, s1, s2, owner index) with
// the (0,0,0) shift omitted. A ghost is every image whose fractional coordinates lie within `shell / height_d` of
// [0,1] along each periodic dimension - the slab test LAMMPS' Comm::borders applies in lamda coordinates.
AtomSet expand_images(const Box& box, std::span<const double> x_owned, std::span<const int> type, std::span<const std::int64_t> tag,
                      const ExpandOptions& opt = {});

// ---- cell grid ---------------------------------------------------------------------------------------------------
// Uniform grid over the bounding box of all atoms (ghosts included, so no periodic wrap-around is ever needed).
// Cells are at least `min_cell` wide, therefore any pair within `min_cell` lies in the same or an adjacent cell.
struct CellGrid {
  Vec3 origin{};
  double cell_size[3]{};
  std::uint32_t ncell[3]{1, 1, 1};
  std::vector<std::uint32_t> atom_cell;    // linear cell id per atom
  std::vector<std::uint32_t> cell_start;   // ncell_total + 1
  std::vector<std::uint32_t> cell_items;   // atom indices, ascending within a cell
  std::uint32_t total() const noexcept { return ncell[0] * ncell[1] * ncell[2]; }
  void coords(std::uint32_t cell, std::uint32_t c[3]) const noexcept;
};
// If the grid would need more than `max_cells` cells (sparse/huge boxes) the cells are coarsened (never made smaller than min_cell).
CellGrid build_cell_grid(std::span<const double> x, double min_cell, std::size_t max_cells = std::size_t{1} << 26);

// ---- CPU-64 far list ---------------------------------------------------------------------------------------------
struct FarList {
  std::vector<std::size_t> row_start;   // nall + 1
  std::vector<std::int32_t> nbr;        // j, ascending within a row
  std::vector<double> dist;             // |x_j - x_i| (double, formula of the reference: sqrt(dx^2+dy^2+dz^2))
  std::vector<double> dvec;             // 3 per entry, x_j - x_i
  std::size_t rows() const noexcept { return row_start.empty() ? 0 : row_start.size() - 1; }
  std::size_t entries() const noexcept { return nbr.size(); }
};
FarList build_far_list(const AtomSet& atoms, const NeighborCutoffs& cut);
// O(N^2) enumeration with the same semantics; the oracle for tests (never for production use).
FarList build_far_list_bruteforce(const AtomSet& atoms, const NeighborCutoffs& cut);

// ---- owner-computes rules (ENGINE_SPEC 3.1) -----------------------------------------------------------------------
// Rule applied by vdW_Coulomb_Energy (reaxff_nonbonded.cpp:104-117) to an owned row i and an entry j with d <= nonb_cut.
// dvec = x_j - x_i. SMALL = 1e-4 as upstream.
enum class PairClass { None, OwnedOwned, OwnedGhost, SelfImage };
PairClass classify_nonbonded_entry(std::size_t i, std::size_t j, std::size_t nlocal, std::int64_t tag_i, std::int64_t tag_j,
                                   const double dvec[3]) noexcept;

struct PairCounts {
  std::size_t oo = 0, og = 0, self = 0;   // matches the reference tallies vdw.oo / vdw.og / vdw.self
  bool operator==(const PairCounts&) const = default;
};
PairCounts count_nonbonded_pairs(const AtomSet& atoms, const FarList& list, const NeighborCutoffs& cut);

// ---- device-list input (shared by the Metal backend and the CPU emulation of its kernels) -------------------------
// Positions are float and relative to `origin` (NUMERICAL_POLICY 4.1, LAMMPS_INTEGRATION C11). The grid and the cell of
// every atom are decided on the host in double precision and handed to the device; kernels never recompute them.
struct DeviceListInput {
  std::uint32_t nall = 0, nlocal = 0;
  Vec3 origin{};
  std::vector<float> x;                     // 3 * nall, x - origin
  CellGrid grid;
  float rc2_owned = 0, rc2_ghost = 0;       // squared row cutoffs (float)
  double list_margin = 0;                   // cutoffs were enlarged by this much before squaring
};
// `margin` enlarges both row cutoffs so that float rounding can never lose a pair that the double precision list has
// (see VALIDATION NBR-2: rows are a superset of CPU-64 pairs with r <= cutoff and a subset of those with r <= cutoff + 2 margin).
DeviceListInput make_device_list_input(const AtomSet& atoms, const Box& box, const NeighborCutoffs& cut, double margin = 1e-3);

// Fixed-capacity rows as the device produces them: entries 0..min(count,cap)-1 of row i are valid, `count[i]` is the true
// number of neighbors (may exceed cap, which tells the host how much to grow).
struct FarRowsF32 {
  std::uint32_t nall = 0, cap = 0;
  std::vector<std::uint32_t> count;
  std::vector<std::int32_t> nbr;   // nall * cap
  std::vector<float> r2;           // nall * cap
  std::uint32_t max_count() const noexcept;
};

// Grow-and-retry (LAMMPS_INTEGRATION S6): `launch(cap)` runs the row kernel with that capacity. If any row needed more than
// `cap`, the true counts are known, so the next attempt uses the exact need (rounded up to a multiple of 8). Throws
// SystemError after `max_attempts` or if nall * cap would exceed 32-bit indexing. `attempts` (optional) receives the number of launches.
FarRowsF32 build_far_rows_with_growth(const std::function<FarRowsF32(std::uint32_t)>& launch, std::uint32_t initial_cap, std::uint32_t nall,
                                      unsigned max_attempts = 4, unsigned* attempts = nullptr);

// Verification of device rows against the CPU-64 list (VALIDATION NBR-1/NBR-2). Contract of the float rows:
//   * superset: every CPU-64 pair (d <= row cutoff) is present            -> `missing` must be 0
//   * bounded:  every device pair has CPU-64 distance <= cutoff + 2*margin -> `illegal_extra` must be 0
//   * the stored r2 equals the double distance squared to 2*(d + margin)*margin
// Pairs in the band (cutoff, cutoff + 2*margin] that the device includes are legal and only counted (`band_extra`).
struct RowComparison {
  std::size_t cpu_entries = 0, device_entries = 0;
  std::size_t missing = 0, illegal_extra = 0, band_extra = 0, bad_r2 = 0, structural = 0;  // structural: j <= i, j >= nall, duplicates
  bool overflow = false;                                                                   // some row has count > cap
  bool ok() const noexcept { return !overflow && missing == 0 && illegal_extra == 0 && bad_r2 == 0 && structural == 0; }
};
RowComparison compare_rows_to_far_list(const FarRowsF32& rows, const FarList& cpu, const AtomSet& atoms, const NeighborCutoffs& cut, double margin);

// ---- deterministic reductions ------------------------------------------------------------------------------------
// Canonical fixed-order float sum: element e belongs to chunk e / chunk_len; chunks are summed sequentially in float, then
// the chunk partials sequentially in float. The Metal reduction kernels execute exactly this order, so CPU and GPU results
// are bitwise equal, run after run (VALIDATION FORCE-2). The final FP64 accumulation of partials (host) is fixed-order too.
std::vector<float> fixed_order_partials_f32(std::span<const float> v, std::uint32_t chunk_len);  // one partial per chunk
float fixed_order_sum_f32(std::span<const float> v, std::uint32_t chunk_len);                     // sequential sum of the partials

}  // namespace reaxmetal
