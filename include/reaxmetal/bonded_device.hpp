// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// Host side of the device bonded-terms pipeline (kernels in src/metal/shaders/reaxmetal_m6.metal): packing of the parameter tables, the
// work-array layout, the grow-and-retry orchestration and the FP64 finish. Backend independent: the Metal context and the CPU emulation of the
// kernels (tests) both implement BondedBackend, so the orchestration is tested without a GPU.
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "reaxmetal/bonded.hpp"
#include "reaxmetal/forcefield.hpp"
#include "reaxmetal/neighbor.hpp"
#include "reaxmetal/system.hpp"

namespace reaxmetal {

struct BondedDeviceInput {
  DeviceListInput list;                    // float positions (relative to the box corner) and cell grid
  std::vector<std::int32_t> type, tag;     // per atom; type -1 = not handled
  std::vector<float> sb_f, tb_f, thb_sets, fb_f, hb_f, gp;
  std::vector<std::int32_t> sb_i, tb_i, thb_idx, fb_has;
  std::uint32_t ntypes = 0, enobonds = 1;
  float bond_cut = 0, bo_cut = 0, thb_cut = 0, thb_cutsq = 0, hbond_cut = 0;
};

// work-array layout for given capacities (offsets are in elements of wf / wi)
struct BondedLayout {
  std::uint32_t N = 0, nlocal = 0, B = 0, H = 0;
  std::size_t NB = 0, o_atom = 0, o_tkl = 0, o_tfl = 0, o_hf = 0, wf_size = 0, o_iatom = 0, o_hi = 0, wi_size = 0;
};
BondedLayout bonded_layout(std::uint32_t N, std::uint32_t nlocal, std::uint32_t B, std::uint32_t H);

struct BondedStep {
  const char* kernel;
  std::uint32_t threads;
};

// A device (or its CPU emulation). setup() allocates zero-initialised work arrays and uploads the inputs; run() executes the steps in order
// (each completes before the next starts); read_*() copy work-array ranges back.
class BondedBackend {
 public:
  virtual ~BondedBackend() = default;
  virtual void setup(const BondedDeviceInput& in, const BondedLayout& layout) = 0;
  virtual void run(std::span<const BondedStep> steps) = 0;
  virtual void read_float(std::size_t offset, std::size_t count, float* dst) = 0;
  virtual void read_int(std::size_t offset, std::size_t count, std::int32_t* dst) = 0;
  virtual double gpu_seconds() const { return 0.0; }
};

struct BondedDeviceOutput {
  std::vector<float> grad;                 // 3 * nall, dE/dx of all bonded terms
  std::array<double, kEnergyTermCount> e{};// per-term energies, FP64 sums of the per-atom FP32 partials in atom order
  std::uint32_t bond_cap = 0, hbond_cap = 0;
  unsigned attempts = 0;
};

BondedDeviceInput make_bonded_device_input(const ForceField& ff, const ControlParams& ctl, const AtomSet& atoms, const Box& box, const BondedOptions& opt,
                                              const DeviceListInput* shared_list = nullptr);   // shared_list: the step's list when another consumer already built it

// Runs rm_b_build / rm_h_build, grows the capacities if a row overflowed (then restarts), runs the rest and reads the results.
BondedDeviceOutput run_bonded_pipeline(BondedBackend& backend, const BondedDeviceInput& in, std::uint32_t initial_bond_cap = 16, std::uint32_t initial_hbond_cap = 16);

BondedResult finish_bonded(const BondedDeviceOutput& out, std::size_t nall);

}  // namespace reaxmetal
