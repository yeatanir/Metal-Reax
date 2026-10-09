// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Metal host layer (milestone M3): Objective-C++ against the system Metal and Foundation frameworks, shaders compiled at run
// time from source (no Xcode / `metal` command-line compiler required, ADR-008, ADR-023). Compiled with -fobjc-arc.
//
// STATUS: written. NOT compiled with the Apple SDK, NOT executed. tests/check_objcxx_syntax.sh checks it only against
// hand-written stub headers. First real build and run: tools/mac/step1_bringup.sh (docs/VALIDATION.md MET-1).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "reaxmetal/metal_backend.hpp"
#include "reaxmetal_m3_types.h"
#include "reaxmetal_m4_types.h"
#include "reaxmetal_m6_types.h"

namespace reaxmetal::mtl {

namespace {

std::string to_std(NSString* s) { return s ? std::string([s UTF8String]) : std::string(); }

[[noreturn]] void fail(const std::string& what, NSError* err) {
  throw MetalError(err ? what + ": " + to_std([err localizedDescription]) : what);
}

struct Dispatch {
  id<MTLComputePipelineState> pso = nil;
  NSUInteger threads = 0;
  std::vector<id<MTLBuffer>> buffers;   // bound at indices 0..k-1
  std::array<unsigned char, 128> params{};
  NSUInteger param_len = 0;
  NSUInteger param_index = 0;
};

template <class P>
void set_params(Dispatch& d, const P& p, NSUInteger index) {
  static_assert(sizeof(P) <= 128, "parameter block too large for Dispatch");
  std::memcpy(d.params.data(), &p, sizeof(P));
  d.param_len = sizeof(P);
  d.param_index = index;
}

}  // namespace

bool compiled_with_metal() noexcept { return true; }

struct Context::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  std::map<std::string, id<MTLComputePipelineState>> pipelines;
  std::vector<KernelInfo> kernels;
  std::string log;
  double compile_seconds = 0.0;
  double gpu_seconds = 0.0;

  id<MTLBuffer> buffer(NSUInteger bytes) {
    id<MTLBuffer> b = [device newBufferWithLength:(bytes > 16 ? bytes : 16) options:MTLResourceStorageModeShared];
    if (b == nil) throw MetalError("could not allocate a " + std::to_string(bytes) + " byte shared buffer");
    return b;
  }
  // persistent buffers for the per-step pipelines: a slot keeps its buffer while it is big enough (no reallocation, no page faults per step)
  std::map<std::string, id<MTLBuffer>> slots;
  std::uint32_t last_bond_cap = 0, last_hbond_cap = 0;
  std::uint32_t qeq_nlocal = 0, qeq_cap = 0;   // state of the last qeq_setup (its buffers live in the slots "qeq_*")
  RmQeqParams qeq_params{};
  id<MTLBuffer> slot(const std::string& name, NSUInteger bytes) {
    id<MTLBuffer> b = slots[name];
    if (b == nil || [b length] < bytes || [b length] > 4 * (bytes > 16 ? bytes : 16)) {
      b = buffer(bytes);
      slots[name] = b;
    }
    return b;
  }
  id<MTLBuffer> slot_from(const std::string& name, const void* data, NSUInteger bytes) {
    id<MTLBuffer> b = slot(name, bytes);
    if (bytes > 0) std::memcpy([b contents], data, bytes);
    return b;
  }
  id<MTLBuffer> buffer_from(const void* data, NSUInteger bytes) {
    id<MTLBuffer> b = buffer(bytes);
    if (bytes > 0) std::memcpy([b contents], data, bytes);
    return b;
  }
  Dispatch make(const char* name, NSUInteger threads) {
    auto it = pipelines.find(name);
    if (it == pipelines.end()) throw MetalError(std::string("kernel not found in the compiled library: ") + name);
    Dispatch d;
    d.pso = it->second;
    d.threads = threads;
    return d;
  }

  // Encodes the dispatches in order into ONE serial compute encoder (a later dispatch sees the results of the earlier ones),
  // runs the command buffer to completion and records the GPU time.
  void run(const std::vector<Dispatch>& list) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (cb == nil || enc == nil) throw MetalError("could not create a command buffer / compute encoder");
    for (const Dispatch& d : list) {
      if (d.threads == 0) continue;
      [enc setComputePipelineState:d.pso];
      for (std::size_t k = 0; k < d.buffers.size(); ++k) [enc setBuffer:d.buffers[k] offset:0 atIndex:k];
      if (d.param_len > 0) [enc setBytes:d.params.data() length:d.param_len atIndex:d.param_index];
      NSUInteger tg = [d.pso maxTotalThreadsPerThreadgroup];
      if (tg > 64) tg = 64;
      [enc dispatchThreads:MTLSizeMake(d.threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] != MTLCommandBufferStatusCompleted) fail("command buffer did not complete", [cb error]);
    gpu_seconds = [cb GPUEndTime] - [cb GPUStartTime];
  }

  FarRowsF32 launch_far_rows(const DeviceListInput& in, std::uint32_t cap) {
    FarRowsF32 rows;
    rows.nall = in.nall;
    rows.cap = cap;
    rows.count.assign(in.nall, 0);
    rows.nbr.assign(static_cast<std::size_t>(in.nall) * cap, -1);
    rows.r2.assign(static_cast<std::size_t>(in.nall) * cap, 0.0f);
    if (in.nall == 0) return rows;
    id<MTLBuffer> bx = buffer_from(in.x.data(), in.x.size() * sizeof(float));
    id<MTLBuffer> bcell = buffer_from(in.grid.atom_cell.data(), in.grid.atom_cell.size() * sizeof(std::uint32_t));
    id<MTLBuffer> bstart = buffer_from(in.grid.cell_start.data(), in.grid.cell_start.size() * sizeof(std::uint32_t));
    id<MTLBuffer> bitems = buffer_from(in.grid.cell_items.data(), in.grid.cell_items.size() * sizeof(std::uint32_t));
    id<MTLBuffer> bnbr = buffer_from(rows.nbr.data(), rows.nbr.size() * sizeof(std::int32_t));   // pre-filled with -1: deterministic bytes
    id<MTLBuffer> br2 = buffer_from(rows.r2.data(), rows.r2.size() * sizeof(float));
    id<MTLBuffer> bcount = buffer_from(rows.count.data(), rows.count.size() * sizeof(std::uint32_t));
    RmFarRowsParams p{};
    p.nall = in.nall; p.nlocal = in.nlocal; p.cap = cap;
    p.ncx = in.grid.ncell[0]; p.ncy = in.grid.ncell[1]; p.ncz = in.grid.ncell[2];
    p.rc2_owned = in.rc2_owned; p.rc2_ghost = in.rc2_ghost;
    Dispatch d = make("rm_far_rows", in.nall);
    d.buffers = {bx, bcell, bstart, bitems, bnbr, br2, bcount};
    set_params(d, p, 7);
    run({d});
    std::memcpy(rows.nbr.data(), [bnbr contents], rows.nbr.size() * sizeof(std::int32_t));
    std::memcpy(rows.r2.data(), [br2 contents], rows.r2.size() * sizeof(float));
    std::memcpy(rows.count.data(), [bcount contents], rows.count.size() * sizeof(std::uint32_t));
    return rows;
  }
};

namespace {
// Metal implementation of the bonded pipeline backend: every kernel takes the same 18 buffers + the RmBParams block (index 18)
struct MetalBondedBackend final : BondedBackend {
  Context::Impl* m;
  std::vector<id<MTLBuffer>> in_bufs;    // x type tag atom_cell cell_start cell_items sb_f sb_i tb_f tb_i thb_idx thb_sets fb_f fb_has hb_f gp
  id<MTLBuffer> wf = nil, wi = nil, xlo = nil;
  RmBParams p{};
  double gpu = 0.0;
  explicit MetalBondedBackend(Context::Impl* impl) : m(impl) {}
  void setup(const BondedDeviceInput& in, const BondedLayout& L) override {
    int slot_no = 0;
    auto up = [&](const auto& v) { return m->slot_from("bonded_in" + std::to_string(slot_no++), v.data(), v.size() * sizeof(v[0])); };
    in_bufs = {up(in.list.x), up(in.type), up(in.tag), up(in.list.grid.atom_cell), up(in.list.grid.cell_start), up(in.list.grid.cell_items), up(in.sb_f),
               up(in.sb_i), up(in.tb_f), up(in.tb_i), up(in.thb_idx), up(in.thb_sets), up(in.fb_f), up(in.fb_has), up(in.hb_f), up(in.gp)};
    xlo = up(in.list.x_lo);
    wf = m->slot("bonded_wf", L.wf_size * sizeof(float));
    wi = m->slot("bonded_wi", L.wi_size * sizeof(std::int32_t));
    std::memset([wf contents], 0, L.wf_size * sizeof(float));
    std::memset([wi contents], 0, L.wi_size * sizeof(std::int32_t));
    p = RmBParams{};
    p.N = L.N; p.nlocal = L.nlocal; p.B = L.B; p.H = L.H; p.NB = static_cast<std::uint32_t>(L.NB); p.ntypes = in.ntypes;
    p.ncx = in.list.grid.ncell[0]; p.ncy = in.list.grid.ncell[1]; p.ncz = in.list.grid.ncell[2];
    p.o_atom = static_cast<std::uint32_t>(L.o_atom); p.o_iatom = static_cast<std::uint32_t>(L.o_iatom); p.o_tkl = static_cast<std::uint32_t>(L.o_tkl);
    p.o_tfl = static_cast<std::uint32_t>(L.o_tfl); p.o_hi = static_cast<std::uint32_t>(L.o_hi); p.o_hf = static_cast<std::uint32_t>(L.o_hf);
    p.enobonds = in.enobonds;
    p.bond_cut = in.bond_cut; p.bo_cut = in.bo_cut; p.thb_cut = in.thb_cut; p.thb_cutsq = in.thb_cutsq; p.hbond_cut = in.hbond_cut;
  }
  void run(std::span<const BondedStep> steps) override {
    std::vector<Dispatch> list;
    for (const BondedStep& st : steps) {
      Dispatch d = m->make(st.kernel, st.threads);
      d.buffers = in_bufs;
      d.buffers.push_back(wf);
      d.buffers.push_back(wi);
      d.buffers.push_back(xlo);
      set_params(d, p, 19);
      list.push_back(d);
    }
    m->run(list);
    gpu += m->gpu_seconds;
  }
  void read_float(std::size_t offset, std::size_t count, float* dst) override { std::memcpy(dst, static_cast<const float*>([wf contents]) + offset, count * sizeof(float)); }
  void read_int(std::size_t offset, std::size_t count, std::int32_t* dst) override { std::memcpy(dst, static_cast<const std::int32_t*>([wi contents]) + offset, count * sizeof(std::int32_t)); }
  double gpu_seconds() const override { return gpu; }
};
}  // namespace

Context::Context() : impl_(new Impl) {
  Impl& m = *impl_;
  m.device = MTLCreateSystemDefaultDevice();
  if (m.device == nil) throw MetalError("MTLCreateSystemDefaultDevice() returned nil: no Metal device");
  m.queue = [m.device newCommandQueue];
  if (m.queue == nil) throw MetalError("could not create a command queue");

  NSString* src = [NSString stringWithUTF8String:shader_source().c_str()];
  MTLCompileOptions* opts = [MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  opts.fastMathEnabled = NO;   // IEEE-conforming math: reproducibility before speed (NUMERICAL_POLICY)
#pragma clang diagnostic pop
  if (@available(macOS 15.0, *)) opts.mathMode = MTLMathModeSafe;

  NSError* err = nil;
  const auto t0 = std::chrono::steady_clock::now();
  m.library = [m.device newLibraryWithSource:src options:opts error:&err];
  const auto t1 = std::chrono::steady_clock::now();
  m.compile_seconds = std::chrono::duration<double>(t1 - t0).count();
  if (m.library == nil) fail("runtime compilation of the shader source failed", err);
  if (err != nil) m.log = to_std([err localizedDescription]);   // warnings of a successful compilation

  for (const std::string& name : kernel_names()) {
    id<MTLFunction> fn = [m.library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    if (fn == nil) throw MetalError("kernel '" + name + "' is missing from the compiled library");
    NSError* perr = nil;
    id<MTLComputePipelineState> pso = [m.device newComputePipelineStateWithFunction:fn error:&perr];
    if (pso == nil) fail("could not create the compute pipeline for '" + name + "'", perr);
    m.pipelines[name] = pso;
    KernelInfo ki;
    ki.name = name;
    ki.max_threads_per_threadgroup = static_cast<std::uint32_t>([pso maxTotalThreadsPerThreadgroup]);
    ki.thread_execution_width = static_cast<std::uint32_t>([pso threadExecutionWidth]);
    ki.static_threadgroup_memory = static_cast<std::uint32_t>([pso staticThreadgroupMemoryLength]);
    m.kernels.push_back(ki);
  }
}

Context::~Context() = default;

DeviceInfo Context::device_info() const {
  const Impl& m = *impl_;
  DeviceInfo i;
  i.name = to_std([m.device name]);
  i.os_version = to_std([[NSProcessInfo processInfo] operatingSystemVersionString]);
  i.unified_memory = [m.device hasUnifiedMemory] ? true : false;
  i.recommended_max_working_set = [m.device recommendedMaxWorkingSetSize];
  i.max_buffer_length = [m.device maxBufferLength];
  i.max_threads_per_threadgroup = static_cast<std::uint32_t>([m.device maxThreadsPerThreadgroup].width);
  auto add = [&](const char* n, MTLGPUFamily f) {
    if ([m.device supportsFamily:f]) i.families += std::string(i.families.empty() ? "" : " ") + n;
  };
  add("Apple7", MTLGPUFamilyApple7);
  add("Apple8", MTLGPUFamilyApple8);
  if (@available(macOS 14.0, *)) add("Apple9", MTLGPUFamilyApple9);
  add("Mac2", MTLGPUFamilyMac2);
  add("Metal3", MTLGPUFamilyMetal3);
  return i;
}

const std::vector<KernelInfo>& Context::kernels() const { return impl_->kernels; }
const std::string& Context::compile_log() const { return impl_->log; }
double Context::compile_seconds() const { return impl_->compile_seconds; }
double Context::last_gpu_seconds() const { return impl_->gpu_seconds; }

std::vector<float> Context::saxpy(float a, std::span<const float> x, std::span<const float> y) {
  if (x.size() != y.size()) throw MetalError("saxpy: x and y differ in size");
  const std::size_t n = x.size();
  if (n == 0) return {};
  if (n > 0xFFFFFFFFull) throw MetalError("saxpy: too many elements");
  Impl& m = *impl_;
  id<MTLBuffer> bx = m.buffer_from(x.data(), n * sizeof(float));
  id<MTLBuffer> by = m.buffer_from(y.data(), n * sizeof(float));
  RmSaxpyParams p{static_cast<std::uint32_t>(n), a};
  Dispatch d = m.make("rm_saxpy", n);
  d.buffers = {bx, by};
  set_params(d, p, 2);
  m.run({d});
  std::vector<float> out(n);
  std::memcpy(out.data(), [by contents], n * sizeof(float));
  return out;
}

MathProbe Context::math_probe() {
  Impl& m = *impl_;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float a = 1.0f + std::ldexp(1.0f, -13), c = -(1.0f + std::ldexp(1.0f, -12));
  const float in[4] = {nan, a, a, c};
  float out[2] = {-1.0f, -1.0f};
  id<MTLBuffer> bin = m.buffer_from(in, sizeof in);
  id<MTLBuffer> bout = m.buffer_from(out, sizeof out);
  Dispatch d = m.make("rm_math_probe", 1);
  d.buffers = {bin, bout};
  m.run({d});
  std::memcpy(out, [bout contents], sizeof out);
  MathProbe pr;
  pr.nan_detected = out[0] == 1.0f;
  pr.mul_add = out[1];
  return pr;
}

FarRowsF32 Context::far_rows(const DeviceListInput& in, std::uint32_t initial_cap, unsigned* launches) {
  Impl& m = *impl_;
  return build_far_rows_with_growth([&](std::uint32_t cap) { return m.launch_far_rows(in, cap); }, initial_cap, in.nall, 4, launches);
}

NonbondedDeviceOutput Context::nonbonded(const NonbondedDeviceInput& in) {
  Impl& m = *impl_;
  const std::size_t nall = in.list.nall, nlocal = in.list.nlocal, cap = in.rows.cap;
  NonbondedDeviceOutput out;
  out.grad.assign(3 * nall, 0.0f);
  out.row_e.assign(2 * nlocal, 0.0f);
  if (nall == 0 || nlocal == 0) return out;
  if (nlocal * cap * 3 > 0xFFFFFFFFull) throw MetalError("nonbonded: pair buffer index exceeds 32 bits");
  std::vector<float> pf_init(nlocal * cap * 3, 0.0f);   // entries that are not counted must read as zero
  id<MTLBuffer> bx = m.buffer_from(in.list.x.data(), in.list.x.size() * sizeof(float));
  id<MTLBuffer> btype = m.buffer_from(in.type.data(), in.type.size() * sizeof(std::int32_t));
  id<MTLBuffer> btag = m.buffer_from(in.tag.data(), in.tag.size() * sizeof(std::int32_t));
  id<MTLBuffer> bq = m.buffer_from(in.q.data(), in.q.size() * sizeof(float));
  id<MTLBuffer> btab = m.buffer_from(in.pair_table.data(), in.pair_table.size() * sizeof(float));
  id<MTLBuffer> bnbr = m.buffer_from(in.rows.nbr.data(), in.rows.nbr.size() * sizeof(std::int32_t));
  id<MTLBuffer> bcount = m.buffer_from(in.rows.count.data(), in.rows.count.size() * sizeof(std::uint32_t));
  id<MTLBuffer> bpf = m.buffer_from(pf_init.data(), pf_init.size() * sizeof(float));
  id<MTLBuffer> brow = m.buffer(out.row_e.size() * sizeof(float));
  id<MTLBuffer> bcs = m.buffer_from(in.column.start.data(), in.column.start.size() * sizeof(std::uint32_t));
  id<MTLBuffer> bci = m.buffer_from(in.column.items.data(), in.column.items.size() * sizeof(std::uint32_t));
  id<MTLBuffer> bgrad = m.buffer(out.grad.size() * sizeof(float));
  id<MTLBuffer> bxlo = m.buffer_from(in.list.x_lo.data(), in.list.x_lo.size() * sizeof(float));
  RmNbParams p{};
  p.nlocal = static_cast<std::uint32_t>(nlocal); p.cap = static_cast<std::uint32_t>(cap); p.ntypes = in.ntypes;
  p.vdw_type = in.vdw_type; p.lg = in.lg; p.p_vdW1 = in.p_vdW1; p.swa = in.swa; p.swb = in.swb;
  Dispatch d1 = m.make("rm_nb_pairs", nlocal);
  d1.buffers = {bx, btype, btag, bq, btab, bnbr, bcount, bpf, brow, bxlo};
  set_params(d1, p, 10);
  RmNbGatherParams g{static_cast<std::uint32_t>(nall), static_cast<std::uint32_t>(nlocal), static_cast<std::uint32_t>(cap)};
  Dispatch d2 = m.make("rm_nb_gather", nall);
  d2.buffers = {bpf, bcount, bcs, bci, bgrad};
  set_params(d2, g, 5);
  m.run({d1, d2});
  std::memcpy(out.row_e.data(), [brow contents], out.row_e.size() * sizeof(float));
  std::memcpy(out.grad.data(), [bgrad contents], out.grad.size() * sizeof(float));
  return out;
}

BondedDeviceOutput Context::bonded(const BondedDeviceInput& in) {
  MetalBondedBackend be(impl_.get());
  // start from the capacities the previous call needed (a system keeps its bond count), so steady state needs no regrow pass
  BondedDeviceOutput out = run_bonded_pipeline(be, in, impl_->last_bond_cap ? impl_->last_bond_cap : 8, impl_->last_hbond_cap ? impl_->last_hbond_cap : 8);
  impl_->last_bond_cap = out.bond_cap;
  impl_->last_hbond_cap = out.hbond_cap;
  impl_->gpu_seconds = be.gpu_seconds();
  return out;
}

void Context::qeq_setup(const QeqDeviceInput& in) {
  Impl& m = *impl_;
  const std::size_t nlocal = in.list.nlocal, cap = in.rows.cap;
  if (nlocal * cap > 0xFFFFFFFFull) throw MetalError("qeq: row buffer index exceeds 32 bits");
  m.qeq_nlocal = static_cast<std::uint32_t>(nlocal);
  m.qeq_cap = static_cast<std::uint32_t>(cap);
  if (nlocal == 0) return;
  m.qeq_params = RmQeqParams{};
  m.qeq_params.nlocal = m.qeq_nlocal; m.qeq_params.cap = m.qeq_cap; m.qeq_params.ntypes = in.ntypes; m.qeq_params.swa = in.swa; m.qeq_params.swb = in.swb;
  id<MTLBuffer> bx = m.slot_from("qeq_x", in.list.x.data(), in.list.x.size() * sizeof(float));
  id<MTLBuffer> bxlo = m.slot_from("qeq_xlo", in.list.x_lo.data(), in.list.x_lo.size() * sizeof(float));
  id<MTLBuffer> btype = m.slot_from("qeq_type", in.type.data(), in.type.size() * sizeof(std::int32_t));
  id<MTLBuffer> bshld = m.slot_from("qeq_shld", in.shld.data(), in.shld.size() * sizeof(float));
  id<MTLBuffer> bnbr = m.slot_from("qeq_nbr", in.rows.nbr.data(), in.rows.nbr.size() * sizeof(std::int32_t));
  id<MTLBuffer> bcount = m.slot_from("qeq_count", in.rows.count.data(), in.rows.count.size() * sizeof(std::uint32_t));
  id<MTLBuffer> bhv = m.slot("qeq_hv", nlocal * cap * sizeof(float));
  m.slot_from("qeq_owner", in.owner.data(), in.owner.size() * sizeof(std::int32_t));
  m.slot_from("qeq_colstart", in.column.start.data(), in.column.start.size() * sizeof(std::uint32_t));
  m.slot_from("qeq_colitems", in.column.items.data(), in.column.items.size() * sizeof(std::uint32_t));
  m.slot_from("qeq_eta", in.eta_atom.data(), in.eta_atom.size() * sizeof(float));
  m.slot("qeq_xv", nlocal * sizeof(float));
  m.slot("qeq_yv", nlocal * sizeof(float));
  Dispatch d = m.make("rm_qeq_h", nlocal);
  d.buffers = {bx, bxlo, btype, bshld, bnbr, bcount, bhv};
  set_params(d, m.qeq_params, 7);
  m.run({d});
}

void Context::qeq_matvec(const double* x, double* y) {
  Impl& m = *impl_;
  const std::size_t n = m.qeq_nlocal;
  if (n == 0) return;
  id<MTLBuffer> bxv = m.slots["qeq_xv"];
  id<MTLBuffer> byv = m.slots["qeq_yv"];
  float* xf = static_cast<float*>([bxv contents]);
  for (std::size_t i = 0; i < n; ++i) xf[i] = static_cast<float>(x[i]);
  Dispatch d = m.make("rm_qeq_mv", n);
  d.buffers = {m.slots["qeq_hv"], m.slots["qeq_nbr"], m.slots["qeq_count"], m.slots["qeq_owner"],
               m.slots["qeq_colstart"], m.slots["qeq_colitems"], bxv, m.slots["qeq_eta"], byv};
  set_params(d, m.qeq_params, 9);
  m.run({d});
  const float* yf = static_cast<const float*>([byv contents]);
  for (std::size_t i = 0; i < n; ++i) y[i] = static_cast<double>(yf[i]);
}

std::vector<float> Context::partial_sums(std::span<const float> v, std::uint32_t chunk) {
  if (chunk == 0) throw MetalError("partial_sums: chunk must be > 0");
  if (v.empty()) return {};
  if (v.size() > 0xFFFFFFFFull) throw MetalError("partial_sums: too many elements");
  Impl& m = *impl_;
  const std::size_t n = v.size(), nchunk = (n + chunk - 1) / chunk;
  id<MTLBuffer> bv = m.buffer_from(v.data(), n * sizeof(float));
  id<MTLBuffer> bpart = m.buffer(nchunk * sizeof(float));
  RmReduceParams p{static_cast<std::uint32_t>(n), chunk};
  Dispatch d = m.make("rm_partial_sums", nchunk);
  d.buffers = {bv, bpart};
  set_params(d, p, 2);
  m.run({d});
  std::vector<float> part(nchunk);
  std::memcpy(part.data(), [bpart contents], nchunk * sizeof(float));
  return part;
}

float Context::sum(std::span<const float> v, std::uint32_t chunk) {
  if (chunk == 0) throw MetalError("sum: chunk must be > 0");
  if (v.empty()) return 0.0f;
  if (v.size() > 0xFFFFFFFFull) throw MetalError("sum: too many elements");
  Impl& m = *impl_;
  const std::size_t n = v.size(), nchunk = (n + chunk - 1) / chunk;
  id<MTLBuffer> bv = m.buffer_from(v.data(), n * sizeof(float));
  id<MTLBuffer> bpart = m.buffer(nchunk * sizeof(float));
  id<MTLBuffer> bout = m.buffer(sizeof(float));
  RmReduceParams p{static_cast<std::uint32_t>(n), chunk};
  Dispatch d1 = m.make("rm_partial_sums", nchunk);
  d1.buffers = {bv, bpart};
  set_params(d1, p, 2);
  Dispatch d2 = m.make("rm_sum_partials", 1);
  d2.buffers = {bpart, bout};
  set_params(d2, p, 2);
  m.run({d1, d2});
  float out = 0.0f;
  std::memcpy(&out, [bout contents], sizeof out);
  return out;
}

}  // namespace reaxmetal::mtl
