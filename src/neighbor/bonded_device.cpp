// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/bonded_device.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "reaxmetal_m3_types.h"
#include "reaxmetal_m4_types.h"
#include "reaxmetal_m6_types.h"

namespace reaxmetal {

BondedLayout bonded_layout(std::uint32_t N, std::uint32_t nlocal, std::uint32_t B, std::uint32_t H) {
  BondedLayout L;
  L.N = N; L.nlocal = nlocal; L.B = B; L.H = H;
  L.NB = static_cast<std::size_t>(N) * B;
  L.o_atom = RM_SF_COUNT * L.NB;
  L.o_tkl = L.o_atom + static_cast<std::size_t>(RM_AF_COUNT) * N;
  L.o_tfl = L.o_tkl + static_cast<std::size_t>(nlocal) * B * B;
  L.o_hf = L.o_tfl + static_cast<std::size_t>(nlocal) * B * B * 3;
  L.wf_size = L.o_hf + static_cast<std::size_t>(nlocal) * H * 3;
  L.o_iatom = RM_SI_COUNT * L.NB;
  L.o_hi = L.o_iatom + static_cast<std::size_t>(RM_AI_COUNT) * N;
  L.wi_size = L.o_hi + static_cast<std::size_t>(nlocal) * H;
  return L;
}

BondedDeviceInput make_bonded_device_input(const ForceField& ff, const ControlParams& ctl, const AtomSet& atoms, const Box& box, const BondedOptions& opt) {
  BondedDeviceInput in;
  NeighborCutoffs cut;
  cut.nonb = ff.file_control().nonb_cut;
  cut.bond = ctl.bond_cut;
  cut.hbond = ctl.hbond_cut;
  in.list = make_device_list_input(atoms, box, cut);
  const std::size_t N = atoms.nall();
  in.type.assign(atoms.type.begin(), atoms.type.end());
  in.tag.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    const std::int64_t t = atoms.tag[i];
    if (t > 0x7FFFFFFF || t < -0x7FFFFFFF) throw SystemError("make_bonded_device_input: atom tag does not fit 32 bits");
    in.tag[i] = static_cast<std::int32_t>(t);
  }
  const int nt = ff.num_types();
  if (static_cast<double>(nt) * nt * nt * nt > 4.0e6) throw SystemError("make_bonded_device_input: too many atom types for the dense four-body table");
  in.ntypes = static_cast<std::uint32_t>(nt);
  const std::size_t unt = static_cast<std::size_t>(nt);
  for (int t = 0; t < nt; ++t) {
    const SingleBody& s = ff.single(t);
    for (double v : {s.valency, s.valency_e, s.valency_boc, s.valency_val, s.nlp_opt, s.r_s, s.r_pi, s.r_pi_pi, s.p_lp2, s.p_ovun2, s.p_ovun5, s.p_val3, s.p_val5})
      in.sb_f.push_back(static_cast<float>(v));
    in.sb_i.push_back(s.flags.heavy_atom_terms ? 1 : 0);
    in.sb_i.push_back(s.flags.c2_species ? 1 : 0);
    in.sb_i.push_back(s.p_hbond);
  }
  for (int a = 0; a < nt; ++a)
    for (int b = 0; b < nt; ++b) {
      const TwoBody& t = ff.pair(a, b);
      for (double v : {t.p_bo1, t.p_bo2, t.p_bo3, t.p_bo4, t.p_bo5, t.p_bo6, t.r_s, t.r_p, t.r_pp, t.p_boc3, t.p_boc4, t.p_boc5, t.ovc, t.v13cor,
                       t.p_be1, t.p_be2, t.De_s, t.De_p, t.De_pp, t.p_ovun1})
        in.tb_f.push_back(static_cast<float>(v));
      in.tb_i.push_back(t.triple_bond_stabilisation ? 1 : 0);
    }
  in.thb_idx.assign(2 * unt * unt * unt, 0);
  for (int i = 0; i < nt; ++i)
    for (int j = 0; j < nt; ++j)
      for (int k = 0; k < nt; ++k) {
        const auto sets = ff.three_body(i, j, k);
        const std::size_t idx = (static_cast<std::size_t>(i) * unt + static_cast<std::size_t>(j)) * unt + static_cast<std::size_t>(k);
        in.thb_idx[2 * idx] = static_cast<std::int32_t>(in.thb_sets.size() / RM_B_THB_F);
        in.thb_idx[2 * idx + 1] = static_cast<std::int32_t>(sets.size());
        for (const ThreeBodySet& s : sets)
          for (double v : {s.theta_00, s.p_val1, s.p_val2, s.p_val4, s.p_val7, s.p_pen1, s.p_coa1}) in.thb_sets.push_back(static_cast<float>(v));
      }
  in.fb_f.assign(unt * unt * unt * unt * RM_B_FB_F, 0.0f);
  in.fb_has.assign(unt * unt * unt * unt, 0);
  in.hb_f.assign(unt * unt * unt * RM_B_HB_F, 0.0f);
  for (int i = 0; i < nt; ++i)
    for (int j = 0; j < nt; ++j)
      for (int k = 0; k < nt; ++k) {
        const std::size_t h3 = (static_cast<std::size_t>(i) * unt + static_cast<std::size_t>(j)) * unt + static_cast<std::size_t>(k);
        if (const HBondParams* hb = ff.hbond(i, j, k)) {
          const double v[4] = {hb->r0_hb, hb->p_hb1, hb->p_hb2, hb->p_hb3};
          for (std::size_t q = 0; q < 4; ++q) in.hb_f[h3 * RM_B_HB_F + q] = static_cast<float>(v[q]);
        } else {
          in.hb_f[h3 * RM_B_HB_F] = -1.0f;
        }
        for (int l = 0; l < nt; ++l) {
          const std::size_t f4 = h3 * unt + static_cast<std::size_t>(l);
          if (const FourBody* fb = ff.four_body(i, j, k, l)) {
            const double v[5] = {fb->V1, fb->V2, fb->V3, fb->p_tor1, fb->p_cot1};
            for (std::size_t q = 0; q < 5; ++q) in.fb_f[f4 * RM_B_FB_F + q] = static_cast<float>(v[q]);
            in.fb_has[f4] = 1;
          }
        }
      }
  in.gp.assign(40, 0.0f);
  const auto& g = ff.global().l;
  for (std::size_t i = 0; i < g.size() && i < 40; ++i) in.gp[i] = static_cast<float>(g[i]);
  in.enobonds = opt.enobonds ? 1u : 0u;
  in.bond_cut = static_cast<float>(ctl.bond_cut);
  in.bo_cut = static_cast<float>(ff.file_control().bo_cut);
  in.thb_cut = static_cast<float>(ctl.thb_cut);
  in.thb_cutsq = static_cast<float>(ctl.thb_cutsq);
  in.hbond_cut = static_cast<float>(ctl.hbond_cut);
  return in;
}

namespace {
std::uint32_t round_up8(std::uint32_t v) { return (v + 7u) / 8u * 8u; }
}

BondedDeviceOutput run_bonded_pipeline(BondedBackend& backend, const BondedDeviceInput& in, std::uint32_t B, std::uint32_t H) {
  const std::uint32_t N = in.list.nall, nl = in.list.nlocal;
  BondedDeviceOutput out;
  if (N == 0) return out;
  B = std::max<std::uint32_t>(B, 4);
  H = std::max<std::uint32_t>(H, 4);
  for (unsigned attempt = 1; attempt <= 4; ++attempt) {
    out.attempts = attempt;
    const BondedLayout L = bonded_layout(N, nl, B, H);
    if (L.wf_size > 0xFFFFFFF0ull || L.wi_size > 0xFFFFFFF0ull) throw SystemError("bonded pipeline: work array exceeds 32-bit indexing (too many atoms or too large bond capacity)");
    backend.setup(in, L);
    const std::array<BondedStep, 2> phase1{{{"rm_b_build", N}, {"rm_h_build", N}}};
    backend.run(phase1);
    std::vector<std::int32_t> nb(N), hnb(N);
    backend.read_int(L.o_iatom + static_cast<std::size_t>(RM_AI_NB) * N, N, nb.data());
    backend.read_int(L.o_iatom + static_cast<std::size_t>(RM_AI_HNB) * N, N, hnb.data());
    std::uint32_t maxb = 0, maxh = 0;
    for (std::uint32_t i = 0; i < N; ++i) { maxb = std::max<std::uint32_t>(maxb, static_cast<std::uint32_t>(nb[i])); maxh = std::max<std::uint32_t>(maxh, static_cast<std::uint32_t>(hnb[i])); }
    if (maxb > B || maxh > H) {
      B = std::max(B, round_up8(maxb));
      H = std::max(H, round_up8(maxh));
      continue;
    }
    const std::array<BondedStep, 10> phase2{{{"rm_b_prime", N}, {"rm_b_correct", N}, {"rm_b_atom", nl}, {"rm_b_valence", nl}, {"rm_b_torsion", nl},
                                              {"rm_b_hbond", nl}, {"rm_b_hbgather", N}, {"rm_b_cdgather", N}, {"rm_b_dbond", N}, {"rm_b_force", N}}};
    backend.run(phase2);
    out.bond_cap = B;
    out.hbond_cap = H;
    out.grad.resize(3 * static_cast<std::size_t>(N));
    std::vector<float> tmp(N);
    for (std::size_t c = 0; c < 3; ++c) {
      backend.read_float(L.o_atom + (static_cast<std::size_t>(RM_AF_GX) + c) * N, N, tmp.data());
      for (std::uint32_t i = 0; i < N; ++i) out.grad[3 * static_cast<std::size_t>(i) + c] = tmp[i];
    }
    const std::array<std::pair<RmAtomF, EnergyTerm>, 10> eterm{{{RM_AF_EBOND, EnergyTerm::Bond}, {RM_AF_ELP, EnergyTerm::LonePair}, {RM_AF_EOV, EnergyTerm::Over},
                                                                  {RM_AF_EUN, EnergyTerm::Under}, {RM_AF_EANG, EnergyTerm::Valence}, {RM_AF_EPEN, EnergyTerm::Penalty},
                                                                  {RM_AF_ECOA, EnergyTerm::Coalition}, {RM_AF_ETOR, EnergyTerm::Torsion}, {RM_AF_ECON, EnergyTerm::Conjugation},
                                                                  {RM_AF_EHB, EnergyTerm::HBond}}};
    for (const auto& [f, t] : eterm) {
      backend.read_float(L.o_atom + static_cast<std::size_t>(f) * N, N, tmp.data());
      double s = 0.0;
      for (std::uint32_t i = 0; i < nl; ++i) s += static_cast<double>(tmp[i]);
      out.e[static_cast<std::size_t>(t)] = s;
    }
    return out;
  }
  throw SystemError("bonded pipeline: capacities still too small after 4 attempts");
}

BondedResult finish_bonded(const BondedDeviceOutput& out, std::size_t nall) {
  BondedResult r;
  for (std::size_t t = 0; t < kEnergyTermCount; ++t) r.e.e[t] = out.e[t];
  r.grad.assign(out.grad.begin(), out.grad.end());
  (void)nall;
  return r;
}

}  // namespace reaxmetal
