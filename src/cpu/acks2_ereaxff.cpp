// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/acks2_ereaxff.hpp"

#include <algorithm>
#include <cmath>

#include "reaxmetal/electron.hpp"
#include "reaxmetal/terms_host.hpp"

namespace reaxmetal {

namespace {
// Dense LU with partial pivoting, row-major n x n. ponytail: O(n^3); fine up to ~1500 unknowns (ms for the 11-atom case); a sparse/iterative
// solver is the upgrade if solved systems grow beyond that.
void lu_solve(std::vector<double>& a, std::vector<double>& b, std::size_t n) {
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t p = k;
    double big = std::fabs(a[k * n + k]);
    for (std::size_t i = k + 1; i < n; ++i)
      if (std::fabs(a[i * n + k]) > big) { big = std::fabs(a[i * n + k]); p = i; }
    if (!(big > 1.0e-14)) throw SystemError("ACKS2 solve: singular matrix (is every charge group non-empty and every kinetic-potential block connected?)");
    if (p != k) {
      for (std::size_t j = 0; j < n; ++j) std::swap(a[k * n + j], a[p * n + j]);
      std::swap(b[k], b[p]);
    }
    const double inv = 1.0 / a[k * n + k];
    for (std::size_t i = k + 1; i < n; ++i) {
      const double f = a[i * n + k] * inv;
      if (f == 0.0) continue;
      for (std::size_t j = k; j < n; ++j) a[i * n + j] -= f * a[k * n + j];
      b[i] -= f * b[k];
    }
  }
  for (std::size_t kk = n; kk-- > 0;) {
    double v = b[kk];
    for (std::size_t j = kk + 1; j < n; ++j) v -= a[kk * n + j] * b[j];
    b[kk] = v / a[kk * n + kk];
  }
}
}  // namespace

Acks2Solution solve_acks2_ereaxff(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far, const std::vector<double>& q_in,
                                  const std::vector<int>& group, const std::vector<double>& group_q, const Acks2Options& opt) {
  const std::size_t n = atoms.nlocal;
  if (q_in.size() != n || group.size() != n) throw SystemError("solve_acks2_ereaxff: charge and group arrays need one entry per owned atom");
  Acks2Solution out;
  out.q = q_in;
  out.s.assign(n, 0.0);
  out.ref.assign(n, 0.0);
  // solved atoms and their local index
  std::vector<int> loc(n, -1);
  std::size_t m = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (atoms.type[i] >= 0 && ff.single(atoms.type[i]).flags.charge_solve) loc[i] = static_cast<int>(m++);
  out.nsolved = m;
  if (m == 0) return out;
  const std::size_t G = group_q.size();
  std::vector<std::size_t> ng(G, 0);
  for (std::size_t i = 0; i < n; ++i)
    if (loc[i] >= 0) {
      if (group[i] < 0 || static_cast<std::size_t>(group[i]) >= G) throw SystemError("solve_acks2_ereaxff: solved atom without a charge group");
      ++ng[static_cast<std::size_t>(group[i])];
    }
  const std::size_t N = 2 * m + 1 + G;
  out.neq = N;
  std::vector<double> A(N * N, 0.0), b(N, 0.0);
  const double evk = opt.legacy_constants ? kLegacyEvKcal : RM_KCAL_TO_EV;
  const double dielec = opt.legacy_constants ? static_cast<double>(14.40f) : electron_terms::kCoulombKcal / evk;
  const double expo = opt.legacy_constants ? static_cast<double>(1.0f / 3.0f) : 1.0 / 3.0;
  const double selfw = opt.legacy_constants ? 1.0 : 2.0;   // diagonal weight per half-listed self-image pair (legacy: 0.5 x both signs)
  const bool soft_fixed = opt.soft_fixed >= 0 ? opt.soft_fixed == 1 : !opt.legacy_constants;
  const double softness = ff.global().l.size() > 34 ? ff.global().l[34] : 0.0;
  const terms::TaperCoeffs tap = terms::taper_coeffs(ff.file_control().nonb_low, ff.file_control().nonb_cut);
  auto at = [&](std::size_t r, std::size_t c) -> double& { return A[r * N + c]; };

  for (std::size_t i = 0; i < n; ++i) {
    if (loc[i] < 0) continue;
    const std::size_t k = static_cast<std::size_t>(loc[i]);
    const SingleBody& s = ff.single(atoms.type[i]);
    const std::size_t g = static_cast<std::size_t>(group[i]);
    at(k, k) += s.eta;                              // single().eta is 2 x the file value = the diagonal 2*eta of the legacy matrix
    b[k] = -s.chi;
    at(k, m + k) = 1.0; at(m + k, k) = 1.0;         // coupling with the kinetic potential
    at(m + k, 2 * m) = 1.0; at(2 * m, m + k) = 1.0; // sum s = 0
    at(k, 2 * m + 1 + g) = 1.0; at(2 * m + 1 + g, k) = 1.0;
    b[m + k] = group_q[g] / static_cast<double>(ng[g]);
    out.ref[i] = b[m + k];
  }
  for (std::size_t g = 0; g < G; ++g) b[2 * m + 1 + g] = group_q[g];

  for (std::size_t i = 0; i < n; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    for (std::size_t pj = far.row_start[i]; pj < far.row_start[i + 1]; ++pj) {
      const std::size_t j = static_cast<std::size_t>(far.nbr[pj]);
      const int tj = atoms.type[j];
      if (tj < 0) continue;
      const double r = far.dist[pj];
      const double* dv = &far.dvec[3 * pj];
      const PairClass pc = classify_nonbonded_entry(i, j, n, atoms.tag[i], atoms.tag[j], dv);
      if (pc == PairClass::None) continue;
      const std::size_t o = static_cast<std::size_t>(atoms.owner[j]);
      const int ki = loc[i], ko = loc[o];
      if (ki < 0 && ko < 0) continue;                // fixed-fixed: no entry (El-El is never part of the system)
      if (r < cut.nonb) {
        double Tap, dTap;
        terms::taper_horner<double>(tap.c, r, Tap, dTap);
        const double w = Tap * dielec / std::pow(r * r * r + ff.pair(ti, tj).gamma, expo);
        if (pc == PairClass::SelfImage) {
          at(static_cast<std::size_t>(ki), static_cast<std::size_t>(ki)) += selfw * w;
        } else if (ki >= 0 && ko >= 0) {
          at(static_cast<std::size_t>(ki), static_cast<std::size_t>(ko)) += w;
          at(static_cast<std::size_t>(ko), static_cast<std::size_t>(ki)) += w;
        } else if (ki >= 0) {
          b[static_cast<std::size_t>(ki)] -= w * q_in[o];       // field of a fixed charge, folded into the right-hand side
        } else {
          b[static_cast<std::size_t>(ko)] -= w * q_in[i];
        }
      }
      // kinetic-energy (bond softness) block. Legacy: solved pairs only. Consistent constants: a pair with ONE fixed particle (s = 0) also adds -bs to the diagonal of
      // the solved atom, because the softness ENERGY contains that pair; without it the energy is not stationary in s and the forces are not its gradient.
      if (pc != PairClass::SelfImage && !(ki >= 0 && ko >= 0 && ki == ko) && (ki >= 0 && ko >= 0 || soft_fixed)) {
        const double xcut = 0.5 * (ff.single(ti).bcut_acks2 + ff.single(tj).bcut_acks2);
        if (r <= xcut) {
          const double d = r / xcut;
          const double bs = softness * std::pow(d, 3.0) * std::pow(1.0 - d, 6.0);
          if (bs > 0.0) {
            if (ki >= 0 && ko >= 0) {
              const std::size_t a1 = m + static_cast<std::size_t>(ki), a2 = m + static_cast<std::size_t>(ko);
              at(a1, a2) += bs; at(a2, a1) += bs; at(a1, a1) -= bs; at(a2, a2) -= bs;
            } else {
              const std::size_t a1 = m + static_cast<std::size_t>(ki >= 0 ? ki : ko);
              at(a1, a1) -= bs;
            }
          }
        }
      }
    }
  }
  std::vector<double> Acopy = A, x = b;
  lu_solve(A, x, N);
  double res = 0.0;
  for (std::size_t r = 0; r < N; ++r) {
    double v = -b[r];
    for (std::size_t c = 0; c < N; ++c) v += Acopy[r * N + c] * x[c];
    res = std::max(res, std::fabs(v));
  }
  out.residual = res;
  for (std::size_t i = 0; i < n; ++i)
    if (loc[i] >= 0) { out.q[i] = x[static_cast<std::size_t>(loc[i])]; out.s[i] = x[m + static_cast<std::size_t>(loc[i])]; }
  return out;
}

double acks2_self_energy(const ForceField& ff, const AtomSet& atoms, const std::vector<double>& q, double evkcmol) {
  double e = 0.0;
  for (std::size_t i = 0; i < atoms.nlocal; ++i) {
    if (atoms.type[i] < 0) continue;
    const SingleBody& s = ff.single(atoms.type[i]);
    if (!s.flags.charge_solve) continue;
    e += evkcmol * (s.chi + 0.5 * s.eta * q[i]) * q[i];
  }
  return e;
}

}  // namespace reaxmetal
