// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#include "reaxmetal/acks2_ereaxff.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

#include "reaxmetal/electron.hpp"
#include "reaxmetal/terms_host.hpp"

namespace reaxmetal {

namespace {

struct Csr {
  std::vector<std::size_t> start;   // m + 1
  std::vector<int> col;
  std::vector<double> val;
};

// Dense LU with partial pivoting, row-major n x n, factors in place.
void lu_factor(std::vector<double>& a, std::vector<std::size_t>& piv, std::size_t n) {
  piv.assign(n, 0);
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t p = k;
    double big = std::fabs(a[k * n + k]);
    for (std::size_t i = k + 1; i < n; ++i)
      if (std::fabs(a[i * n + k]) > big) { big = std::fabs(a[i * n + k]); p = i; }
    if (!(big > 1.0e-14)) throw SystemError("ACKS2 solve: singular matrix (is every charge group non-empty and every kinetic-potential block connected?)");
    piv[k] = p;
    if (p != k) for (std::size_t j = 0; j < n; ++j) std::swap(a[k * n + j], a[p * n + j]);
    const double inv = 1.0 / a[k * n + k];
    for (std::size_t i = k + 1; i < n; ++i) {
      const double f = a[i * n + k] * inv;
      a[i * n + k] = f;
      if (f == 0.0) continue;
      for (std::size_t j = k + 1; j < n; ++j) a[i * n + j] -= f * a[k * n + j];
    }
  }
}
void lu_solve(const std::vector<double>& a, const std::vector<std::size_t>& piv, std::vector<double>& b, std::size_t n) {
  for (std::size_t k = 0; k < n; ++k) if (piv[k] != k) std::swap(b[k], b[piv[k]]);   // rows were swapped whole (LAPACK convention): all swaps first
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t i = k + 1; i < n; ++i) b[i] -= a[i * n + k] * b[k];
  for (std::size_t kk = n; kk-- > 0;) {
    double v = b[kk];
    for (std::size_t j = kk + 1; j < n; ++j) v -= a[kk * n + j] * b[j];
    b[kk] = v / a[kk * n + kk];
  }
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

// Preconditioned MINRES (Paige-Saunders) for a symmetric indefinite A and an SPD diagonal preconditioner (given as the inverse diagonal minv).
// Stops when the preconditioned residual-norm estimate <= tol * ||b||_{M^-1}. Returns the iteration count; x holds the start and the result.
int minres(const std::function<void(const std::vector<double>&, std::vector<double>&)>& A, const std::vector<double>& minv, const std::vector<double>& b,
           std::vector<double>& x, double tol, int maxit, bool& converged) {
  const std::size_t n = b.size();
  std::vector<double> r1(n), r2(n), y(n), v(n), w(n, 0.0), w1(n, 0.0), w2(n, 0.0), Av(n);
  A(x, Av);
  for (std::size_t i = 0; i < n; ++i) r1[i] = b[i] - Av[i];
  double nb = 0.0;
  for (std::size_t i = 0; i < n; ++i) nb += b[i] * b[i] * minv[i];
  nb = std::sqrt(nb);
  for (std::size_t i = 0; i < n; ++i) y[i] = minv[i] * r1[i];
  double beta1 = std::sqrt(std::max(0.0, dot(r1, y)));
  converged = true;
  if (beta1 <= tol * nb || beta1 == 0.0) return 0;
  converged = false;
  double oldb = 0.0, beta = beta1, dbar = 0.0, epsln = 0.0, phibar = beta1, cs = -1.0, sn = 0.0;
  r2 = r1;
  for (int itn = 1; itn <= maxit; ++itn) {
    const double sc = 1.0 / beta;
    for (std::size_t i = 0; i < n; ++i) v[i] = sc * y[i];
    A(v, y);
    if (itn >= 2) for (std::size_t i = 0; i < n; ++i) y[i] -= (beta / oldb) * r1[i];
    const double alfa = dot(v, y);
    for (std::size_t i = 0; i < n; ++i) y[i] -= (alfa / beta) * r2[i];
    r1 = r2; r2 = y;
    for (std::size_t i = 0; i < n; ++i) y[i] = minv[i] * r2[i];
    oldb = beta;
    beta = std::sqrt(std::max(0.0, dot(r2, y)));
    const double oldeps = epsln, delta = cs * dbar + sn * alfa, gbar = sn * dbar - cs * alfa;
    epsln = sn * beta; dbar = -cs * beta;
    const double gamma = std::max(std::sqrt(gbar * gbar + beta * beta), 1.0e-300);
    cs = gbar / gamma; sn = beta / gamma;
    const double phi = cs * phibar;
    phibar = sn * phibar;
    const double denom = 1.0 / gamma;
    w1 = w2; w2 = w;
    for (std::size_t i = 0; i < n; ++i) { w[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) * denom; x[i] += phi * w[i]; }
    if (std::fabs(phibar) <= tol * nb) { converged = true; return itn; }
    if (beta == 0.0) { converged = true; return itn; }
  }
  return maxit;
}

}  // namespace

struct Acks2Engine::Impl {
  // ---- cache key
  bool valid = false;
  std::vector<double> key_x, key_box;
  std::vector<int> key_type, key_group;
  std::vector<double> key_gq;
  bool key_legacy = false;
  const ForceField* key_ff = nullptr;
  // ---- assembled solved-solved system
  std::size_t m = 0, G = 0, N = 0;
  std::vector<int> loc;                  // owned index -> solved index or -1
  std::vector<int> gidx;                 // solved index -> group
  std::vector<double> dq, xd, ref, chi;  // q-row diagonal (2 eta + self-image), s-row diagonal (-sum bs over solved pairs), reference charge, chi
  Csr H, X;                              // off-diagonal blocks (both triangles)
  // ---- dense factorisation cache
  std::vector<double> lu; std::vector<std::size_t> piv; std::vector<double> lu_dX; bool lu_valid = false;
  // ---- warm start
  std::vector<double> xprev; std::size_t prev_m = 0, prev_G = 0;
};

Acks2Engine::Acks2Engine() : p_(new Impl) {}
Acks2Engine::~Acks2Engine() = default;
void Acks2Engine::reset() { p_.reset(new Impl); }

Acks2Solution Acks2Engine::solve(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far, const std::vector<double>& q_in,
                                 const std::vector<int>& group, const std::vector<double>& group_q, const std::vector<double>& box_key, const Acks2Options& opt) {
  Impl& S = *p_;
  const std::size_t n = atoms.nlocal;
  if (q_in.size() != n || group.size() != n) throw SystemError("ACKS2: charge and group arrays need one entry per owned atom");
  Acks2Solution out;
  out.q = q_in; out.s.assign(n, 0.0); out.ref.assign(n, 0.0);

  // ---- solved set and cache key
  std::vector<int> loc(n, -1);
  std::size_t m = 0;
  std::vector<double> kx; std::vector<int> kt, kg;
  for (std::size_t i = 0; i < n; ++i)
    if (atoms.type[i] >= 0 && ff.single(atoms.type[i]).flags.charge_solve) {
      loc[i] = static_cast<int>(m++);
      kx.push_back(atoms.x[3 * i]); kx.push_back(atoms.x[3 * i + 1]); kx.push_back(atoms.x[3 * i + 2]);
      kt.push_back(atoms.type[i]); kg.push_back(group[i]);
    }
  out.nsolved = m;
  if (m == 0) return out;
  const std::size_t G = group_q.size();
  const std::size_t N = 2 * m + 1 + G;
  out.neq = N;
  const bool hit = opt.cache && S.valid && S.key_ff == &ff && S.key_legacy == opt.legacy_constants && S.m == m && S.G == G && S.key_x == kx && S.key_type == kt &&
                   S.key_group == kg && S.key_gq == group_q && S.key_box == box_key;
  out.cache_hit = hit;

  const double dielec = opt.legacy_constants ? static_cast<double>(14.40f) : electron_terms::kCoulombKcal / (opt.legacy_constants ? kLegacyEvKcal : RM_KCAL_TO_EV);
  const double expo = opt.legacy_constants ? static_cast<double>(1.0f / 3.0f) : 1.0 / 3.0;
  const double selfw = opt.legacy_constants ? 1.0 : 2.0;   // diagonal weight per half-listed self-image pair (legacy: 0.5 x both signs)
  const double softness = ff.global().l.size() > 34 ? ff.global().l[34] : 0.0;
  const bool soft_fixed = opt.soft_fixed >= 0 ? opt.soft_fixed == 1 : !opt.legacy_constants;
  const terms::TaperCoeffs tap = terms::taper_coeffs(ff.file_control().nonb_low, ff.file_control().nonb_cut);
  auto pair_w = [&](double r, int ti, int tj) {
    double Tap, dTap;
    terms::taper_horner<double>(tap.c, r, Tap, dTap);
    return Tap * dielec / std::pow(r * r * r + ff.pair(ti, tj).gamma, expo);
  };
  auto bsoft = [&](double r, int ti, int tj) {
    const double xcut = 0.5 * (ff.single(ti).bcut_acks2 + ff.single(tj).bcut_acks2);
    if (!(r <= xcut)) return 0.0;
    const double d = r / xcut;
    const double bs = softness * std::pow(d, 3.0) * std::pow(1.0 - d, 6.0);
    return bs > 0.0 ? bs : 0.0;
  };

  // ---- assemble the solved-solved part (cache miss)
  if (!hit) {
    S.valid = false; S.lu_valid = false;
    S.m = m; S.G = G; S.N = N; S.loc = loc;
    S.gidx.assign(m, 0); S.dq.assign(m, 0.0); S.xd.assign(m, 0.0); S.ref.assign(m, 0.0); S.chi.assign(m, 0.0);
    std::vector<std::size_t> ng(G, 0);
    for (std::size_t i = 0; i < n; ++i)
      if (loc[i] >= 0) {
        if (group[i] < 0 || static_cast<std::size_t>(group[i]) >= G) throw SystemError("ACKS2: solved atom without a charge group");
        ++ng[static_cast<std::size_t>(group[i])];
      }
    for (std::size_t i = 0; i < n; ++i)
      if (loc[i] >= 0) {
        const std::size_t k = static_cast<std::size_t>(loc[i]);
        const SingleBody& sb = ff.single(atoms.type[i]);
        S.gidx[k] = group[i];
        S.dq[k] = sb.eta;                       // single().eta is 2 x the file value = the diagonal 2*eta of the legacy matrix
        S.chi[k] = sb.chi;
        S.ref[k] = group_q[static_cast<std::size_t>(group[i])] / static_cast<double>(ng[static_cast<std::size_t>(group[i])]);
      }
    std::vector<std::vector<std::pair<int, double>>> hrow(m), xrow(m);
    for (std::size_t i = 0; i < n; ++i) {
      const int ti = atoms.type[i];
      if (ti < 0 || loc[i] < 0) continue;
      const std::size_t ki = static_cast<std::size_t>(loc[i]);
      for (std::size_t pj = far.row_start[i]; pj < far.row_start[i + 1]; ++pj) {
        const std::size_t j = static_cast<std::size_t>(far.nbr[pj]);
        const int tj = atoms.type[j];
        if (tj < 0) continue;
        const double r = far.dist[pj];
        const double* dv = &far.dvec[3 * pj];
        const PairClass pc = classify_nonbonded_entry(i, j, n, atoms.tag[i], atoms.tag[j], dv);
        if (pc == PairClass::None) continue;
        const std::size_t o = static_cast<std::size_t>(atoms.owner[j]);
        if (loc[o] < 0) continue;               // solved-fixed pairs: per call
        const std::size_t ko = static_cast<std::size_t>(loc[o]);
        if (r < cut.nonb) {
          const double w = pair_w(r, ti, tj);
          if (pc == PairClass::SelfImage) S.dq[ki] += selfw * w;
          else { hrow[ki].emplace_back(static_cast<int>(ko), w); hrow[ko].emplace_back(static_cast<int>(ki), w); }
        }
        if (pc != PairClass::SelfImage && ki != ko) {
          const double bs = bsoft(r, ti, tj);
          if (bs > 0.0) {
            xrow[ki].emplace_back(static_cast<int>(ko), bs); xrow[ko].emplace_back(static_cast<int>(ki), bs);
            S.xd[ki] -= bs; S.xd[ko] -= bs;
          }
        }
      }
    }
    auto to_csr = [&](std::vector<std::vector<std::pair<int, double>>>& rows, Csr& c) {
      c.start.assign(m + 1, 0); c.col.clear(); c.val.clear();
      for (std::size_t k = 0; k < m; ++k) {
        auto& r = rows[k];
        std::sort(r.begin(), r.end());
        for (std::size_t a = 0; a < r.size();) {
          std::size_t b = a; double v = 0.0;
          while (b < r.size() && r[b].first == r[a].first) v += r[b++].second;
          c.col.push_back(r[a].first); c.val.push_back(v); a = b;
        }
        c.start[k + 1] = c.col.size();
      }
    };
    to_csr(hrow, S.H); to_csr(xrow, S.X);
    S.key_x = kx; S.key_type = kt; S.key_group = kg; S.key_gq = group_q; S.key_box = box_key; S.key_legacy = opt.legacy_constants; S.key_ff = &ff;
    S.valid = opt.cache;
  }
  for (std::size_t i = 0; i < n; ++i) if (loc[i] >= 0) out.ref[i] = S.ref[static_cast<std::size_t>(loc[i])];

  // ---- per call: field of the fixed particles (right-hand side) and, in consistent mode, their bond-softness share of the diagonal of X
  std::vector<double> b(N, 0.0), dX(m, 0.0);
  for (std::size_t k = 0; k < m; ++k) { b[k] = -S.chi[k]; b[m + k] = S.ref[k]; }
  for (std::size_t g = 0; g < G; ++g) b[2 * m + 1 + g] = group_q[g];
  for (std::size_t i = 0; i < n; ++i) {
    const int ti = atoms.type[i];
    if (ti < 0) continue;
    for (std::size_t pj = far.row_start[i]; pj < far.row_start[i + 1]; ++pj) {
      const std::size_t j = static_cast<std::size_t>(far.nbr[pj]);
      const int tj = atoms.type[j];
      if (tj < 0) continue;
      const std::size_t o = static_cast<std::size_t>(atoms.owner[j]);
      const int ki = loc[i], ko = loc[o];
      if ((ki >= 0) == (ko >= 0)) continue;     // exactly one solved
      const double r = far.dist[pj];
      const double* dv = &far.dvec[3 * pj];
      const PairClass pc = classify_nonbonded_entry(i, j, n, atoms.tag[i], atoms.tag[j], dv);
      if (pc == PairClass::None) continue;
      if (r < cut.nonb) {
        const double w = pair_w(r, ti, tj);
        if (ki >= 0) b[static_cast<std::size_t>(ki)] -= w * q_in[o]; else b[static_cast<std::size_t>(ko)] -= w * q_in[i];   // fixed charge field
      }
      if (soft_fixed) {
        const double bs = bsoft(r, ti, tj);
        if (bs > 0.0) dX[static_cast<std::size_t>(ki >= 0 ? ki : ko)] -= bs;
      }
    }
  }

  // ---- solve
  const bool iterative = opt.solver == Acks2SolverKind::Iterative || (opt.solver == Acks2SolverKind::Auto && static_cast<int>(m) > opt.dense_max_atoms);
  out.iterative = iterative;
  std::vector<double> x(N, 0.0);
  auto matvec = [&](const std::vector<double>& v, std::vector<double>& y) {
    const double lam = v[2 * m];
    double ssum = 0.0;
    std::vector<double> gsum(G, 0.0);
    for (std::size_t k = 0; k < m; ++k) {
      double aq = S.dq[k] * v[k] + v[m + k] + v[2 * m + 1 + static_cast<std::size_t>(S.gidx[k])];
      for (std::size_t c = S.H.start[k]; c < S.H.start[k + 1]; ++c) aq += S.H.val[c] * v[static_cast<std::size_t>(S.H.col[c])];
      y[k] = aq;
      double as = v[k] + (S.xd[k] + dX[k]) * v[m + k] + lam;
      for (std::size_t c = S.X.start[k]; c < S.X.start[k + 1]; ++c) as += S.X.val[c] * v[m + static_cast<std::size_t>(S.X.col[c])];
      y[m + k] = as;
      ssum += v[m + k];
      gsum[static_cast<std::size_t>(S.gidx[k])] += v[k];
    }
    y[2 * m] = ssum;
    for (std::size_t g = 0; g < G; ++g) y[2 * m + 1 + g] = gsum[g];
  };
  if (!iterative) {
    // dense: factorisation reused while the matrix and the fixed-particle diagonal are unchanged
    if (!(opt.cache && S.lu_valid && S.lu_dX == dX)) {
      S.lu.assign(N * N, 0.0);
      auto at = [&](std::size_t r, std::size_t c) -> double& { return S.lu[r * N + c]; };
      for (std::size_t k = 0; k < m; ++k) {
        at(k, k) = S.dq[k]; at(k, m + k) = 1.0; at(m + k, k) = 1.0; at(m + k, m + k) = S.xd[k] + dX[k];
        at(m + k, 2 * m) = 1.0; at(2 * m, m + k) = 1.0;
        at(k, 2 * m + 1 + static_cast<std::size_t>(S.gidx[k])) = 1.0; at(2 * m + 1 + static_cast<std::size_t>(S.gidx[k]), k) = 1.0;
        for (std::size_t c = S.H.start[k]; c < S.H.start[k + 1]; ++c) at(k, static_cast<std::size_t>(S.H.col[c])) += S.H.val[c];
        for (std::size_t c = S.X.start[k]; c < S.X.start[k + 1]; ++c) at(m + k, m + static_cast<std::size_t>(S.X.col[c])) += S.X.val[c];
      }
      lu_factor(S.lu, S.piv, N);
      S.lu_dX = dX; S.lu_valid = opt.cache;
    }
    x = b;
    lu_solve(S.lu, S.piv, x, N);
  } else {
    // diagonal (Jacobi) preconditioner |diag A|; constraint rows 1; s rows floored at 1 eV so that an atom without softness neighbours does not blow up
    std::vector<double> minv(N, 1.0);
    for (std::size_t k = 0; k < m; ++k) {
      minv[k] = 1.0 / std::max(std::fabs(S.dq[k]), 1.0e-3);
      minv[m + k] = 1.0 / std::max(std::fabs(S.xd[k] + dX[k]), 1.0);
    }
    const bool warm = opt.warm_start && S.prev_m == m && S.prev_G == G && S.xprev.size() == N;
    out.warm = warm;
    if (warm) x = S.xprev;
    else for (std::size_t k = 0; k < m; ++k) x[k] = S.ref[k];
    bool conv = false;
    out.iterations = minres(matvec, minv, b, x, opt.tol, opt.max_iter, conv);
    if (!conv) throw SystemError("ACKS2: MINRES did not converge in " + std::to_string(opt.max_iter) + " iterations (tolerance " + std::to_string(opt.tol) + ")");
    // The energy is first order in a violation of the charge constraints (it is only stationary on the constraint surface), so remove it exactly: shift the
    // charges of every group to its total charge and the kinetic potentials to sum 0. Costs O(m), changes the solution by less than the residual.
    {
      std::vector<double> gs(G, 0.0), gn(G, 0.0);
      double ss = 0.0;
      for (std::size_t k = 0; k < m; ++k) { gs[static_cast<std::size_t>(S.gidx[k])] += x[k]; gn[static_cast<std::size_t>(S.gidx[k])] += 1.0; ss += x[m + k]; }
      for (std::size_t k = 0; k < m; ++k) {
        const std::size_t g = static_cast<std::size_t>(S.gidx[k]);
        x[k] -= (gs[g] - group_q[g]) / gn[g];
        x[m + k] -= ss / static_cast<double>(m);
      }
    }
    S.xprev = x; S.prev_m = m; S.prev_G = G;
  }
  // true residual (eV)
  std::vector<double> Ax(N);
  matvec(x, Ax);
  double res = 0.0;
  for (std::size_t r = 0; r < N; ++r) res = std::max(res, std::fabs(Ax[r] - b[r]));
  out.residual = res;
  for (std::size_t i = 0; i < n; ++i)
    if (loc[i] >= 0) { out.q[i] = x[static_cast<std::size_t>(loc[i])]; out.s[i] = x[m + static_cast<std::size_t>(loc[i])]; }
  return out;
}

Acks2Solution solve_acks2_ereaxff(const ForceField& ff, const NeighborCutoffs& cut, const AtomSet& atoms, const FarList& far, const std::vector<double>& q_in,
                                  const std::vector<int>& group, const std::vector<double>& group_q, const Acks2Options& opt) {
  Acks2Engine e;
  Acks2Options o = opt; o.solver = Acks2SolverKind::Dense; o.cache = false;
  return e.solve(ff, cut, atoms, far, q_in, group, group_q, {}, o);
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
