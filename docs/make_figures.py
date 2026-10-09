#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Figures of the README: (1) renders of the 30,000-atom random-carbon NVT run, (2) charts of the measured comparisons (numbers are those recorded in docs/VALIDATION.md and
docs/DEVELOPMENT_LOG.md). usage: make_figures.py <dir with g4000.*.dump>   (writes docs/img/*.png)"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree

OUT = Path(__file__).parent / "img"; OUT.mkdir(exist_ok=True)
BOX = 84.26481
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
C = {"stock": "#7f7f7f", "cpu64": "#1f77b4", "metal": "#d62728", "mixed": "#2ca02c"}


def read(p):
    L = Path(p).read_text().splitlines(); k = [i for i, l in enumerate(L) if l.startswith("ITEM: ATOMS")][0] + 1
    a = np.array([[float(x) for x in l.split()] for l in L[k:] if l.strip()]); a = a[np.argsort(a[:, 0])]
    return int(L[1]), a[:, 1:4] % BOX


def coordination(x, rcut=1.85):
    t = cKDTree(x, boxsize=BOX); pr = t.query_pairs(rcut, output_type="ndarray")
    return np.bincount(pr.ravel(), minlength=len(x)), pr


def renders(dumpdir):
    steps = [0, 120000, 360000]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    cm = ListedColormap(["#bbbbbb", "#4c9be8", "#1a9850", "#f1a340", "#c51b7d"])   # CN 1 grey, 2 blue, 3 green (sp2), 4 orange, >=5 magenta
    for a, s in zip(ax, steps):
        step, x = read(Path(dumpdir) / f"g4000.{s}.dump"); cn, pr = coordination(x)
        sl = (x[:, 2] > 30) & (x[:, 2] < 44)   # a 14 A slab
        idx = np.where(sl)[0]; inslab = np.zeros(len(x), bool); inslab[idx] = True
        for i, j in pr:
            if inslab[i] and inslab[j]:
                d = x[j, :2] - x[i, :2]; d -= BOX * np.round(d / BOX)
                if np.linalg.norm(d) < 2.0: a.plot([x[i, 0], x[i, 0] + d[0]], [x[i, 1], x[i, 1] + d[1]], color="#555555", lw=0.5, zorder=1)
        a.scatter(x[idx, 0], x[idx, 1], c=np.clip(cn[idx], 1, 5) - 1, cmap=cm, vmin=0, vmax=4, s=9, zorder=2, linewidths=0)
        a.set_xlim(0, BOX); a.set_ylim(0, BOX); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"step {step:,}  ({step * 0.25e-3:.0f} ps)\n{np.mean(cn == 3) * 100:.0f}% 3-coordinated (sp$^2$-like)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in zip(["#bbbbbb", "#4c9be8", "#1a9850", "#f1a340"], ["1 neighbour", "2", "3 (sp$^2$-like)", "4"])]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("30,000 random carbon atoms, 1 g/cm$^3$, NVT 4000 K, 2013 C ReaxFF on Metal: 14 Å slab coloured by coordination", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97)); fig.savefig(OUT / "graphitization_renders.png"); plt.close(fig)
    # time series
    rows = []
    for s in sorted(int(p.name.split(".")[1]) for p in Path(dumpdir).glob("g4000.*.dump")):
        step, x = read(Path(dumpdir) / f"g4000.{s}.dump"); cn, pr = coordination(x)
        rows.append((step * 0.25e-3, [np.mean(cn == k) for k in (1, 2, 3, 4)], len(pr)))
    t = np.array([r[0] for r in rows]); f = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (lab, col) in enumerate(zip(["1", "2", "3 (sp$^2$-like)", "4"], ["#999999", "#4c9be8", "#1a9850", "#f1a340"])): ax[0].plot(t, f[:, k], "o-", color=col, label=f"{lab} neighbours")
    ax[0].set_xlabel("time (ps)"); ax[0].set_ylabel("atom fraction"); ax[0].legend(frameon=False); ax[0].set_title("coordination of the random-carbon network")
    ax[1].plot(t, [r[2] / 30000 for r in rows], "o-", color="#444444"); ax[1].set_xlabel("time (ps)"); ax[1].set_ylabel("C–C bonds per atom"); ax[1].set_title("network connectivity")
    fig.tight_layout(); fig.savefig(OUT / "graphitization_timeseries.png"); plt.close(fig)


def charts():
    # 1 speedup vs size (idle machine, CHO water NVT, serial LAMMPS): stock vs Metal pair + GPU QEq, ms/step
    n = np.array([648, 5184, 24000, 65856]); stock = np.array([5.1, 32.6, 153, 411]); metal = np.array([5.6, 9.3, 21.2, 39.1])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].loglog(n, stock, "o-", color=C["stock"], label="stock reaxff + qeq/reaxff (1 core)"); ax[0].loglog(n, metal, "o-", color=C["metal"], label="reaxff/metal + qeq/reaxff/metal (M5 Max GPU)")
    ax[0].set_xlabel("atoms"); ax[0].set_ylabel("ms per MD step"); ax[0].legend(frameon=False); ax[0].set_title("time per step (CHO water, NVT)")
    sp = stock / metal; ax[1].semilogx(n, sp, "o-", color=C["metal"]); ax[1].axhline(1, color="#999", ls="--", lw=0.8)
    for x_, y_ in zip(n, sp): ax[1].annotate(f"{y_:.1f}x", (x_, y_), textcoords="offset points", xytext=(0, 7), ha="center")
    ax[1].set_xlabel("atoms"); ax[1].set_ylabel("speedup over stock"); ax[1].set_title("speedup"); ax[1].set_ylim(0, 12.5)
    fig.tight_layout(); fig.savefig(OUT / "speedup.png"); plt.close(fig)
    # 2 accuracy vs stock (INT-2, 58 fixtures): force max / rms for the modes
    modes = ["cpu64", "mixed\n(bonded cpu64)", "metal\n(all FP32)"]; fmax = [4.4e-12, 2.7e-4, 4.0e-3]; frms = [1.2e-12, 1.15e-4, 9.1e-4]
    fig, ax = plt.subplots(figsize=(6, 3.8)); xs = np.arange(3)
    ax.bar(xs - 0.2, fmax, 0.4, color=[C["cpu64"], C["mixed"], C["metal"]], label="max force-component error"); ax.bar(xs + 0.2, frms, 0.4, color=[C["cpu64"], C["mixed"], C["metal"]], alpha=0.5, label="RMS")
    ax.set_yscale("log"); ax.axhline(0.05, color="k", ls=":", lw=0.9); ax.text(2.45, 0.058, "limit 0.05", ha="right", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(modes); ax.set_ylabel("|F$_{plugin}$ − F$_{stock}$|  (kcal/mol/Å)"); ax.set_title("force accuracy vs stock LAMMPS\n(58 reference configurations)"); ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "accuracy.png"); plt.close(fig)
    # 3 NVE energy noise, oxide, 20 ps (or 12000-step runs): RMS of total-energy fluctuation
    lab = ["stock", "cpu64\n(4 ps run)", "metal\n(all FP32)", "mixed:\nbonded cpu64,\nrest GPU"]; rms = [1.105e-4, 1.2e-4, 1.24e-3, 1.07e-4]; drift = [3.7e-6, np.nan, 4.0e-4, 3.4e-6]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8)); cols = [C["stock"], C["cpu64"], C["metal"], C["mixed"]]
    ax[0].bar(range(4), rms, color=cols); ax[0].set_yscale("log"); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(lab, fontsize=8); ax[0].set_ylabel("RMS of E$_{tot}$ about its trend (kcal/mol/atom)"); ax[0].set_title("NVE energy noise, VO oxide (512 atoms, 20 ps)")
    d = [v for v in drift]; ax[1].bar([0, 2, 3], [drift[0], drift[2], drift[3]], color=[cols[0], cols[2], cols[3]]); ax[1].set_yscale("log"); ax[1].set_xticks([0, 2, 3]); ax[1].set_xticklabels([lab[0], lab[2], lab[3]], fontsize=8)
    ax[1].set_ylabel("|energy drift| (kcal/mol/atom/ps)"); ax[1].set_title("NVE energy drift, same runs")
    fig.tight_layout(); fig.savefig(OUT / "nve_noise.png"); plt.close(fig)
    # 4 charge models: PE/atom of the 3000-atom water example, stock vs plugin (cpu64, metal)
    models = ["qeq", "qeq/rel", "qtpie", "acks2", "qeq +field", "qeq/rel +field", "qtpie +field", "acks2 +field"]
    stock = [-85.680993, -85.680993, -85.433701, -84.965812, -79.158769, -85.680919, -85.433728, -80.932334]
    metal = [-85.681003, -85.681003, -85.433711, -84.965822, -79.158779, -85.680930, -85.433739, -80.932345]
    fig, ax = plt.subplots(figsize=(8.5, 3.6)); xs = np.arange(len(models))
    ax.bar(xs - 0.2, np.array(metal) - np.array(stock), 0.4, color=C["metal"], label="metal − stock"); ax.bar(xs + 0.2, np.zeros(len(models)) + 1e-9, 0.4, color=C["cpu64"], label="cpu64 − stock (< 1e-9)")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(xs); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8); ax.set_ylabel("ΔPE per atom (kcal/mol)")
    ax.set_title("every ReaxFF charge model of LAMMPS, 3000-atom water (ΔPE/atom limit 1e-3)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "charge_models.png"); plt.close(fig)
    # 5 FP32 decision-flip windows of Metal around the 48 scanned thresholds (docs/img/decision_windows.json = the Metal column of tests/python/test_threshold_scan.py)
    import json
    w = np.array(json.loads((OUT / "decision_windows.json").read_text()))
    fig, ax = plt.subplots(figsize=(6.2, 3.6)); vals, counts = np.unique(w, return_counts=True)
    ax.bar([f"{v:.0e}" for v in vals], counts, color=["#4c9be8" if v <= 1e-7 else "#f1a340" for v in vals]); ax.set_xlabel("flip window of Metal-32 around the threshold (Å)"); ax.set_ylabel("number of thresholds")
    ax.set_title(f"{len(w)} scanned decision thresholds: Metal takes the reference's\ndecision except within ~1e-7 Å (1e-3: the continuous SBO<=0 branch)"); fig.tight_layout(); fig.savefig(OUT / "decision_windows.png"); plt.close(fig)


if __name__ == "__main__":
    charts()
    if len(sys.argv) > 1: renders(sys.argv[1])
