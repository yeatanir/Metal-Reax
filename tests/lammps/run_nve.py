#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""NVE-1: LAMMPS-hosted NVE drift of `reaxff/metal` vs stock `reaxff` under the identical protocol (same data, velocities, dt, QEq tolerance,
neighbor settings). Criteria (NUMERICAL_POLICY 5.2, provisional): drift slope <= max(2 x stock slope, 2e-4 kcal/mol/atom/ps); total-energy RMS
fluctuation within 10 % of stock; mean T within 3 sigma of the block averages; no NaN/Inf. Both runs go in parallel.
usage: run_nve.py --lmp BIN --plugin SO --ffield-dir DIR --case NAME [--replicate 3] [--steps 80000] [--backend metal] [--gpu-qeq] [--dt 0.25]"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "reaxref"))
import runner  # noqa: E402


def thermo_rows(text):
    rows, on, nc = [], False, 0
    for line in text.splitlines():
        if line.startswith("   Step"): on = True; nc = len(line.split()); continue
        if line.startswith("Loop time"): on = False
        if on:
            try:
                v = [float(x) for x in line.split()]
                if len(v) == nc: rows.append(v)   # warnings printed between thermo lines are skipped
            except ValueError: pass
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--case", required=True); ap.add_argument("--replicate", type=int, default=1); ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--dt", type=float, default=0.25); ap.add_argument("--temp", type=float, default=300.0); ap.add_argument("--qeq-tol", default="1e-6")
    ap.add_argument("--backend", default="metal"); ap.add_argument("--gpu-qeq", action="store_true"); ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--keep", default=""); ap.add_argument("--npt", action="store_true", help="INT-4: fix npt iso 1 atm instead of NVE; statistics (<T>, <V>, <P>) compared within 3 block sigma"); ap.add_argument("--equil", type=int, default=4000, help="NVT equilibration steps with the stock style before the compared runs (same start for both)")
    a = ap.parse_args()
    case = json.loads((ROOT / "tests" / "fixtures" / "cases" / f"{a.case}.json").read_text())
    ffpath = Path(a.ffield_dir) / case["ffield"]["name"]
    ffp = runner.parse_ffield(ffpath)
    els = " ".join(case["elements"])
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"), LAMMPS_POTENTIALS=a.ffield_dir)
    qfix = "qeq/reaxff/metal" if a.gpu_qeq else "qeq/reaxff"
    with tempfile.TemporaryDirectory() as td:
        td = Path(a.keep) if a.keep else Path(td); td.mkdir(parents=True, exist_ok=True)
        runner.write_data(case, ffp, td / "data.lmp")
        if a.equil > 0:   # common starting point: minimised, thermalised by the stock style; the compared runs restart from the data file
            (td / "in.eq").write_text(f"""units real
atom_style charge
atom_modify map array
boundary p p p
read_data data.lmp
replicate {a.replicate} {a.replicate} {a.replicate}
pair_style reaxff NULL
pair_coeff * * "{ffpath}" {els}
neighbor 2.0 bin
fix q all qeq/reaxff 1 0.0 10.0 1e-8 reaxff maxiter 500
minimize 1e-8 1e-8 500 5000
reset_timestep 0
velocity all create {a.temp} 4928 dist gaussian
fix integ all nvt temp {a.temp} {a.temp} 25.0
timestep {a.dt}
run {a.equil}
write_data eq.data nocoeff
""")
            r = subprocess.run([a.lmp, "-in", "in.eq", "-log", "none", "-nocite"], cwd=td, env=env, capture_output=True, text=True)
            if r.returncode: print(r.stdout[-800:], r.stderr[-800:]); return 1
        INTEG = f'fix integ all npt temp {a.temp} {a.temp} 25.0 iso 1.0 1.0 250.0' if a.npt else 'fix integ all nve'
        procs = {}
        for tag in ("stock", "ours"):
            ours = tag == "ours"
            style = f"plugin load {a.plugin}\npair_style reaxff/metal NULL backend {a.backend}" if ours else "pair_style reaxff NULL"
            q = qfix if ours else "qeq/reaxff"
            (td / f"in.{tag}").write_text(f"""units real
atom_style charge
atom_modify map array
boundary p p p
{('read_data eq.data' if a.equil > 0 else f'read_data data.lmp' + chr(10) + f'replicate {a.replicate} {a.replicate} {a.replicate}')}
{style}
pair_coeff * * "{ffpath}" {els}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
fix q all {q} 1 0.0 10.0 {a.qeq_tol} reaxff maxiter 500
{'' if a.equil > 0 else f'velocity all create {a.temp} 4928 dist gaussian'}
{INTEG}
thermo_style custom step temp pe ke etotal press vol
thermo_modify format float %.12g
thermo {a.every}
timestep {a.dt}
run {a.steps}
""")
            procs[tag] = subprocess.Popen([a.lmp, "-in", f"in.{tag}", "-log", "none", "-nocite"], cwd=td, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = {t: p.communicate()[0] for t, p in procs.items()}
        res = {}
        for t in out:
            rows = thermo_rows(out[t])
            if len(rows) < 10: print(t, "produced no thermo:\n", out[t][-800:]); return 1
            res[t] = rows
    n = len(case["atoms"]) * a.replicate ** 3
    ok = True
    if a.npt:
        def blk(x): return np.mean(x), np.std([b.mean() for b in np.array_split(x, 10)])
        for name, col in (("T", 1), ("P", 5), ("V", 6)):
            (ms, ss), (mo, so) = blk(res["stock"][len(res["stock"]) // 5:, col]), blk(res["ours"][len(res["ours"]) // 5:, col])
            good = abs(mo - ms) <= 3 * max(ss, so) and bool(np.isfinite(res["ours"]).all())
            print(f"  {'ok  ' if good else 'FAIL'} <{name}> stock {ms:.5g} +- {ss:.3g}  ours {mo:.5g} +- {so:.3g}"); ok &= good
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    st = {}
    for t, rows in res.items():
        ps = rows[:, 0] * a.dt * 1e-3
        e = rows[:, 4] / n
        finite = bool(np.isfinite(rows).all())
        slope = np.polyfit(ps, e, 1)[0]
        res_e = e - np.polyval(np.polyfit(ps, e, 1), ps)
        blocks = np.array_split(rows[:, 1], 10)
        st[t] = dict(finite=finite, slope=slope, rms=res_e.std(), T=rows[:, 1].mean(), Tsig=np.std([b.mean() for b in blocks]), n=len(rows))
        print(f"{t:6s} samples {len(rows)}  drift {slope:+.3e} kcal/mol/atom/ps  Etot residual RMS {res_e.std():.3e}  <T> {rows[:,1].mean():.3f} (block sigma {st[t]['Tsig']:.3f})  finite {finite}")
    s, o = st["stock"], st["ours"]
    lim = max(2 * abs(s["slope"]), 2e-4)
    c = [("no NaN/Inf", o["finite"]), (f"|drift| {abs(o['slope']):.2e} <= max(2 x stock, 2e-4) = {lim:.2e}", abs(o["slope"]) <= lim),
         ("Etot RMS within 10% of stock (or both <1e-5)", abs(o["rms"] - s["rms"]) <= 0.1 * s["rms"] or max(o["rms"], s["rms"]) < 1e-5),
         ("mean T within 3 block sigma", abs(o["T"] - s["T"]) <= 3 * max(s["Tsig"], o["Tsig"]))]
    for name, v in c:
        print(("  ok   " if v else "  FAIL ") + name); ok &= v
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
