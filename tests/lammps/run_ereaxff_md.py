#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Short MD with eReaxFF through LAMMPS integrators: forces on El atoms must drive fix nve (energy conservation) and fix temp/berendsen.
Start = the first frame of the ereac reference frames (relaxed geometry), Maxwell velocities.
usage: run_ereaxff_md.py --lmp BIN --plugin SO --ffield F --geo GEO --frames frames.json [--steps 2000] [--dt 0.1] [--temp 298] [--berendsen TAU_FS]"""
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ereaxff as R

ap = argparse.ArgumentParser()
ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--ffield", required=True)
ap.add_argument("--geo", required=True); ap.add_argument("--frames", required=True)
ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--dt", type=float, default=0.1)
ap.add_argument("--temp", type=float, default=298.0); ap.add_argument("--berendsen", type=float, default=0.0)
ap.add_argument("--max-drift", type=float, default=0.05, help="allowed |E_total(end) - E_total(0)| in kcal/mol (NVE only)")
a = ap.parse_args()
sym, pos, q, cell = R.read_geo(a.geo)
p0 = json.loads(Path(a.frames).read_text())["frames"][0]["pos"]
with tempfile.TemporaryDirectory() as td:
    R.write_data(Path(td) / "data.lmp", sym, p0, q, cell, True)
    integ = ("fix t all temp/berendsen %g %g %g\nfix i all nve" % (a.temp, a.temp, a.berendsen)) if a.berendsen > 0 else "fix i all nve"
    (Path(td) / "in.lmp").write_text(f"""plugin load {a.plugin}
units real
atom_style charge
boundary p p p
newton on
read_data data.lmp
pair_style reaxff/metal NULL ereaxff yes electron_taper 10.0
pair_coeff * * {a.ffield} C El
comm_modify cutoff 11.5
velocity all create {a.temp} 4928459 dist gaussian mom yes rot no
timestep {a.dt}
{integ}
thermo_style custom step temp pe ke etotal press
thermo_modify format float %.10g
thermo 100
run {a.steps}
""")
    env = dict(os.environ, OMP_NUM_THREADS="1", DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"))
    r = subprocess.run([a.lmp, "-in", "in.lmp", "-log", "none", "-nocite"], cwd=td, env=env, capture_output=True, text=True)
    if r.returncode: sys.exit((r.stdout + r.stderr)[-1500:])
rows, on = [], False
for line in r.stdout.splitlines():
    if line.lstrip().startswith("Step"): on = True; continue
    if line.startswith("Loop time"): on = False
    if on:
        v = line.split()
        if len(v) == 6: rows.append([float(x) for x in v])
d = np.array(rows)
print("step  temp  pe  ke  etotal")
for row in d[:: max(1, len(d) // 12)]: print("%6d %8.2f %12.4f %10.4f %12.4f" % tuple(row[:5]))
print("T mean %.1f K (target %.0f)  etotal start %.4f end %.4f  max-min %.4f kcal/mol" % (d[:, 1].mean(), a.temp, d[0, 4], d[-1, 4], d[:, 4].max() - d[:, 4].min()))
ok = np.all(np.isfinite(d))
if a.berendsen <= 0:
    ok &= abs(d[-1, 4] - d[0, 4]) <= a.max_drift
    print("NVE drift |E_end - E_0| = %.4f kcal/mol (limit %.3f)" % (abs(d[-1, 4] - d[0, 4]), a.max_drift))
else:
    ok &= abs(d[:, 1].mean() - a.temp) < 0.25 * a.temp
print("RESULT: PASS" if ok else "RESULT: FAIL"); sys.exit(0 if ok else 1)
