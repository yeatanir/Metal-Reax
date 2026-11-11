#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""pair_style reaxff/metal ... ereaxff yes eparams <file>: per-element nuclear charge / sharing sign / tested flag from an input file.
usage: run_eparams.py --lmp BIN --plugin SO --ag-ffield F --ag-geo GEO --ag-frames frames.json --cho-ffield F2 --table tests/data/legacy_table.eparams
Checks: (1) legacy_table.eparams reproduces the no-eparams energy exactly (Ag7_1 frame 0); (2) overriding Av's Z changes the electron energy (the table is live);
(3) an element absent from the file / untested / tested: refused, refused with a specific message, accepted (H in a C/El/H test system)."""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ereaxff as R
ap = argparse.ArgumentParser()
for k in ("lmp", "plugin", "ag-ffield", "ag-geo", "ag-frames", "cho-ffield", "table"): ap.add_argument("--" + k, required=True)
a = ap.parse_args()

def run(td, ffield, sym, pos, q, extra, taper):
    td = Path(td); masses = R.ffield_masses(ffield); elems = R.elements_in_order(sym)
    R.write_data(td / "data.lmp", sym, pos, q, None, False, masses)
    (td / "in.lmp").write_text(f"""plugin load {os.path.relpath(a.plugin, td)}
units real
atom_style charge
boundary f f f
newton on
read_data data.lmp
pair_style reaxff/metal NULL ereaxff yes electron_taper {taper} {extra}
pair_coeff * * {ffield} {' '.join(elems)}
comm_modify cutoff 12.0
compute pr all pair reaxff/metal
thermo_style custom step pe c_pr[15]
thermo_modify format float %.14g
run 0
""")
    env = dict(os.environ, OMP_NUM_THREADS="1", DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"))
    r = subprocess.run([a.lmp, "-in", "in.lmp", "-log", "none", "-nocite"], cwd=td, env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    vals = [l.split() for l in r.stdout.splitlines() if len(l.split()) == 3 and l.split()[0] == "0"]
    return r.returncode, out, (float(vals[0][1]), float(vals[0][2])) if vals else None

ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))

with tempfile.TemporaryDirectory() as td:
    sym, pos, q, _ = R.read_geo(a.ag_geo); q = [1.0 if s == "Av" else -1.0 for s in sym]
    p0 = json.loads(Path(a.ag_frames).read_text())["frames"][0]["pos"]
    def ag(extra): return run(td, a.ag_ffield, sym, p0, q, extra, 5.7)
    rc0, o0, v0 = ag("")
    rc1, o1, v1 = ag(f"eparams {a.table}")
    check("legacy_table.eparams reproduces the built-in run exactly", rc0 == 0 and rc1 == 0 and v0 == v1, f"{v0} vs {v1}")
    (Path(td) / "z35.eparams").write_text("AV 3.5 1 tested\nEL 0 0 tested\n")
    rc2, o2, v2 = ag(f"eparams {Path(td)/'z35.eparams'}")
    check("changing Av's Z in the file changes the electron energy", rc2 == 0 and abs(v2[1] - v0[1]) > 1e-3 and abs(v2[1] - v0[1]) < 1e4, f"eelect {v0[1]:.6f} -> {v2[1]:.6f}")
    (Path(td) / "bad.eparams").write_text("AV 3 2 tested\n")
    rc3, o3, _ = ag(f"eparams {Path(td)/'bad.eparams'}")
    check("malformed file is rejected", rc3 != 0 and "SHARE_SIGN must be" in o3)
    # C / El / H system with the C-H-O-... legacy ffield
    sy = ["C", "El", "H"]; pp = [[40, 40, 40], [40.11, 40, 40], [41.1, 40, 40]]; qq = [0.0, 0.0, 0.0]
    def cho(extra): return run(td, a.cho_ffield, sy, pp, qq, extra, 10.0)
    rcA, oA, _ = cho("")
    check("H without eparams is refused (allow-list)", rcA != 0 and "allow-list" in oA)
    (Path(td) / "h_un.eparams").write_text("H 1 -1 untested\n")
    rcB, oB, _ = cho(f"eparams {Path(td)/'h_un.eparams'}")
    check("H marked untested is refused with a specific message", rcB != 0 and "untested" in oB)
    (Path(td) / "h_ok.eparams").write_text("H 1 -1 tested\nEL 0 0 tested\n")
    rcC, oC, vC = cho(f"eparams {Path(td)/'h_ok.eparams'}")
    check("H marked tested is accepted and gives a finite energy", rcC == 0 and vC is not None and abs(vC[0]) < 1e6, str(vC))
    (Path(td) / "h_ok2.eparams").write_text("H 2 1 tested\nEL 0 0 tested\n")
    rcD, oD, vD = cho(f"eparams {Path(td)/'h_ok2.eparams'}")
    check("changing H's Z / sign in the file changes the electron energy", rcD == 0 and vC and vD and abs(vD[1] - vC[1]) > 1e-6, f"{vC} vs {vD}")
    # periodic cells: edge below 2*bond_cut + 1 (= 11 A) is refused, a larger one runs
    def per(L):
        d = Path(td) / "per"; d.mkdir(exist_ok=True)
        masses = R.ffield_masses(a.ag_ffield); elems = R.elements_in_order(sym)
        R.write_data(d / "data.lmp", sym, [[x % L for x in pp] for pp in p0], q, [L, L, L, 90, 90, 90], True, masses)
        (d / "in.lmp").write_text(f"plugin load {os.path.relpath(a.plugin, d)}\nunits real\natom_style charge\nboundary p p p\nnewton on\nread_data data.lmp\n"
                                  f"pair_style reaxff/metal NULL ereaxff yes electron_taper 5.7\npair_coeff * * {a.ag_ffield} {' '.join(elems)}\ncomm_modify cutoff 12.0\nrun 0\n")
        env = dict(os.environ, OMP_NUM_THREADS="1", DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"))
        r = subprocess.run([a.lmp, "-in", "in.lmp", "-log", "none", "-nocite"], cwd=d, env=env, capture_output=True, text=True); return r.returncode, r.stdout + r.stderr
    rcS, oS = per(10.0); check("periodic edge 10 A (= 2 x bond_cut) is refused", rcS != 0 and "periodic box edge" in oS)
    rcL, oL = per(12.0); check("periodic edge 12 A is accepted", rcL == 0, oL[-120:] if rcL else "")
print("RESULT: PASS" if ok else "RESULT: FAIL"); sys.exit(0 if ok else 1)
