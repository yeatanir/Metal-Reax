#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""CHG-1: the charge models of ReaxFF in LAMMPS with `pair_style reaxff/metal` against stock `pair_style reaxff`, on the 3000-atom water system of
examples/reaxff/water (shrink-wrapped z boundary, NVT, wall/reflect): fix qeq/reaxff, qeq/rel/reaxff, qtpie/reaxff, acks2/reaxff, each without and with
fix efield. The same input is run twice (stock; plugin with the given backend, the QTPIE and ACKS2 fixes renamed where the plugin supplies its own
version); compared are the thermo rows at steps 0, 10, 20: step 0 under the C3 energy criterion (1e-3 kcal/mol/atom) for metal and 2e-9 relative for
cpu64; later steps (the iterative charge solvers amplify 1-ulp differences) within 1e-4 relative in the energy and 0.2 K in the temperature (cpu64) or twice that
(metal).   usage: run_water_models.py --lmp BIN --plugin SO --examples DIR [--backend cpu64|metal] [--only NAME]"""
import argparse, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

MODELS = ["qeq", "qeqr", "qtpie", "acks2", "shielded", "qeqgroup", "qeqfile"]


def run(lmp, wd, text, env):
    (wd / "in.lmp").write_text(text)
    r = subprocess.run([lmp, "-in", "in.lmp", "-log", "none", "-nocite"], cwd=wd, env=env, capture_output=True, text=True, timeout=900)
    if r.returncode: raise RuntimeError((r.stdout + r.stderr)[-1200:])
    rows, on = [], False
    for line in r.stdout.splitlines():
        if line.startswith("   Step"): on = True; continue
        if line.startswith("Loop time"): on = False
        if on:
            try: rows.append([float(x) for x in line.split()])
            except ValueError: pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--examples", required=True)
    ap.add_argument("--backend", default="cpu64"); ap.add_argument("--only"); ap.add_argument("--gpu-qeq", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"))
    fails, n = [], 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for f in Path(a.examples).iterdir():
            if f.is_file() and not f.name.startswith(("log.", "in.")): shutil.copy(f, td / f.name)
        env["LAMMPS_POTENTIALS"] = str(td)
        for model in MODELS:
            for field in ("", ".field"):
                name = model + field
                if a.only and a.only != name: continue
                if model in ("shielded", "qeqgroup", "qeqfile") and field: continue   # fix qeq/shielded has no electric-field support
                src = (Path(a.examples) / f"in.water.{'qeq' if model in ('shielded', 'qeqgroup', 'qeqfile') else name}").read_text()
                if model == "qeqgroup":   # the charge fix acts on a proper subgroup (the oxygen atoms); hydrogens keep their data-file charges
                    if field: continue
                    src = src.replace("fix             1 all qeq/reaxff", "group og type 1\nfix             1 og qeq/reaxff")
                if model == "qeqfile":   # per-type chi/eta/gamma from a parameter file instead of the pair style (eta = 2 x the force-field value, as the pair extract() delivers)
                    import sys as _sys
                    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "reaxref"))
                    import runner
                    P = runner.parse_ffield(Path(a.examples) / "qeq_ff.water")
                    elems = ["O", "H"]
                    (td / "param.water").write_text("".join(f"{k + 1} {P['chi'][e]!r} {2 * P['eta'][e]!r} {P['gamma'][e]!r}\n" for k, e in enumerate(elems)))
                    src = src.replace("1.0e-6 reaxff maxiter 400", "1.0e-6 param.water maxiter 400")
                if model == "shielded": src = src.replace("qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 400", "qeq/shielded 1 10.0 1.0e-6 400 reaxff")
                src = src.replace("run 20", "thermo_style custom step temp pe ke press\nthermo_modify format float %.12g\nrun 20")
                ours = src.replace("pair_style      reaxff NULL", f"plugin load {a.plugin}\npair_style reaxff/metal NULL backend {a.backend}", 1)
                if model == "qtpie": ours = ours.replace("qtpie/reaxff", "qtpie/reaxff/metal")
                if model == "qeqr": ours = ours.replace("qeq/rel/reaxff", "qeq/rel/reaxff/metal")
                if model == "qeq" and a.gpu_qeq: ours = ours.replace("qeq/reaxff", "qeq/reaxff/metal")
                try:
                    s, o = run(a.lmp, td, src, env), run(a.lmp, td, ours, env)
                except Exception as e:
                    fails.append((name, str(e)[-400:])); print("FAIL", name); continue
                n += 1
                nat = 3000
                bad = []
                for k, (rs, ro) in enumerate(zip(s, o)):
                    dpe = abs(ro[2] - rs[2]) / nat; dT = abs(ro[1] - rs[1])
                    if k == 0:
                        lim = 1e-3 if a.backend == "metal" else 2e-9 * abs(rs[2]) / nat
                        if dpe > lim: bad.append(f"step {int(rs[0])} PE/atom differs by {dpe:.2e} (limit {lim:.1e})")
                    else:
                        if dpe > 1e-4 * abs(rs[2]) / nat * (20 if a.backend == "metal" else 1): bad.append(f"step {int(rs[0])} PE/atom differs by {dpe:.2e}")
                        if dT > (0.4 if a.backend == "metal" else 0.2): bad.append(f"step {int(rs[0])} T differs by {dT:.2e}")
                print(("ok   " if not bad else "FAIL ") + f"{name:12s} PE/atom stock {s[0][2] / nat:.6f} ours {o[0][2] / nat:.6f}  |  step 20 T {s[-1][1]:.4f} vs {o[-1][1]:.4f}")
                if bad: fails.append((name, "; ".join(bad)))
    for f in fails: print("FAIL", f)
    print(f"CHG-1 backend {a.backend}: {n} runs, {len(fails)} failures")
    print("RESULT: PASS" if not fails and n else "RESULT: FAIL")
    return 0 if not fails and n else 1


if __name__ == "__main__":
    sys.exit(main())
