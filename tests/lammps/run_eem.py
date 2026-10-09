#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""EEM-1: the strict mode and the charge verification of `fix qeq/reaxff/metal` (keywords `strict`, `verify <eV>`), in-LAMMPS:
  strict + a converging solve passes;  strict + too few iterations -> ERROR (did not converge);  the same without strict -> only a warning;
  verify 1e-3 eV passes with the stock CPU matrix and with the GPU matrix;  verify 1e-7 eV fails with the FP32 GPU matrix (the check is sensitive)
  and passes with the double precision CPU matrix;  strict with a taper radius beyond the ghost shell -> ERROR;  malformed keywords -> ERROR.
usage: run_eem.py --lmp BIN --plugin SO --ffield-dir DIR [--metal]"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "reaxref"))
import runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--metal", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"), LAMMPS_POTENTIALS=a.ffield_dir)
    case = json.loads((ROOT / "tests/fixtures/cases/cho_water_box_8.json").read_text())
    ffpath = Path(a.ffield_dir) / case["ffield"]["name"]
    fails = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        runner.write_data(case, runner.parse_ffield(ffpath), td / "data.lmp")
        base = runner.build_input(case, ffpath, td / "data.lmp", td / "x.dump").split("fix integ all nve")[0]
        def run(backend, fixargs, style="qeq/reaxff/metal", tol="1e-10", maxiter=500):
            text = base.replace("pair_style reaxff NULL", f"plugin load {a.plugin}\npair_style reaxff/metal NULL backend {backend}", 1)
            import re
            text = re.sub(r"fix q all qeq/reaxff 1 0.0 ([\d.]+) [^\n]*", lambda m: f"fix q all {style} 1 0.0 {m.group(1)} {tol} reaxff maxiter {maxiter} {fixargs}", text)
            text += "thermo_style custom step pe\nrun 0\n"
            (td / "in.lmp").write_text(text)
            r = subprocess.run([a.lmp, "-in", "in.lmp", "-log", "none", "-nocite"], cwd=td, env=env, capture_output=True, text=True, timeout=300)
            return r.returncode, r.stdout + r.stderr
        def expect(name, res, ok_rc, needle=None):
            rc, out = res
            good = (rc == 0) == ok_rc and (needle is None or needle in out)
            print(("ok   " if good else "FAIL ") + name)
            if not good: fails.append((name, rc, out[-300:]))
        backends = ["cpu64"] + (["metal"] if a.metal else [])
        for b in backends:
            expect(f"[{b}] strict, converging solve", run(b, "strict"), True)
            expect(f"[{b}] strict, maxiter 3 -> error", run(b, "strict", tol="1e-12", maxiter=3), False, "did not converge")
            expect(f"[{b}] stock behaviour (no strict), maxiter 3 -> warning only", run(b, "", tol="1e-12", maxiter=3), True, "convergence failed")
            expect(f"[{b}] verify 1e-3 eV passes", run(b, "verify 1e-3"), True)
            expect(f"[{b}] verify without value -> error", run(b, "verify"), False, "needs a residual")
        expect("[cpu64] verify 1e-7 eV passes with the double precision matrix (residual ~3e-9 eV)", run("cpu64", "verify 1e-7"), True)
        if a.metal:
            expect("[metal] verify 1e-7 eV fails with the FP32 matrix", run("metal", "verify 1e-7"), False, "equalisation residual")
        # taper radius beyond the ghost shell (comm cutoff = pair cutoff 10 A here)
        text = base.replace("pair_style reaxff NULL", f"plugin load {a.plugin}\npair_style reaxff/metal NULL backend cpu64", 1)
        import re
        text = re.sub(r"fix q all qeq/reaxff 1 0.0 [\d.]+ ", "fix q all qeq/reaxff/metal 1 0.0 14.0 ", text) + "thermo_style custom step pe\nrun 0\n"
        for extra, ok, needle, label in (("strict", False, "exceeds the ghost cutoff", "strict, taper 14 A > ghost cutoff -> error"), ("", True, None, "stock behaviour, taper 14 A -> silently truncated")):
            (td / "in.lmp").write_text(text.replace("reaxff maxiter 500", f"reaxff maxiter 500 {extra}"))
            r = subprocess.run([a.lmp, "-in", "in.lmp", "-log", "none", "-nocite"], cwd=td, env=env, capture_output=True, text=True, timeout=300)
            expect(label, (r.returncode, r.stdout + r.stderr), ok, needle)
    for f in fails: print("FAIL", f)
    print(f"EEM-1: {len(fails)} failures")
    print("RESULT: PASS" if not fails else "RESULT: FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
