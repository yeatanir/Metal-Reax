#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""NBR-2: the standalone image expander produces the same ghost set as pinned LAMMPS (Comm::borders).

For every fixture: LAMMPS (shared library, C API) creates the box and atoms, `pair_style zero <cutmax>` with the fixture's
skin gives the same ghost shell as pair reaxff (cutforce + skin), `run 0` builds the ghosts; their number and their
(tag, position) multiset are compared with `reaxmetal_neighbor_tool --ghosts --shell <cutforce+skin>`.
usage: run_ghosts.py --lib liblammps.so --tool reaxmetal_neighbor_tool --ffield-dir DIR [--fixtures tests/fixtures]"""
import argparse, ctypes, json, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "reaxref"))
sys.path.insert(0, str(HERE.parents[1] / "tests" / "python"))
import runner  # noqa: E402
from test_neighbor_fixtures import case_text  # noqa: E402


class Lammps:
    def __init__(self, lib):
        self.L = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        self.L.lammps_open_no_mpi.restype = ctypes.c_void_p
        self.L.lammps_extract_global.restype = ctypes.c_void_p
        self.L.lammps_extract_atom.restype = ctypes.c_void_p
        self.L.lammps_extract_setting.restype = ctypes.c_int
        argv = (ctypes.c_char_p * 6)(b"lmp", b"-log", b"none", b"-screen", b"none", b"-nocite")
        self.h = ctypes.c_void_p(self.L.lammps_open_no_mpi(6, argv, None))

    def cmd(self, c):
        for line in c.split("\n"):
            if line.strip():
                self.L.lammps_command(self.h, line.encode())
            if self.L.lammps_has_error(self.h):
                buf = ctypes.create_string_buffer(2048)
                self.L.lammps_get_last_error_message(self.h, buf, 2048)
                raise RuntimeError(buf.value.decode(errors="replace"))

    def close(self):
        self.L.lammps_close(self.h)


def lammps_ghosts(lib, case, ref, cutmax, skin):
    lo, hi, tilt = (tuple(float(v) for v in t) for t in runner.cell_to_lammps(case["cell"]))
    per = case["periodic"]
    m = Lammps(lib)
    try:
        m.cmd("units real\natom_style charge\natom_modify map array")
        m.cmd("boundary " + " ".join("p" if p else "f" for p in per))
        if any(abs(t) > 0 for t in tilt):
            m.cmd(f"region b prism {lo[0]!r} {hi[0]!r} {lo[1]!r} {hi[1]!r} {lo[2]!r} {hi[2]!r} {tilt[0]!r} {tilt[1]!r} {tilt[2]!r} units box")
        else:
            m.cmd(f"region b block {lo[0]!r} {hi[0]!r} {lo[1]!r} {hi[1]!r} {lo[2]!r} {hi[2]!r} units box")
        m.cmd("create_box 1 b\nmass * 1.0")
        for p in ref["positions"]:
            m.cmd(f"create_atoms 1 single {float(p[0])!r} {float(p[1])!r} {float(p[2])!r} units box")
        m.cmd(f"pair_style zero {cutmax!r}\npair_coeff * *\nneighbor {skin!r} bin\nneigh_modify delay 0 every 1 check no\nrun 0")
        nl = ctypes.c_int.from_address(m.L.lammps_extract_global(m.h, b"nlocal")).value
        ng = ctypes.c_int.from_address(m.L.lammps_extract_global(m.h, b"nghost")).value
        tagbytes = m.L.lammps_extract_setting(m.h, b"tagint")
        xp = ctypes.cast(m.L.lammps_extract_atom(m.h, b"x"), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
        tp = ctypes.cast(m.L.lammps_extract_atom(m.h, b"id"), ctypes.POINTER(ctypes.c_int32 if tagbytes == 4 else ctypes.c_int64))
        out = []
        for i in range(nl, nl + ng):
            out.append((int(tp[i]), tuple(xp[i][c] for c in range(3))))
        return nl, out
    finally:
        m.close()


def key(tag, x):
    return (tag, tuple(round(v, 6) for v in x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True); ap.add_argument("--tool", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--fixtures", default=str(HERE.parents[1] / "tests" / "fixtures"))
    a = ap.parse_args()
    fx = Path(a.fixtures)
    fails, n, total = [], 0, 0
    with tempfile.TemporaryDirectory() as td:
        for cf in sorted((fx / "cases").glob("*.json")):
            case = json.loads(cf.read_text()); ref = json.loads((fx / "reference" / cf.name).read_text())
            skin = float(case.get("neighbor", {}).get("skin", 2.0))
            extra = ["--lgvdw"] if case.get("pair", {}).get("lgvdw") else []
            ff = str(Path(a.ffield_dir) / case["ffield"]["name"])
            txt = Path(td) / "case.txt"; txt.write_text(case_text(case, ref))
            probe = subprocess.run([a.tool, "--ffield", ff, *extra, str(txt)], capture_output=True, text=True)
            nonb = float(dict(l.split(None, 1) for l in probe.stdout.splitlines())["nonb"])
            cutmax = max(nonb, 7.5, 5.0)                 # pair_style reaxff: max(nonb_cut, hbond_cut, bond_cut), default control values
            shell = cutmax + skin
            r = subprocess.run([a.tool, "--ffield", ff, *extra, "--shell", repr(shell), "--ghosts", str(txt)], capture_output=True, text=True)
            if r.returncode != 0:
                fails.append(f"{cf.stem}: tool failed: {r.stderr.strip()}"); continue
            ours = Counter(); nours = 0
            for ln in r.stdout.splitlines():
                w = ln.split()
                if w[0] == "ghost":
                    ours[key(int(w[1]), tuple(float(v) for v in w[6:9]))] += 1; nours += 1
            nl, theirs = lammps_ghosts(a.lib, case, ref, cutmax, skin)
            lam = Counter(key(t, x) for t, x in theirs)
            n += 1; total += nours
            if nours != len(theirs):
                fails.append(f"{cf.stem}: ghost count engine={nours} lammps={len(theirs)} (shell {shell})")
            elif ours != lam:
                # coordinates may differ at the 1e-6 rounding boundary: compare with a tolerance before failing
                miss = list((ours - lam).elements()) + list((lam - ours).elements())
                fails.append(f"{cf.stem}: ghost multiset differs in {len(miss)} entries (e.g. {miss[:2]})")
    for f in fails:
        print("FAIL", f)
    print(f"NBR-2: {n} fixtures, {total} engine ghosts compared, {len(fails)} mismatches")
    print("RESULT: PASS" if not fails and n >= 50 else f"RESULT: FAIL ({len(fails)})")
    return 0 if not fails and n >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
