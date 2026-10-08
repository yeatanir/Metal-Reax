#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""INT-7 (adapter A2): the ghost-native host view and the engine's far list, checked INSIDE pinned LAMMPS.

For every fixture the geometry is loaded into LAMMPS (C library), `pair_style reaxff/metal ... reaxmetal_selfcheck yes` is
selected and `run 0` is issued. At the first compute() the adapter (plugin/adapter) (1) builds the ghost-native view of
LAMMPS' own owned+ghost arrays and verifies ghost == owner + lattice shift, (2) builds the engine's far list on that view and
compares it row by row with LAMMPS' own half/newton-off/ghost neighbor list, (3) applies the owner-computes rules and counts the
pair classes; then compute() refuses (no force backend) and the summary is read from the error text. The class counts must equal
the interaction tallies recorded from stock pair reaxff in M1 (vdw.oo / vdw.og / vdw.self).
usage: run_a2.py --lib liblammps --plugin reaxmetaladapterplugin.so --ffield-dir DIR [--fixtures tests/fixtures]"""
import argparse, ctypes, json, re, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "reaxref"))
import runner  # noqa: E402


class Lammps:
    def __init__(self, lib):
        self.L = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        self.L.lammps_open_no_mpi.restype = ctypes.c_void_p
        argv = (ctypes.c_char_p * 6)(b"lmp", b"-log", b"none", b"-screen", b"none", b"-nocite")
        self.h = ctypes.c_void_p(self.L.lammps_open_no_mpi(6, argv, None))

    def cmd(self, c):
        """returns None on success, else the error message of the first failing command"""
        for line in c.split("\n"):
            if not line.strip():
                continue
            self.L.lammps_command(self.h, line.encode())
            if self.L.lammps_has_error(self.h):
                buf = ctypes.create_string_buffer(4096)
                self.L.lammps_get_last_error_message(self.h, buf, 4096)
                return buf.value.decode(errors="replace")
        return None

    def close(self):
        self.L.lammps_close(self.h)


def run_case(a, case, ref, trim=False, control="NULL"):
    lo, hi, tilt = (tuple(float(v) for v in t) for t in runner.cell_to_lammps(case["cell"]))
    els = case["elements"]
    types = [int(t) for t in ref["types"]]
    if trim:   # only the element types that actually have atoms (avoids the strict Q-12 rule on never-used element pairs)
        used = sorted(set(types)); els = [els[u - 1] for u in used]; types = [used.index(t) + 1 for t in types]
    skin = float(case.get("neighbor", {}).get("skin", 2.0))
    lg = "yes" if case.get("pair", {}).get("lgvdw") else "no"
    m = Lammps(a.lib)
    try:
        steps = ["units real\natom_style charge\natom_modify map array",
                 "boundary " + " ".join("p" if p else "f" for p in case["periodic"])]
        steps.append(f"region b prism {lo[0]!r} {hi[0]!r} {lo[1]!r} {hi[1]!r} {lo[2]!r} {hi[2]!r} {tilt[0]!r} {tilt[1]!r} {tilt[2]!r} units box"
                     if any(abs(t) > 0 for t in tilt) else f"region b block {lo[0]!r} {hi[0]!r} {lo[1]!r} {hi[1]!r} {lo[2]!r} {hi[2]!r} units box")
        steps.append(f"create_box {len(els)} b\nmass * 1.0")
        for p, t in zip(ref["positions"], types):
            steps.append(f"create_atoms {t} single {float(p[0])!r} {float(p[1])!r} {float(p[2])!r} units box")
        steps.append(f"plugin load {a.plugin}")
        steps.append(f"pair_style reaxff/metal {control} checkqeq no lgvdw {lg} reaxmetal_selfcheck yes")
        steps.append(f'pair_coeff * * "{Path(a.ffield_dir) / case["ffield"]["name"]}" ' + " ".join(els))
        steps.append(f"neighbor {skin!r} bin\nneigh_modify delay 0 every 1 check no\nthermo_style custom step pe\nrun 0")
        for s in steps:
            err = m.cmd(s)
            if err:
                return err
        return "run 0 completed without the expected refusal"
    finally:
        m.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--fixtures", default=str(HERE.parents[1] / "tests" / "fixtures"))
    a = ap.parse_args()
    fx = Path(a.fixtures)
    fails, ok_n, strict_n, nghost_total = [], 0, 0, 0
    for cf in sorted((fx / "cases").glob("*.json")):
        case = json.loads(cf.read_text()); ref = json.loads((fx / "reference" / cf.name).read_text())
        msg = run_case(a, case, ref)
        if "bond-parameter block" in msg:
            first = msg.splitlines()[0][:90]
            msg = run_case(a, case, ref, trim=True)
            strict_n += 1
            print(f"  {cf.stem}: all-elements type map rejected by strict Q-12 ({first}...); re-run with only the elements that have atoms")
        mm = re.search(r"A2 self-check (OK|FAILED): nlocal=(\d+) nghost=(\d+) rows=(\d+) lammps_entries=(\d+) engine_entries=(\d+) mismatched_rows=(\d+) vdw_oo=(\d+) vdw_og=(\d+) vdw_self=(\d+)", msg)
        if mm is None:
            fails.append(f"{cf.stem}: no self-check summary: {msg[:300]}"); continue
        status, nl, ng, rows, le, ee, mis, oo, og, sf = mm.groups()
        t = ref["tallies"]; want = (t.get("vdw.oo", 0), t.get("vdw.og", 0), t.get("vdw.self", 0))
        nghost_total += int(ng)
        if status != "OK" or int(mis) != 0 or le != ee:
            fails.append(f"{cf.stem}: self-check {status}: rows mismatched {mis}, lammps_entries {le} vs engine {ee}")
        elif (int(oo), int(og), int(sf)) != want:
            fails.append(f"{cf.stem}: engine pair classes {(oo, og, sf)} != M1 reference tallies {want}")
        elif "force backend is not implemented" not in msg:
            fails.append(f"{cf.stem}: compute() did not refuse after the self-check")
        else:
            ok_n += 1
    # C3, strict where the reference only warns: a ghost shell narrower than max(nonb_cut, hbond_cut, 2*bond_cut) is an error
    with tempfile.TemporaryDirectory() as td:
        ctl = Path(td) / "ctl.reaxff"; ctl.write_text("nbrhood_cutoff 8.0\n")
        case = json.loads((fx / "cases" / "cho_water_box_8.json").read_text()); ref = json.loads((fx / "reference" / "cho_water_box_8.json").read_text())
        msg = run_case(a, case, ref, control=str(ctl))
        if "ghost shell too narrow" not in msg:
            fails.append(f"narrow ghost shell was not rejected: {msg[:300]}")
        else:
            print("  C3: nbrhood_cutoff 8.0 (needs a 16 A ghost shell, LAMMPS provides 12 A) -> rejected as required")
    for f in fails:
        print("FAIL", f)
    print(f"INT-7: {ok_n} fixtures verified inside LAMMPS ({nghost_total} ghost atoms), {strict_n} needed the trimmed element map (strict Q-12), {len(fails)} failures")
    good = not fails and ok_n >= 58
    print("RESULT: PASS" if good else f"RESULT: FAIL ({len(fails)} failures, {ok_n} verified)")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
