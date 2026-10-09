#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Adapter A1 (milestone M2) versus stock `pair_style reaxff`, through the LAMMPS C library (ctypes):
  E1 extract(chi|eta|gamma) of `reaxff/metal` equals stock `reaxff` BIT FOR BIT for every bundled force field and a set of type maps
     (full file order, reversed subset, NULL entries, lower-case symbols, repeated elements);
  E2 the same inputs are accepted/rejected by both, except the documented strict rejection of element pairs without a bond block (Q-12);
  E3 host checks: no charge fix, newton off, no charge attribute, deferred charge models, unsupported options, bad control file;
  E4 with every check passed compute() runs (CPU-64 engine since M4);
  E5 (optional, --mpi-lmp/--mpi-plugin) a 2-rank run computes (multi-rank is supported since the MPI work).
usage: run_a1.py --lib <liblammps.so> --plugin <reaxmetaladapterplugin.so> --ffield-dir DIR"""
import argparse, ctypes, itertools, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "reaxref"))
import runner  # noqa: E402


class Lammps:
    def __init__(self, lib):
        self.L = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)   # plugins resolve LAMMPS symbols from the global scope
        self.L.lammps_open_no_mpi.restype = ctypes.c_void_p
        self.L.lammps_extract_pair.restype = ctypes.c_void_p
        argv = (ctypes.c_char_p * 6)(b"lmp", b"-log", b"none", b"-screen", b"none", b"-nocite")
        self.h = ctypes.c_void_p(self.L.lammps_open_no_mpi(6, argv, None))

    def cmd(self, c):
        """run one command; returns None on success or the error message"""
        for line in c.split("\n"):
            self.L.lammps_command(self.h, line.encode())
            if self.L.lammps_has_error(self.h): break
        if self.L.lammps_has_error(self.h):
            buf = ctypes.create_string_buffer(2048)
            self.L.lammps_get_last_error_message(self.h, buf, 2048)
            return buf.value.decode(errors="replace")
        return None

    def extract(self, name, ntypes):
        p = self.L.lammps_extract_pair(self.h, name.encode())
        if not p: return None
        return list((ctypes.c_double * (ntypes + 1)).from_address(p))

    def close(self):
        self.L.lammps_close(self.h)


def setup(lib, ntypes, plugin=None, atoms=False, atom_style="charge", newton=None):
    m = Lammps(lib)
    if newton: m.cmd(f"newton {newton}")
    m.cmd(f"units real\natom_style {atom_style}")
    m.cmd("region b block 0 10 0 10 0 10")
    m.cmd(f"create_box {ntypes} b")
    m.cmd("mass * 1.0")
    if atoms: m.cmd("create_atoms 1 single 5 5 5")
    if plugin:
        e = m.cmd(f"plugin load {plugin}")
        assert e is None, e
        # 'plugin load' reports dlopen failures on the screen only; make a missing registration fatal and explicit
        e = m.cmd("pair_style reaxff/metal NULL")
        assert e is None or "Unrecognized pair style" not in e, f"plugin did not register reaxff/metal (dlopen/ABI failure? see plugin/adapter/lammps_fmt_abi.h): {e}"
    return m


def listed_pairs(path):
    L = Path(path).read_text(errors="replace").splitlines()
    k = [i for i, l in enumerate(L) if "Nr of bonds" in l][0]; n = int(L[k].split()[0])
    return {frozenset((int(L[k + 2 + 2 * j].split()[0]), int(L[k + 2 + 2 * j].split()[1]))) for j in range(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--mpi-lmp", default=""); ap.add_argument("--mpi-plugin", default=""); ap.add_argument("--mpirun", default="mpirun.openmpi --allow-run-as-root --oversubscribe")
    a = ap.parse_args()
    fails = []; nexact = 0; nreject_both = 0; nstrict = 0

    # ---------------------------------------------------------------- E1 + E2
    for ff in sorted(Path(a.ffield_dir).glob("ffield.reax.*")):
        order = runner.parse_ffield(ff)["order"]
        lg = "yes" if ff.name.endswith(".lg") else "no"
        listed = listed_pairs(ff)
        maps = {"full": order, "reversed_subset": list(reversed(order[:3])), "with_NULL": [order[0], "NULL", order[-1]],
                "lowercase": [s.lower() for s in order[:2]], "repeated": [order[0], order[0], order[1]], "single": [order[1]],
                "unknown": [order[0], "Zz"]}
        for mname, els in maps.items():
            n = len(els)
            res = {}
            for tag, style, plug in (("stock", "reaxff", None), ("ours", "reaxff/metal", a.plugin)):
                m = setup(a.lib, n, plug)
                e1 = m.cmd(f"pair_style {style} NULL lgvdw {lg}")
                e2 = m.cmd(f'pair_coeff * * "{ff}" {" ".join(els)}') if e1 is None else e1
                ex = {k: m.extract(k, n) for k in ("chi", "eta", "gamma")} if e2 is None else None
                m.close()
                res[tag] = (e2, ex)
            (se, sx), (oe, ox) = res["stock"], res["ours"]
            idx = [order.index(e.upper()) + 1 for e in els if e != "NULL" and e.upper() in [o for o in order]]
            missing = [(x, y) for x, y in itertools.combinations_with_replacement(sorted(set(idx)), 2) if frozenset((x, y)) not in listed]
            if se is None and oe is None:
                if missing: fails.append(f"{ff.name}/{mname}: ours accepted element pairs without a bond block {missing} (Q-12 phantom bond)")
                elif sx == ox: nexact += 1
                else: fails.append(f"{ff.name}/{mname}: extract() differs: stock={sx} ours={ox}")
            elif se is None and oe is not None:
                if missing and "bond-parameter block" in oe: nstrict += 1     # Q-12: documented strict rejection
                else: fails.append(f"{ff.name}/{mname}: stock accepted but ours rejected: {oe[:120]}")
            elif se is not None and oe is None:
                fails.append(f"{ff.name}/{mname}: stock rejected ({se[:80]}) but ours accepted")
            else:
                nreject_both += 1
    print(f"E1/E2: extract() identical in {nexact} cases; both reject in {nreject_both}; documented Q-12 strict rejections {nstrict}; failures {len(fails)}")

    # ---------------------------------------------------------------- E3 / E4 host checks
    cho = Path(a.ffield_dir) / "ffield.reax.cho"
    def host(name, expect, style_args="NULL", fixes=("fix q all qeq/reaxff 1 0.0 10.0 1e-6 reaxff",), atom_style="charge", newton=None, atoms=True, pre=()):
        m = setup(a.lib, 3, a.plugin, atoms=atoms, atom_style=atom_style, newton=newton)
        err = m.cmd(f"pair_style reaxff/metal {style_args}")
        if err is None: err = m.cmd(f'pair_coeff * * "{cho}" C H O')
        for c in pre:
            if err is None: err = m.cmd(c)
        for f in fixes:
            if err is None: err = m.cmd(f)
        if err is None:
            m.cmd("thermo_style custom step pe"); err = m.cmd("run 0")
        m.close()
        ok = (err is None) if expect is None else (err is not None and expect in err)
        if not ok: fails.append(f"host check '{name}': expected an error containing {expect!r}, got {err!r}")
        return err
    host("no charge fix", "exactly one of the fix qeq/reaxff", fixes=())
    host("two charge fixes", "exactly one of the fix qeq/reaxff", fixes=("fix q1 all qeq/reaxff 1 0.0 10.0 1e-6 reaxff", "fix q2 all qeq/reaxff 1 0.0 10.0 1e-6 reaxff"))
    host("newton off", "requires newton pair on", newton="off")
    host("no q attribute", "requires atom attribute q", atom_style="atomic", fixes=())
    import tempfile
    gauss = os.path.join(tempfile.mkdtemp(prefix="a1gauss"), "gauss.txt")
    with open(gauss, "w") as gf: gf.write("1 0.5\n2 0.5\n3 0.5\n")
    host("deferred qtpie", "does not support", fixes=(f"fix q all qtpie/reaxff 1 0.0 10.0 1e-6 reaxff {gauss}",))
    host("tabulate deferred", "opt.tabulate", style_args="NULL tabulate 20", fixes=())
    host("unknown keyword", "unknown keyword", style_args="NULL frobnicate 1", fixes=())
    host("missing control file", "cannot open ReaxFF control file", style_args="/nonexistent/ctl", fixes=())
    # E4: all checks pass -> explicit refusal from compute
    # (since M4 compute() runs the CPU-64 engine: with every check passed the run completes without error)
    host("compute runs (cpu64)", None)
    host("checkqeq no needs no fix", None, style_args="NULL checkqeq no", fixes=())
    host("qeq/shielded accepted", None, fixes=("fix q all qeq/shielded 1 10.0 1e-6 100 reaxff",))
    host("backend invalid", "backend must be cpu64 or metal", style_args="NULL backend gpu")
    print(f"E3/E4 host checks done; failures so far {len(fails)}")

    # ---------------------------------------------------------------- E5 multi-rank
    if a.mpi_lmp:
        d = Path(a.ffield_dir).parent / "a1_mpi"; d.mkdir(exist_ok=True)
        (d / "in.lmp").write_text(f"units real\natom_style charge\nregion b block 0 10 0 10 0 10\ncreate_box 3 b\nmass * 1.0\ncreate_atoms 1 single 5 5 5\n"
                                  f"plugin load {a.mpi_plugin}\npair_style reaxff/metal NULL\npair_coeff * * \"{cho}\" C H O\nfix q all qeq/reaxff 1 0.0 10.0 1e-6 reaxff\nrun 0\n")
        env = dict(os.environ, LD_LIBRARY_PATH=str(Path(a.mpi_lmp).resolve().parents[1] / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        for np_ in (1, 2):
            pr = subprocess.run([*a.mpirun.split(), "-np", str(np_), a.mpi_lmp, "-in", str(d / "in.lmp"), "-log", "none", "-nocite"], cwd=d, env=env, capture_output=True, text=True)
            out = pr.stdout + pr.stderr
            want = "Loop time of"
            if want not in out: fails.append(f"MPI np={np_}: expected {want!r}; got {out[-200:]!r}")
        print("E5 multi-rank checked (np=1 and np=2 reach compute)")

    print("RESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
    for f in fails[:20]: print("  ", f)
    return 0 if not fails else 2


if __name__ == "__main__":
    sys.exit(main())
