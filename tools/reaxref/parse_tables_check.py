#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""PARSE-1: the tables built by reaxmetal's parser must equal the tables pinned LAMMPS stores, for every bundled force field.
For each file: run the INSTRUMENTED pinned LAMMPS (params.txt dump) and `reaxmetal_ffield_dump`, convert the first to canonical text
and compare character by character. Prints the SHA-256 of the portable and full canonical text (goldens for the CTest
`ffield_tables`; only hashes are committed, never the derived tables - ADR-017).
usage: parse_tables_check.py --lmp BIN --dump-tool BIN --ffield-dir DIR --workdir DIR [--write-hashes FILE]"""
import argparse, hashlib, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canonical_tables, runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ("lmp", "dump-tool", "ffield-dir", "workdir"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--write-hashes", default="")
    a = ap.parse_args()
    work = Path(a.workdir); work.mkdir(parents=True, exist_ok=True)
    libdir = Path(a.lmp).resolve().parents[1] / "lib"
    rows = []; bad = 0
    for ff in sorted(Path(a.ffield_dir).glob("ffield.reax.*")):
        lg = ff.name.endswith(".lg")
        el = runner.parse_ffield(ff)["order"][0]
        d = work / ff.name; d.mkdir(exist_ok=True)
        (d / "diag").mkdir(exist_ok=True)
        (d / "in.lmp").write_text(f"units real\natom_style charge\nregion b block 0 10 0 10 0 10\ncreate_box 1 b\nmass 1 1.0\n"
                                  f"pair_style reaxff NULL lgvdw {'yes' if lg else 'no'}\npair_coeff * * \"{ff}\" {el}\n")
        env = dict(os.environ, REAXMETAL_DIAG_DIR=str(d / "diag"), LD_LIBRARY_PATH=f"{libdir}:{os.environ.get('LD_LIBRARY_PATH', '')}")
        pr = subprocess.run([a.lmp, "-in", str(d / "in.lmp"), "-log", "none", "-screen", "none", "-nocite"], cwd=d, env=env, capture_output=True, text=True)
        if pr.returncode: print(f"{ff.name}: LAMMPS failed: {pr.stderr[-200:]}"); bad += 1; continue
        res = {}
        for mode, flag in (("portable", ["--portable"]), ("full", [])):
            ref = canonical_tables.convert(d / "diag" / "params.txt", portable=(mode == "portable"))
            ours = subprocess.run([a.dump_tool, *(["--lgvdw"] if lg else []), *flag, str(ff)], capture_output=True, text=True)
            if ours.returncode: print(f"{ff.name}: OUR PARSER REJECTED: {ours.stderr.strip()[:200]}"); bad += 1; break
            if ours.stdout != ref:
                bad += 1
                la, lb = ours.stdout.splitlines(), ref.splitlines()
                first = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
                print(f"{ff.name} [{mode}]: MISMATCH at line {first}: ours={la[first][:100] if first < len(la) else None!r} lammps={lb[first][:100] if first < len(lb) else None!r}")
            res[mode] = hashlib.sha256(ref.encode()).hexdigest()
        else:
            print(f"{ff.name:26s} {len(ref.splitlines()):6d} lines  identical (portable and full)  sha256 portable={res['portable'][:16]}.. full={res['full'][:16]}..")
            rows.append((ff.name, res["portable"], res["full"]))
    if a.write_hashes and not bad:
        Path(a.write_hashes).write_text("# name\tsha256(portable canonical dump)\tsha256(full canonical dump); source: instrumented pinned LAMMPS tables, tools/reaxref/parse_tables_check.py\n"
                                        + "".join(f"{n}\t{p}\t{f}\n" for n, p, f in rows))
    print("RESULT:", "ALL IDENTICAL" if not bad and len(rows) == 11 else f"{bad} problems, {len(rows)} identical")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
