#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Run every fixture case against one LAMMPS build and write <out>/<case>/result.json plus <out>/summary.json.
usage: run_suite.py [--launcher "mpirun -np N"] [--lmp-args "..."] --lmp BIN --label NAME --cases DIR --ffield-dir DIR --out DIR [--ids a,b] [--jobs N] [--no-diag]"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner  # noqa: E402


def one(args):
    case_path, lmp, label, ffdir, out, diag, launcher, lmp_args = args
    case = json.loads(Path(case_path).read_text())
    t0 = time.time()
    try:
        res = runner.run_case(case, lmp, ffdir, Path(out) / case["id"], label, diag=diag, launcher=launcher, lmp_args=lmp_args)
    except Exception as e:   # a harness failure is a result too: never swallowed
        res = {"case_id": case["id"], "valid": False, "invalid_reasons": [f"runner exception: {type(e).__name__}: {e}"]}
    res["wall_seconds"] = time.time() - t0
    (Path(out) / case["id"]).mkdir(parents=True, exist_ok=True)
    (Path(out) / case["id"] / "result.json").write_text(json.dumps(res, indent=1))
    return case["id"], res["valid"], res["invalid_reasons"], res["wall_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--cases", required=True); ap.add_argument("--ffield-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=""); ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--no-diag", action="store_true")
    ap.add_argument("--launcher", default="", help="command prefix, e.g. \"mpirun -np 2\"")
    ap.add_argument("--lmp-args", default="", help="extra LAMMPS command-line args, e.g. \"-k on t 1 -sf kk\"")
    a = ap.parse_args()
    files = sorted(Path(a.cases).glob("*.json"))
    if a.ids: files = [f for f in files if f.stem in set(a.ids.split(","))]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), a.lmp, a.label, a.ffield_dir, a.out, not a.no_diag, a.launcher.split(), a.lmp_args.split()) for f in files]
    summ = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for cid, ok, why, wall in ex.map(one, jobs):
            summ.append({"case": cid, "valid": ok, "reasons": why, "wall_s": round(wall, 2)})
            print(f"{'OK  ' if ok else 'FAIL'} {cid:32s} {wall:6.1f}s {'; '.join(why)[:160]}", flush=True)
    Path(a.out, "summary.json").write_text(json.dumps({"label": a.label, "lmp": a.lmp, "cases": summ}, indent=1))
    n_ok = sum(1 for s in summ if s["valid"])
    print(f"{n_ok}/{len(summ)} valid")
    return 0 if n_ok == len(summ) else 2


if __name__ == "__main__":
    sys.exit(main())
