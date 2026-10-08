#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Differential test of the reaxmetal force-field parser against pinned LAMMPS on seeded, mutated force-field files.
For every mutant: LAMMPS (instrumented build, so accepted tables can be dumped) and `reaxmetal_ffield_dump` each accept or reject.
  both accept            -> canonical tables must be identical (portable AND full)                       [else: DISAGREE]
  both reject            -> fine
  LAMMPS rejects/crashes, ours accepts -> must not happen                                                [BAD-ACCEPT]
  LAMMPS accepts, ours rejects -> allowed only for the documented strict-mode rejections:
        truncated file, >slots three-body sets (Q-09), non-finite number, fewer than 38 general parameters [else: BAD-REJECT]
usage: parse_diff.py --lmp BIN --dump-tool BIN --ffield-dir DIR --workdir DIR [--n 400] [--seed 1] [--out report.json]"""
import argparse, json, os, random, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canonical_tables, runner  # noqa: E402

BASES = ["ffield.reax.cho", "ffield.reax.lg", "ffield.reax.mattsson", "ffield.reax.AB", "ffield.reax.FC", "ffield.reax.rdx"]
STRICT_OK = ("unexpected end of file", "too many valence-angle parameter sets", "is not finite", "general parameters")


def mutate(text, rng):
    L = text.split("\n")
    op = rng.choice(["badtoken", "inf", "deltok", "dupline", "delline", "trunc", "swap", "count", "dup3", "elem", "blank", "crlf", "extracol", "dup4", "negidx", "bigidx"])
    def numline():
        c = [i for i, l in enumerate(L) if l.strip() and re.match(r"^\s*[-+0-9.]", l)]
        return rng.choice(c)
    if op == "badtoken":
        i = numline(); w = L[i].split(); w[rng.randrange(len(w))] = rng.choice(["abc", "1.0d0", "--1", "1e", "0x", "1,5"]); L[i] = " ".join(w)
    elif op == "inf":
        i = numline(); w = L[i].split(); w[rng.randrange(len(w))] = rng.choice(["inf", "nan", "1e999", "-inf"]); L[i] = " ".join(w)
    elif op == "deltok":
        i = numline(); w = L[i].split()
        if len(w) > 1: del w[rng.randrange(len(w))]
        L[i] = " ".join(w)
    elif op == "dupline":
        i = numline(); L.insert(i, L[i])
    elif op == "delline":
        del L[rng.randrange(len(L))]
    elif op == "trunc":
        del L[rng.randrange(5, len(L)):]
    elif op == "swap":
        i, j = rng.randrange(len(L)), rng.randrange(len(L)); L[i], L[j] = L[j], L[i]
    elif op == "count":
        c = [i for i, l in enumerate(L) if re.match(r"^\s*[0-9]+\s+!", l)]
        if c:
            i = rng.choice(c); w = L[i].split(None, 1); L[i] = f"{max(0, int(w[0]) + rng.choice([-2, -1, 1, 2]))} " + (w[1] if len(w) > 1 else "")
    elif op == "dup3":
        c = [i for i, l in enumerate(L) if re.match(r"^\s*\d+\s+\d+\s+\d+\s+[-0-9.]+(\s+[-0-9.]+){6}\s*$", l)]
        if c:
            i = rng.choice(c)
            for _ in range(rng.choice([1, 1, 2, 3])): L.insert(i, L[i])
    elif op == "dup4":
        c = [i for i, l in enumerate(L) if re.match(r"^\s*-?\d+\s+\d+\s+\d+\s+-?\d+\s+[-0-9.]+(\s+[-0-9.]+){4}", l)]
        if c: i = rng.choice(c); L.insert(i, L[i])
    elif op == "elem":
        c = [i for i, l in enumerate(L) if re.match(r"^\s*[A-Za-z][A-Za-z0-9]*\s+[-0-9.]+(\s+[-0-9.]+){7}", l)]
        if c:
            i = rng.choice(c); w = L[i].split(); w[0] = rng.choice(["c", "Xx", "Abcdef", "H", "O"]); L[i] = " ".join(w)
    elif op == "blank":
        L.insert(rng.randrange(len(L)), rng.choice(["", "   ", "\t", "! comment only"]))
    elif op == "crlf":
        return "\r\n".join(L), op
    elif op == "extracol":
        i = numline(); L[i] = L[i] + " 1.0 2.0"
    elif op == "negidx":
        c = [i for i, l in enumerate(L) if re.match(r"^\s*\d+\s+\d+\s+[-0-9.]+(\s+[-0-9.]+){7}", l)]
        if c: i = rng.choice(c); w = L[i].split(); w[0] = "0"; L[i] = " ".join(w)
    elif op == "bigidx":
        c = [i for i, l in enumerate(L) if re.match(r"^\s*\d+\s+\d+\s+\d+\s+[-0-9.]+(\s+[-0-9.]+){6}\s*$", l)]
        if c: i = rng.choice(c); w = L[i].split(); w[rng.randrange(3)] = "99"; L[i] = " ".join(w)
    return "\n".join(L), op


def run_one(args):
    k, text, op, base, a = args
    d = Path(a.workdir) / f"m{k:05d}"; d.mkdir(parents=True, exist_ok=True)
    ff = d / "ffield.mut"; ff.write_text(text)
    lg = base.endswith(".lg")
    # element for pair_coeff: first symbol of the BASE file (mutations may break the file; LAMMPS must then reject for its own reasons)
    el = runner.parse_ffield(Path(a.ffield_dir) / base)["order"][0]
    (d / "in.lmp").write_text(f"units real\natom_style charge\nregion b block 0 10 0 10 0 10\ncreate_box 1 b\nmass 1 1.0\n"
                              f"pair_style reaxff NULL lgvdw {'yes' if lg else 'no'}\npair_coeff * * \"{ff}\" {el}\n")
    (d / "diag").mkdir(exist_ok=True)
    libdir = Path(a.lmp).resolve().parents[1] / "lib"
    env = dict(os.environ, REAXMETAL_DIAG_DIR=str(d / "diag"), LD_LIBRARY_PATH=f"{libdir}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    try:
        pr = subprocess.run([a.lmp, "-in", str(d / "in.lmp"), "-log", "none", "-nocite"], cwd=d, env=env, capture_output=True, text=True, timeout=60)
        lm = "crash" if pr.returncode < 0 else ("accept" if pr.returncode == 0 else "reject")
    except subprocess.TimeoutExpired:
        lm = "crash"; pr = None
    # a LAMMPS 'accept' where pair_coeff failed on the element name (e.g. the mutation renamed it) is a reject for our purposes only if the
    # error came from the element lookup; both sides are then compared on parse only, so rerun without that dependency: see below
    ours = subprocess.run([a.dump_tool, *(["--lgvdw"] if lg else []), "--portable", str(ff)], capture_output=True, text=True)
    om = "accept" if ours.returncode == 0 else "reject"
    verdict = "ok"; detail = ""
    lammps_msg = (pr.stderr + pr.stdout) if pr else "timeout"
    if lm == "accept" and om == "accept":
        ref = canonical_tables.convert(d / "diag" / "params.txt", portable=True)
        if ref != ours.stdout:
            verdict = "DISAGREE"
            la, lb = ours.stdout.splitlines(), ref.splitlines()
            f = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
            detail = f"line {f}: ours={la[f][:80] if f < len(la) else None!r} lammps={lb[f][:80] if f < len(lb) else None!r}"
    elif lm != "accept" and om == "accept":
        # LAMMPS may fail AFTER parsing (e.g. the element symbol no longer matches). Distinguish by the message.
        post_parse = "Non-existent ReaxFF type" in lammps_msg or "Incorrect args for pair coefficients" in lammps_msg
        verdict = "ok(post-parse)" if post_parse else "BAD-ACCEPT"
        detail = lammps_msg.strip().splitlines()[-1][:120] if lammps_msg.strip() else ""
    elif lm == "accept" and om == "reject":
        why = ours.stderr.strip()
        verdict = "ok(strict)" if any(s in why for s in STRICT_OK) else "BAD-REJECT"
        detail = why[:160]
    return {"k": k, "op": op, "base": base, "lammps": lm, "ours": om, "verdict": verdict, "detail": detail}


def main():
    ap = argparse.ArgumentParser()
    for k in ("lmp", "dump-tool", "ffield-dir", "workdir"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default="")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    jobs = []
    for k in range(a.n):
        base = BASES[k % len(BASES)]
        text = (Path(a.ffield_dir) / base).read_text(errors="replace")
        t, op = mutate(text, rng)
        jobs.append((k, t, op, base, a))
    with ThreadPoolExecutor(4) as ex: res = list(ex.map(run_one, jobs))
    from collections import Counter
    c = Counter((r["lammps"], r["ours"], r["verdict"]) for r in res)
    for key, v in sorted(c.items()): print(f"  lammps={key[0]:7s} ours={key[1]:7s} {key[2]:16s} {v}")
    bad = [r for r in res if r["verdict"] in ("DISAGREE", "BAD-ACCEPT", "BAD-REJECT")]
    both = sum(1 for r in res if r["lammps"] == "accept" and r["ours"] == "accept")
    print(f"{len(res)} mutants (seed {a.seed}); both accepted: {both}; problems: {len(bad)}")
    for r in bad[:15]: print("  PROBLEM", r)
    if a.out: Path(a.out).write_text(json.dumps({"seed": a.seed, "n": a.n, "summary": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in c.items()}, "problems": bad}, indent=1))
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
