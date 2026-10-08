#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""M0.5 integration probe harness (stdlib only). Runs the pinned stock LAMMPS with the reaxff/metal probe plugin.

  python3 -I run_probe.py --lmp <bin/lmp> --plugin <reaxmetalprobeplugin.so> --potentials <lammps>/potentials --work <dir>

Prints a machine-readable summary; exit status != 0 only if a *stated expectation* fails. It records facts, so
informational lines (e.g. multiplicity) are printed, not asserted, unless a contract rule depends on them.
"""
import argparse, os, re, subprocess, sys

def run(lmp, infile, work, env_extra, cwd):
    env = dict(os.environ); env.update(env_extra)
    cmd = [lmp, "-in", infile, "-log", "none", "-nocite"]
    for k, v in env_extra.items():
        if k.isupper() and k not in ("LAMMPS_POTENTIALS",):
            cmd += ["-var", k, v]
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr

def read_charges(path):
    q = {}
    with open(path) as f:
        lines = f.read().splitlines()
    i = [n for n, l in enumerate(lines) if l.startswith("ITEM: ATOMS")][0]
    for l in lines[i + 1:]:
        a = l.split()
        if len(a) >= 3: q[int(a[0])] = float(a[2])
    return q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True)
    ap.add_argument("--potentials", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--inputs", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    for f in os.listdir(a.inputs):
        if f.startswith("in."):
            open(os.path.join(a.work, f), "w").write(open(os.path.join(a.inputs, f)).read())
    base = {"LAMMPS_POTENTIALS": a.potentials, "PLUGIN_SO": os.path.abspath(a.plugin)}
    fails = []

    # 1. contract probe for each neighbor-request mode
    for mode in ("none", "half", "halfoff", "full", "fullghost", "halfoffghost"):
        rc, out = run(a.lmp, "in.contract_probe", a.work, dict(base, NEIGH=mode), a.work)
        pr = [l for l in out.splitlines() if l.startswith("PROBE ")]
        res = re.search(r"RESULT fx1=(\S+) fxsum=(\S+)", out)
        print("== neigh=%s rc=%d" % (mode, rc))
        if rc != 0:
            err = [l for l in out.splitlines() if "ERROR" in l]
            print("   FAILED:", err[:2]); fails.append("contract %s rc=%d" % (mode, rc)); continue
        print("   " + (pr[0] if pr else "NO PROBE LINE"))
        print("   fx[1] after reverse comm =", res.group(1) if res else None)
        m = re.search(r"nlocal=(\d+) nghost=(\d+).*mult_tag1=(\d+)", pr[0]) if pr else None
        if res and m:
            nall = int(m.group(1)) + int(m.group(2))
            ok1 = abs(float(res.group(2)) - nall) < 1e-9          # every atom incl. ghosts got +1 -> total must equal nall
            ok2 = abs(float(res.group(1)) - int(m.group(3))) < 1e-9  # owner force = number of copies of tag 1
            print("   fold-back conservation: sum fx=%s vs nall=%d -> %s ; fx[1] vs copies(tag1)=%s -> %s" % (res.group(2), nall, "OK" if ok1 else "FAIL", m.group(3), "OK" if ok2 else "FAIL"))
            if not (ok1 and ok2): fails.append("foldback %s" % mode)
    # 1b. MD: ghost data over steps / reneighbouring, with and without a pair neighbor request
    for mode in ("none", "fullghost"):
        rc, out = run(a.lmp, "in.contract_md", a.work, dict(base, NEIGH=mode, SORTFREQ="5"), a.work)
        rows = [re.search(r"step=(\d+) ago=(\d+).*nlocal=(\d+) nghost=(\d+).*ghost_not_lattice_shift=(\d+).*order_hash=(\w+)", l) for l in out.splitlines() if l.startswith("PROBE ")]
        rows = [r.groups() for r in rows if r]
        if rc != 0 or not rows:
            print("== md neigh=%s FAILED rc=%d" % (mode, rc)); fails.append("md " + mode); continue
        ghosts = sorted(set(int(r[3]) for r in rows)); rebuilds = [int(r[0]) for r in rows if r[1] == "0"]
        bad = sum(int(r[4]) for r in rows)
        # every step must have ghost == owner + lattice shift (positions refreshed each step by LAMMPS forward comm)
        # topology-epoch rule: owned-atom order and nghost may change ONLY at steps where ago == 0
        viol = 0; reorder_at_rebuild = 0
        for prev, cur in zip(rows, rows[1:]):
            if cur[1] != "0" and (cur[5] != prev[5] or cur[3] != prev[3]): viol += 1
            if cur[1] == "0" and cur[5] != prev[5]: reorder_at_rebuild += 1
        print("   topology-epoch violations (order/nghost changed while ago>0): %d ; owned-order changes at rebuilds: %d" % (viol, reorder_at_rebuild))
        if viol: fails.append("epoch " + mode)
        ok = bad == 0 and len(rows) == 61 and viol == 0
        print("== md neigh=%s: %d compute() calls, rebuild(ago==0) steps=%s, distinct nghost values=%d, ghost-not-lattice-shift total=%d -> %s" % (mode, len(rows), rebuilds, len(ghosts), bad, "OK" if ok else "FAIL"))
        if not ok: fails.append("md " + mode)
    rc, out = run(a.lmp, "in.contract_min", a.work, dict(base, NEIGH="none"), a.work)
    res = re.search(r"RESULT e1=(\S+) e14=(\S+)", out); npro = len([l for l in out.splitlines() if l.startswith("PROBE ")])
    ok = rc == 0 and res and float(res.group(1)) == 1.0 and float(res.group(2)) == 14.0 and npro >= 2
    print("== minimize + compute pair reaxff/metal: rc=%d compute() calls=%d pvector[0]=%s pvector[13]=%s -> %s" % (rc, npro, res.group(1) if res else None, res.group(2) if res else None, "OK" if ok else "FAIL"))
    if not ok: fails.append("min/pvector"); print(out[-800:])
    # 1c. negative paths must be explicit errors
    for infile, needle in (("in.neg_newton_off", "newton pair on"), ("in.neg_no_charge", "attribute q"),
                           ("in.neg_unsupported_fix", None), ("in.neg_no_qeq_fix_probe", None)):
        rc, out = run(a.lmp, infile, a.work, dict(base), a.work)
        err = [l for l in out.splitlines() if "ERROR" in l]
        ok = rc != 0 and bool(err) and (needle is None or needle in err[0])
        print("== negative %s: rc=%d error=%s -> %s" % (infile, rc, err[0][:150] if err else None, "OK (explicit error)" if ok else "UNEXPECTED"))
        if needle is not None and not ok: fails.append(infile)
    # 2. charge equilibration: stock pair reaxff vs probe pair, fix qeq/reaxff and qeq/shielded
    rc, out = run(a.lmp, "in.qeq_stock", a.work, dict(base, OUT="q_stock.dump"), a.work)
    if rc != 0:
        print("stock run FAILED"); print(out[-1500:]); return 2
    qs = read_charges(os.path.join(a.work, "q_stock.dump"))
    for fixname, mode in (("qeq/reaxff", "none"), ("qeq/reaxff", "halfoff"), ("qeq/reaxff", "halfoffghost"),
                          ("qeq/reaxff", "full"), ("qeq/shielded", "full")):
        fx = fixname if fixname != "qeq/shielded" else "qeq/shielded"
        env = dict(base, NEIGH=mode, QEQFIX=fx, OUT="q_%s_%s.dump" % (fx.replace("/", "_"), mode))
        rc, out = run(a.lmp, "in.qeq_probe", a.work, env, a.work)
        if fx == "qeq/shielded":   # shielded takes (cutoff tol maxiter) not (swa swb tol)
            txt = open(os.path.join(a.work, "in.qeq_probe")).read().replace("${QEQFIX} 1 0.0 8.0 1.0e-20 reaxff", "qeq/shielded 1 8.0 1.0e-20 500 reaxff")
            open(os.path.join(a.work, "in.qeq_probe_sh"), "w").write(txt)
            rc, out = run(a.lmp, "in.qeq_probe_sh", a.work, env, a.work)
        label = "%s + probe(neigh=%s)" % (fx, mode)
        if rc != 0:
            err = [l for l in out.splitlines() if "ERROR" in l]
            print("== %s: FAILED %s" % (label, err[:2])); fails.append(label); continue
        qp = read_charges(os.path.join(a.work, env["OUT"]))
        dmax = max(abs(qp[i] - qs[i]) for i in qs)
        qmax = max(abs(v) for v in qs.values())
        print("== %s: max|dq| vs stock reaxff+qeq/reaxff = %.3e (max|q|=%.3e, N=%d, sum q=%.3e)" % (label, dmax, qmax, len(qs), sum(qp.values())))
    ex = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "eem_dense_check.py")
    if os.path.exists(ex):
        r = subprocess.run([sys.executable, "-I", ex, os.path.join(a.work, "q_stock.dump"), os.path.join(a.potentials, "ffield.reax.mattsson"), "8.0", "3.77", "2"], capture_output=True, text=True)
        print("== dense explicit-image EEM vs stock fix qeq/reaxff (informational):"); print("   " + r.stdout.strip().replace("\n", "\n   ") + (r.stderr[-300:] if r.returncode else ""))
    print("SUMMARY failures=%d %s" % (len(fails), fails))
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
