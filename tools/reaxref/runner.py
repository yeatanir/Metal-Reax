#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""ReaxMetal M1 reference runner.

Runs ONE case through the pinned LAMMPS (stock or instrumented build) and writes a machine-readable result.json with
total energy, the 14 pvector slots, per-atom charges and forces, optional instrumented diagnostics, QEq convergence
facts, and full provenance. A run is marked invalid (never silently accepted) if LAMMPS warned, QEq did not reach its
tolerance, charges are not neutral, or the independent equalization residual exceeds the case threshold.

Case spec (JSON), all lengths in Angstrom, energies kcal/mol, charges in e:
  id, description
  ffield:   {"name": "ffield.reax.cho"}            resolved through --ffield-dir; sha256 recorded (and checked if given)
  elements: ["C","H"]                              LAMMPS type i+1 <-> pair_coeff element i (ffield symbol)
  cell:     {"lo":[x,y,z],"hi":[x,y,z],"tilt":[xy,xz,yz]}   (tilt optional)   or   {"vectors":[[ax,ay,az],[bx,by,bz],[cx,cy,cz]],"origin":[..]}
  periodic: [true,true,true]
  atoms:    [{"el":"C","xyz":[x,y,z],"q":0.0}, ...]   ids are 1..N in list order
  charge:   {"model":"qeq/reaxff","swb":10.0,"tolerance":1e-12,"maxiter":500}   |   {"model":"fixed"}
  pair:     {"checkqeq":"yes|no","lgvdw":false,"enobonds":true,"safezone":..,"mincap":..,"minhbonds":..}   (all optional)
  neighbor: {"skin":2.0}  (optional)
"""
import argparse, hashlib, json, os, platform, re, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

SLOT_NAMES = ["eb", "ea", "elp", "emol", "ev", "epen", "ecoa", "ehb", "et", "eco", "ew", "ep", "efi", "eqeq"]
ENERGY_FIELDS = ["e_bond", "e_ov", "e_un", "e_lp", "e_ang", "e_pen", "e_coa", "e_hb", "e_tor", "e_con",
                 "e_vdW", "e_ele", "e_pol"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


# ------------------------------------------------------------------------------------------------ cell handling
def cell_to_lammps(cell):
    """Return (lo, hi, tilt) for LAMMPS restricted-triclinic form. `vectors` must already be in the restricted form
    (a along +x, b in the xy plane, c with positive z): we refuse anything else instead of silently rotating atoms."""
    if "lo" in cell:
        lo = np.array(cell["lo"], float); hi = np.array(cell["hi"], float)
        tilt = np.array(cell.get("tilt", [0.0, 0.0, 0.0]), float)
        return lo, hi, tilt
    v = np.array(cell["vectors"], float)
    o = np.array(cell.get("origin", [0.0, 0.0, 0.0]), float)
    if abs(v[0, 1]) > 1e-12 or abs(v[0, 2]) > 1e-12 or abs(v[1, 2]) > 1e-12 or v[0, 0] <= 0 or v[1, 1] <= 0 or v[2, 2] <= 0:
        raise ValueError("cell vectors are not in LAMMPS restricted triclinic form (a=(ax,0,0), b=(bx,by,0), c=(cx,cy,cz>0))")
    lo = o; hi = o + np.array([v[0, 0], v[1, 1], v[2, 2]])
    return lo, hi, np.array([v[1, 0], v[2, 0], v[2, 1]])


def cell_matrix(cell):
    lo, hi, t = cell_to_lammps(cell)
    L = hi - lo
    return np.array([[L[0], 0, 0], [t[0], L[1], 0], [t[1], t[2], L[2]]])  # rows = a, b, c


# ------------------------------------------------------------------------------------------------ LAMMPS inputs
def write_data(case, ffparams, path):
    lo, hi, tilt = cell_to_lammps(case["cell"])
    els = case["elements"]
    atoms = case["atoms"]
    L = []
    L.append("LAMMPS data file written by reaxmetal reference runner\n")
    L.append(f"{len(atoms)} atoms\n{len(els)} atom types\n")
    for d, k in enumerate("xyz"):
        L.append(f"{lo[d]:.17g} {hi[d]:.17g} {k}lo {k}hi\n")
    if np.any(tilt != 0.0):
        L.append(f"{tilt[0]:.17g} {tilt[1]:.17g} {tilt[2]:.17g} xy xz yz\n")
    L.append("\nMasses\n\n")
    for i, e in enumerate(els):
        L.append(f"{i + 1} {ffparams['mass'][e.upper()]:.17g}\n")
    L.append("\nAtoms # charge\n\n")
    for n, a in enumerate(atoms):
        t = els.index(a["el"]) + 1
        x = a["xyz"]
        L.append(f"{n + 1} {t} {a.get('q', 0.0):.17g} {x[0]:.17g} {x[1]:.17g} {x[2]:.17g}\n")
    Path(path).write_text("".join(L))


def build_input(case, ffield_path, data_path, dump_path, extra_pair_args=""):
    pr = case.get("pair", {})
    ch = case.get("charge", {"model": "qeq/reaxff", "swb": 10.0, "tolerance": 1e-12, "maxiter": 500})
    per = case.get("periodic", [True, True, True])
    bnd = " ".join("p" if p else "f" for p in per)
    checkqeq = pr.get("checkqeq", "yes" if ch["model"] != "fixed" else "no")
    pair_args = f"checkqeq {checkqeq}"
    if "lgvdw" in pr: pair_args += f" lgvdw {'yes' if pr['lgvdw'] else 'no'}"
    if "enobonds" in pr: pair_args += f" enobonds {'yes' if pr['enobonds'] else 'no'}"
    for k in ("safezone", "mincap", "minhbonds"):
        if k in pr: pair_args += f" {k} {pr[k]}"
    pair_args += (" " + extra_pair_args) if extra_pair_args else ""
    skin = case.get("neighbor", {}).get("skin", 2.0)
    s = []
    s.append("units real\natom_style charge\natom_modify map array\n")
    s.append(f"boundary {bnd}\n")
    s.append(f'read_data "{data_path}"\n')
    s.append(f"pair_style reaxff NULL {pair_args}\n")
    s.append(f'pair_coeff * * "{ffield_path}" {" ".join(case["elements"])}\n')
    s.append(f"neighbor {skin} bin\nneigh_modify delay 0 every 1 check no\n")
    for cmd in case.get("extra_commands", []):   # experiment hooks (e.g. comm_modify cutoff ...); recorded in the case spec
        s.append(cmd + "\n")
    if ch["model"] == "qeq/reaxff":
        s.append(f"fix q all qeq/reaxff 1 0.0 {ch['swb']:.17g} {ch['tolerance']:.17g} reaxff maxiter {ch['maxiter']}\n")
    elif ch["model"] != "fixed":
        raise ValueError("unsupported charge model " + ch["model"])
    s.append("fix integ all nve\ncompute pp all pair reaxff\n")
    pv = " ".join(f"c_pp[{i + 1}]" for i in range(14))
    s.append(f"thermo_style custom step pe {pv}\nthermo_modify format float %.17g norm no\nthermo 1\n")
    s.append("run 0\n")
    s.append(f"write_dump all custom \"{dump_path}\" id type q x y z fx fy fz modify sort id "
             f"format line \"%d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g\"\n")
    return "".join(s)


# ------------------------------------------------------------------------------------------------ ffield (independent parse)
def parse_ffield(path):
    """Independent parser for the parts the checks need (masses, chi, eta, gamma, symbol order). Layout from the file
    format, not from LAMMPS source: header, general parameters, atom blocks (4 lines, 5 with lgvdw)."""
    L = Path(path).read_text(errors="replace").splitlines()
    i = 1
    ng = int(L[i].split()[0]); i += 1 + ng
    nat = int(L[i].split()[0]); i += 4
    # lgvdw adds a fifth line per atom; detect by checking whether line i+4 starts a new element or a bond-count line
    def is_new_atom(line):
        w = line.split()
        try:
            float(w[0]); return False
        except ValueError:
            return True
    lines_per = 4
    if nat > 1 and not is_new_atom(L[i + 4]): lines_per = 5
    out = {"mass": {}, "chi": {}, "eta": {}, "gamma": {}, "order": []}
    for a in range(nat):
        b = L[i + a * lines_per:i + (a + 1) * lines_per]
        w0, w1 = b[0].split(), b[1].split()
        sym = w0[0].upper()
        out["order"].append(sym)
        out["mass"][sym] = float(w0[3]); out["gamma"][sym] = float(w0[6])
        out["chi"][sym] = float(w1[5]); out["eta"][sym] = float(w1[6])
    return out


# ------------------------------------------------------------------------------------------------ diag readers
def read_params_dump(path):
    P = {"S": {}, "T": {}, "T2": {}, "G": None, "H3": {}, "HB": {}, "H4": {}, "S2": {}, "meta": ""}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"): P["meta"] = line; continue
        w = line.split()
        k = w[0]
        if k == "G": P["G"] = [float(x) for x in w[1:]]
        elif k == "S": P["S"][int(w[1])] = w[2:]
        elif k == "S2": P["S2"][int(w[1])] = [float(x) for x in w[2:]]
        elif k == "T": P["T"][(int(w[1]), int(w[2]))] = [float(x) for x in w[3:]]
        elif k == "T2": P["T2"][(int(w[1]), int(w[2]))] = [float(x) for x in w[3:]]
        elif k == "H3": P["H3"][tuple(int(x) for x in w[1:4])] = w[4:]
        elif k == "HB": P["HB"][tuple(int(x) for x in w[1:4])] = [float(x) for x in w[4:]]
        elif k == "H4": P["H4"][tuple(int(x) for x in w[1:5])] = w[5:]
    return P


def read_call_dump(path):
    D = {"atoms": [], "vars": [], "bonds": [], "forces": [], "tallies": {}, "meta": {}, "control": {}, "energy": {}, "qeq": None}
    def kv(line):
        return {a.split("=")[0]: float(a.split("=")[1]) for a in line.split()[1:] if "=" in a}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#meta"): D["meta"] = kv(line)
        elif line.startswith("#control"): D["control"] = kv(line)
        elif line.startswith("#energy"): D["energy"] = kv(line)
        elif line.startswith("#qeq"):
            D["qeq"] = None if line.split()[1] == "none" else kv(line)
        elif line.startswith("A "): D["atoms"].append(line.split()[1:])
        elif line.startswith("V "): D["vars"].append([float(x) for x in line.split()[1:]])
        elif line.startswith("B "): D["bonds"].append(line.split()[1:])
        elif line.startswith("F "): D["forces"].append([float(x) for x in line.split()[2:]])
        elif line.startswith("T "):
            _, key, cnt, e = line.split()
            D["tallies"][key] = {"count": int(cnt), "energy": float(e)}
    return D


# ------------------------------------------------------------------------------------------------ independent QEq check
def taper_coef(swb):
    x = lambda r: r / swb
    return lambda r: 1 - 35 * x(r) ** 4 + 84 * x(r) ** 5 - 70 * x(r) ** 6 + 20 * x(r) ** 7


def image_shifts(cell_mat, periodic, rcut):
    """All integer lattice shifts n with possible |n.cell| <= rcut + cell diameter slack (conservative)."""
    inv = np.linalg.inv(cell_mat.T)
    # perpendicular heights: height_k = volume / |cross of other two|
    vol = abs(np.linalg.det(cell_mat))
    a, b, c = cell_mat
    h = [vol / np.linalg.norm(np.cross(b, c)), vol / np.linalg.norm(np.cross(c, a)), vol / np.linalg.norm(np.cross(a, b))]
    rng = [int(np.ceil(rcut / h[k])) + 1 if periodic[k] else 0 for k in range(3)]
    return [(i, j, k) for i in range(-rng[0], rng[0] + 1) for j in range(-rng[1], rng[1] + 1) for k in range(-rng[2], rng[2] + 1)]


def eem_residual(x, types, chi, eta, gamma, q, cell_mat, periodic, swb, ke=14.4):
    """Independent check that q is the neutral-constrained EEM solution with explicit periodic images.
    Model (ENGINE_SPEC 7): A_ii = eta_i + sum_{n!=0} H(|nL|); A_ij = sum_n H(|xj + nL - xi|); minimise chi.q + q.A.q/2, sum q = 0.
    Returns (max |(Aq+chi) - mu|, mu, sum q) -- the equalization residual in eV (chi, eta are in eV)."""
    n = len(x)
    tap = taper_coef(swb)
    A = np.zeros((n, n))
    shifts = image_shifts(cell_mat, periodic, swb)
    sh = np.array(shifts, float) @ cell_mat
    for i in range(n):
        A[i, i] += eta[i]
        for j in range(n):
            shield = (gamma[i] * gamma[j]) ** -1.5
            d = x[j] + sh - x[i]
            r = np.sqrt((d * d).sum(axis=1))
            m = (r <= swb) & (r > 0.0)
            rr = r[m]
            A[i, j] += (ke * tap(rr) / (rr ** 3 + shield) ** (1.0 / 3.0)).sum()
    g = A @ q + chi
    mu = g.mean()
    return float(np.abs(g - mu).max()), float(mu), float(q.sum())


# ------------------------------------------------------------------------------------------------ run
def parse_thermo(logtext):
    lines = logtext.splitlines()
    for k, l in enumerate(lines):
        if l.strip().startswith("Step") and "PotEng" in l:
            hdr = l.split()
            vals = lines[k + 1].split()
            return dict(zip(hdr, [float(v) for v in vals]))
    return None


def parse_dump(path, n):
    rows = []
    started = False
    for line in Path(path).read_text().splitlines():
        if line.startswith("ITEM: ATOMS"): started = True; continue
        if started and line.strip(): rows.append([float(v) for v in line.split()])
    a = np.array(rows)
    assert a.shape == (n, 9), a.shape
    return a


def run_case(case, lmp_bin, ffield_dir, workdir, build_label, diag=True, env_extra=None, timeout=3600, extra_pair_args="", launcher=None, lmp_args=None):
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    ffname = case["ffield"]["name"]
    ffpath = Path(ffield_dir) / ffname
    ffsha = sha256_file(ffpath)
    if "sha256" in case["ffield"] and case["ffield"]["sha256"] != ffsha:
        raise RuntimeError(f"force field hash mismatch for {ffname}: {ffsha}")
    ffp = parse_ffield(ffpath)
    case_bytes = canonical_json(case)
    data = workdir / "data.lmp"; inp = workdir / "in.lmp"; dump = workdir / "state.dump"
    write_data(case, ffp, data)
    inp.write_text(build_input(case, ffpath, data, dump, extra_pair_args))
    diagdir = workdir / "diag"
    env = dict(os.environ)
    env.pop("OMP_NUM_THREADS", None)
    env["OMP_NUM_THREADS"] = "1"
    if diag:
        diagdir.mkdir(exist_ok=True)
        for f in diagdir.glob("*"): f.unlink()
        env["REAXMETAL_DIAG_DIR"] = str(diagdir)
    else:
        env.pop("REAXMETAL_DIAG_DIR", None)
    if env_extra: env.update(env_extra)
    # lmp needs the shared library on its path when installed outside the system
    libdir = Path(lmp_bin).resolve().parents[1] / "lib"
    env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = list(launcher or []) + [str(lmp_bin)] + list(lmp_args or []) + ["-in", str(inp), "-log", str(workdir / "log.lammps"), "-screen", "none", "-nocite"]
    pr = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
    log = (workdir / "log.lammps").read_text() if (workdir / "log.lammps").exists() else ""
    result = {"schema": "reaxmetal.reference_result/1", "case_id": case["id"], "case_sha256": hashlib.sha256(case_bytes).hexdigest(),
              "valid": False, "invalid_reasons": [], "returncode": pr.returncode}
    result["provenance"] = {
        "lammps_binary": str(lmp_bin), "lammps_binary_sha256": sha256_file(lmp_bin), "build_label": build_label,
        "lammps_pinned_commit": "8de817dd79bfe4525d5d39246a212d833e6dee07",
        "ffield": {"name": ffname, "sha256": ffsha}, "command": cmd, "diag_enabled": diag,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "launcher": list(launcher or []), "lammps_args": list(lmp_args or []),
    }
    all_warns = [l for l in log.splitlines() if l.startswith("WARNING") or "ERROR" in l]
    # A warning is accepted only if the case declares a matching regex in `expected_warnings` (and every declared
    # pattern must actually occur). Anything else invalidates the run.
    pats = [re.compile(p_) for p_ in case.get("expected_warnings", [])]
    warns = [w for w in all_warns if not any(p_.search(w) for p_ in pats)]
    result["lammps_warnings"] = warns
    result["lammps_warnings_expected"] = [w for w in all_warns if w not in warns]
    for p_ in pats:
        if not any(p_.search(w) for w in all_warns):
            result["invalid_reasons"].append(f"expected warning not seen: {p_.pattern}")
    if pr.returncode != 0:
        result["invalid_reasons"].append(f"lammps exited with {pr.returncode}: {(pr.stderr or pr.stdout)[-400:]}")
        return result
    if warns: result["invalid_reasons"].append("LAMMPS emitted warnings: " + " | ".join(warns))
    th = parse_thermo(log)
    n = len(case["atoms"])
    a = parse_dump(dump, n)
    pv = [th[f"c_pp[{i + 1}]"] for i in range(14)]
    result["energy"] = {"total_pe": th["PotEng"], "slots": dict(zip(SLOT_NAMES, pv)), "slot_sum": float(sum(pv))}
    result["charges"] = a[:, 2].tolist()
    result["forces"] = a[:, 6:9].tolist()
    result["positions"] = a[:, 3:6].tolist()
    result["types"] = [int(t) for t in a[:, 1]]
    result["sum_q"] = float(a[:, 2].sum())
    result["force_sum"] = [float(v) for v in a[:, 6:9].sum(axis=0)]
    ch = case.get("charge", {"model": "qeq/reaxff", "swb": 10.0, "tolerance": 1e-12, "maxiter": 500})
    result["charge_settings"] = ch
    if diag:
        calls = sorted(diagdir.glob("call_*.txt"))
        if len(calls) != 1:
            result["invalid_reasons"].append(f"expected exactly 1 Compute_Forces call, saw {len(calls)}")
        if calls:
            D = read_call_dump(calls[-1])
            result["diag"] = {"file": str(calls[-1]),
                              "energy_fields": {k: D["energy"].get(k) for k in ENERGY_FIELDS},
                              "tallies": D["tallies"], "n": int(D["meta"]["n"]), "N": int(D["meta"]["N"]),
                              "n_bonds_directed": len(D["bonds"]), "control": D["control"], "qeq": D["qeq"]}
            q = D["qeq"]
            if ch["model"] == "qeq/reaxff":
                if q is None:
                    result["invalid_reasons"].append("QEq diagnostics missing")
                else:
                    ok = q["relf_s"] <= ch["tolerance"] and q["relf_t"] <= ch["tolerance"]
                    result["qeq"] = {"imax": int(q["imax"]), "tolerance": q["tolerance"], "iters_s": int(q["iters_s"]),
                                     "iters_t": int(q["iters_t"]), "final_rel_residual_s": q["relf_s"],
                                     "final_rel_residual_t": q["relf_t"], "converged_to_tolerance": bool(ok)}
                    if not ok: result["invalid_reasons"].append(
                        f"QEq did not reach tolerance {ch['tolerance']:g}: final rel residual s={q['relf_s']:.3e} t={q['relf_t']:.3e}")
            # energy bookkeeping: pvector must equal the diag copy of energy_data
            ef = result["diag"]["energy_fields"]
            slots = [ef["e_bond"], ef["e_ov"] + ef["e_un"], ef["e_lp"], 0.0, ef["e_ang"], ef["e_pen"], ef["e_coa"], ef["e_hb"],
                     ef["e_tor"], ef["e_con"], ef["e_vdW"], ef["e_ele"], 0.0, ef["e_pol"]]
            if slots != pv: result["invalid_reasons"].append("pvector != energy_data copy (instrumentation mismatch)")
        # parameter dump + independent parse cross-check
        if (diagdir / "params.txt").exists():
            P = read_params_dump(diagdir / "params.txt")
            result["param_table_check"] = param_parse_check(P, ffp, case["elements"])
        # independent EEM residual
        if ch["model"] == "qeq/reaxff":
            els = [case["elements"][t - 1].upper() for t in result["types"]]
            x = np.array(result["positions"])
            chi = np.array([ffp["chi"][e] for e in els]); gam = np.array([ffp["gamma"][e] for e in els])
            # eta as stored by LAMMPS is 2x the file column (dense-EEM check, M0.5); use that and report the residual
            eta = np.array([2.0 * ffp["eta"][e] for e in els])
            res, mu, sq = eem_residual(x, None, chi, eta, gam, np.array(result["charges"]), cell_matrix(case["cell"]),
                                       case.get("periodic", [True, True, True]), ch["swb"])
            result["independent_eem"] = {"equalization_residual_eV": res, "mu_eV": mu, "sum_q": sq}
            thr = case.get("eem_residual_threshold_eV", 1e-8)
            if res > thr: result["invalid_reasons"].append(f"independent EEM equalization residual {res:.3e} eV > {thr:g}")
    if abs(result["sum_q"]) > case.get("neutrality_threshold", 1e-9) and ch["model"] == "qeq/reaxff":
        result["invalid_reasons"].append(f"sum q = {result['sum_q']:.3e}")
    result["valid"] = not result["invalid_reasons"]
    return result


def ffield_indices(P, elements):
    """LAMMPS type i+1 -> index into the parameter tables LAMMPS stored (= position of the element in the force field file)."""
    names = {P["S"][i][0].upper(): i for i in P["S"]}
    return [names[e.upper()] for e in elements]


def param_parse_check(P, ffp, elements):
    """Cross-check the independent ffield parser against the parameter table LAMMPS actually stored."""
    mism = []
    idx = ffield_indices(P, elements)
    for t, e in enumerate(elements):
        s = P["S"][idx[t]]
        name = s[0].upper()
        mass, chi, eta, gamma = float(s[3]), float(s[15]), float(s[16]), float(s[6])
        e = e.upper()
        if name != e: mism.append(f"type {t} name {name} != {e}")
        if mass != ffp["mass"][e]: mism.append(f"{e} mass {mass} != {ffp['mass'][e]}")
        if chi != ffp["chi"][e]: mism.append(f"{e} chi {chi} != {ffp['chi'][e]}")
        if eta != 2.0 * ffp["eta"][e]: mism.append(f"{e} eta stored {eta} != 2*{ffp['eta'][e]}")
        if gamma != ffp["gamma"][e]: mism.append(f"{e} gamma {gamma} != {ffp['gamma'][e]}")
    return {"ok": not mism, "mismatches": mism}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case"); ap.add_argument("--lmp", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--build-label", default="unlabelled")
    ap.add_argument("--no-diag", action="store_true")
    a = ap.parse_args()
    case = json.loads(Path(a.case).read_text())
    res = run_case(case, a.lmp, a.ffield_dir, a.out, a.build_label, diag=not a.no_diag)
    Path(a.out, "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("case_id", "valid", "invalid_reasons")}))
    return 0 if res["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
