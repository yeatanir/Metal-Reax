#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""INT-2: in-LAMMPS A/B. For every M1 fixture the SAME input is run with stock `pair_style reaxff` and with the plugin
`pair_style reaxff/metal ... backend <b>` (identical atoms, ghosts, skin, `fix qeq/reaxff` settings); compared: the 14 energy slots of
`compute pair`, potential energy, charges, total forces and the pressure (global virial by fdotr). Thresholds: frozen C1 (primary) for backend
cpu64 -- slot energies |d| <= rtol*max(1,|ref|), force component and RMS on the well-conditioned fixtures, charges; pressure: relative 1e-9.
usage: run_int2.py --lmp BIN --plugin SO --ffield-dir DIR [--backend cpu64] [--only NAME]"""
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "reaxref"))
import runner  # noqa: E402

SLOTS = ["eb", "ea", "elp", "emol", "ev", "epen", "ecoa", "ehb", "et", "eco", "ew", "ep", "efi", "eqeq"]


def run_lmp(lmp, case, ffield_dir, workdir, plugin=None, backend=None, env=None, gpu_qeq=False, np_=1):
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    ffpath = Path(ffield_dir) / case["ffield"]["name"]
    ffp = runner.parse_ffield(ffpath)
    data, inp, dump = workdir / "data.lmp", workdir / "in.lmp", workdir / "state.dump"
    runner.write_data(case, ffp, data)
    text = runner.build_input(case, ffpath, data, dump)
    if plugin:
        text = text.replace("pair_style reaxff NULL", f"plugin load {plugin}\npair_style reaxff/metal NULL backend {backend}", 1)
        text = text.replace("compute pp all pair reaxff", "compute pp all pair reaxff/metal")
        if gpu_qeq:   # fixtures with fixed charges have no fix to replace
            text = text.replace(" qeq/reaxff ", " qeq/reaxff/metal ")
    text = text.replace("thermo_style custom step pe", "thermo_style custom step pe press", 1)
    inp.write_text(text)
    launcher = ["mpirun", "--oversubscribe", "-np", str(np_)] if np_ > 1 else []
    r = subprocess.run(launcher + [lmp, "-in", str(inp), "-log", "none", "-nocite"], cwd=workdir, capture_output=True, text=True, env=env, timeout=180)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        raise RuntimeError(out[-2500:])
    th = runner.parse_thermo(out)
    if th is None:
        raise RuntimeError("no thermo line: " + out[-400:])
    n = len(case["atoms"])
    return th, runner.parse_dump(dump, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", required=True); ap.add_argument("--plugin", required=True); ap.add_argument("--ffield-dir", required=True)
    ap.add_argument("--backend", default="cpu64"); ap.add_argument("--only"); ap.add_argument("--np", type=int, default=1, help="MPI ranks for BOTH the stock and the plugin run (needs an MPI LAMMPS and a plugin built with REAXMETAL_LAMMPS_MPI)"); ap.add_argument("--gpu-qeq", action="store_true", help="plugin runs use fix qeq/reaxff/metal (needs --backend metal to use the GPU)")
    ap.add_argument("--fixtures", default=str(ROOT / "tests" / "fixtures"))
    a = ap.parse_args()
    fx = Path(a.fixtures)
    tol = json.loads((ROOT / "tolerances" / "tolerances.json").read_text())["C1"]["primary"]["threshold"]
    rtol, fcomp, frms, qtol = tol["energy_slot_rel_to_max1"], tol["force_component_abs"], tol["force_rms"], tol["charge_abs"]
    c3 = json.loads((ROOT / "tolerances" / "tolerances.json").read_text())["C3"]
    metal = a.backend == "metal"   # Metal-32 is judged by the owner-set C3 criteria (energy per atom, force max / RMS, charge), not the FP64 parity thresholds of C1
    if metal:
        fcomp, frms, qtol = c3["force_max_component_kcal_mol_A"], c3["force_rms_kcal_mol_A"], c3["charge_max_abs_e"]
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"), LD_LIBRARY_PATH=str(Path(a.lmp).resolve().parents[1] / "lib"),
               LAMMPS_POTENTIALS=a.ffield_dir)
    fails, n, nwc = [], 0, 0
    worst = {"slot": 0.0, "q": 0.0, "f": 0.0, "frms": 0.0, "p": 0.0, "pe": 0.0}
    with tempfile.TemporaryDirectory() as td:
        for cf in sorted((fx / "cases").glob("*.json")):
            if a.only and cf.stem != a.only:
                continue
            case = json.loads(cf.read_text()); ref = json.loads((fx / "reference" / cf.name).read_text())
            if not ref.get("valid", True):
                continue
            used = sorted(set(int(t) for t in ref["types"]))
            tried = [case] + [dict(case, elements=[case["elements"][u - 1] for u in used])]   # second try: only elements that have atoms (strict Q-12)
            res = None
            for k, c in enumerate(tried):
                try:
                    s_th, s_dump = run_lmp(a.lmp, c, a.ffield_dir, Path(td) / (cf.stem + "_stock"), env=env, np_=a.np)
                    o_th, o_dump = run_lmp(a.lmp, c, a.ffield_dir, Path(td) / (cf.stem + "_ours"), plugin=a.plugin, backend=a.backend, env=env, gpu_qeq=a.gpu_qeq, np_=a.np)
                    res = (s_th, s_dump, o_th, o_dump); break
                except RuntimeError as e:
                    if k == 0 and "bond-parameter block" in str(e):
                        continue
                    fails.append(f"{cf.stem}: {str(e)[:300]}"); break
            if res is None:
                continue
            s_th, s_dump, o_th, o_dump = res
            n += 1
            nat = len(case["atoms"])
            for i in range(14):
                key = f"c_pp[{i + 1}]"
                d = abs(o_th[key] - s_th[key]) / max(1.0, abs(s_th[key]))
                worst["slot"] = max(worst["slot"], d)
                if not metal and d > rtol: fails.append(f"{cf.stem}: slot {SLOTS[i]} stock {s_th[key]!r} ours {o_th[key]!r}")
                if metal and abs(o_th[key] - s_th[key]) / nat > c3["energy_per_atom_abs_kcal_mol"]: fails.append(f"{cf.stem}: slot {SLOTS[i]} stock {s_th[key]!r} ours {o_th[key]!r}")
            dpe = abs(o_th["PotEng"] - s_th["PotEng"]) / max(1.0, abs(s_th["PotEng"])); worst["pe"] = max(worst["pe"], dpe)
            worst["pe_atom"] = max(worst.get("pe_atom", 0.0), abs(o_th["PotEng"] - s_th["PotEng"]) / nat)
            if not metal and dpe > rtol: fails.append(f"{cf.stem}: PotEng stock {s_th['PotEng']!r} ours {o_th['PotEng']!r}")
            if metal and abs(o_th["PotEng"] - s_th["PotEng"]) / nat > c3["energy_per_atom_abs_kcal_mol"]: fails.append(f"{cf.stem}: PotEng stock {s_th['PotEng']!r} ours {o_th['PotEng']!r}")
            dq = abs(o_dump[:, 2] - s_dump[:, 2]).max(); worst["q"] = max(worst["q"], dq)
            if dq > qtol: fails.append(f"{cf.stem}: charges differ by {dq:.3e}")
            df = o_dump[:, 6:9] - s_dump[:, 6:9]
            fm, fr = abs(df).max(), (df ** 2).mean() ** 0.5
            wc = ref["conditioning"].get("well_conditioned_extended", True)
            worst["fall"] = max(worst.get("fall", 0.0), fm)
            if wc:
                nwc += 1
                worst["f"] = max(worst["f"], fm); worst["frms"] = max(worst["frms"], fr)
                if fm > fcomp or fr > frms: fails.append(f"{cf.stem}: forces differ: max {fm:.3e} rms {fr:.3e}")
            sp, op = s_th["Press"], o_th["Press"]
            dp = abs(op - sp) / max(1e-6, abs(sp), 1e-3); worst["p"] = max(worst["p"], dp)
            if not metal and abs(op - sp) > 1e-9 * max(1.0, abs(sp)): fails.append(f"{cf.stem}: pressure stock {sp!r} ours {op!r}")
    for f in fails[:60]: print("FAIL", f)
    print(f"worst: slots {worst['slot']:.2e} (limit {rtol:g}), PotEng {worst['pe']:.2e} (per atom {worst.get('pe_atom', 0):.2e}), charges {worst['q']:.2e} (limit {qtol:g}), "
          f"forces max {worst['f']:.2e} (limit {fcomp:g}) rms {worst['frms']:.2e} (limit {frms:g}); all fixtures incl. ill-conditioned: {worst.get('fall', 0):.2e}, pressure rel {worst['p']:.2e}")
    print(f"INT-2 backend {a.backend}: {n} fixtures compared, {nwc} well conditioned, {len(fails)} failures")
    ok = not fails and n >= (1 if a.only else 50)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
