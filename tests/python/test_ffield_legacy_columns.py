#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Runs the independent legacy-format slicer on a force field and hands the result to test_ffield_legacy_columns.
usage: test_ffield_legacy_columns.py --tool <test_ffield_legacy_columns> --ffield <file>"""
import argparse, subprocess, sys, tempfile
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--tool", required=True); ap.add_argument("--ffield", required=True)
a = ap.parse_args()
slicer = Path(__file__).with_name("legacy_ffield_slice.py")
with tempfile.TemporaryDirectory() as td:
    exp = Path(td) / "expected.txt"
    r = subprocess.run([sys.executable, "-I", str(slicer), a.ffield], capture_output=True, text=True)
    if r.returncode: print(r.stderr); sys.exit("slicer failed")
    exp.write_text(r.stdout)
    t = subprocess.run([a.tool, a.ffield, str(exp)], capture_output=True, text=True)
    print(t.stdout + t.stderr)
    sys.exit(t.returncode)
