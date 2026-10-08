#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Write tests/fixtures/FIXTURES.sha256 (cases, golden references, force-field manifest). usage: hash_fixtures.py [dir]"""
import hashlib, sys
from pathlib import Path
d = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures")
files = sorted(list((d / "cases").glob("*.json")) + list((d / "reference").glob("*.json")) + [d / "ffield_manifest.tsv"])
(d / "FIXTURES.sha256").write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(d).as_posix()}\n" for f in files))
print(f"hashed {len(files)} files")
