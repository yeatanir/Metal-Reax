#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
"""Per-file license-notice audit of the upstream LAMMPS files this project derives from, links against, or tests with.

usage: audit_licenses.py <lammps-tree> > third_party/lammps/LICENSE_AUDIT.tsv
Classification is by the notice TEXT found in the first 60 lines of each file (never assumed uniform):
  GPL-2.0-or-later-PuReMD  : "either version 2 of the License, or (at your option) any later version"
  GPL-LAMMPS               : LAMMPS notice "distributed under the GNU General Public License" (resolved to GPL-2.0-only
                             by the tree's LICENSE / README: "GNU Public License (GPL) version 2")
  LGPL / MIT / other-named : reported verbatim by keyword
  NO-NOTICE                : no recognised notice (data files, YAML, ffields) -> falls under the tree-level LICENSE only
Not legal advice; a provenance record for the project owner.
"""
import glob, hashlib, os, re, sys

root = sys.argv[1]
SCOPE = ["src/REAXFF/*", "src/KOKKOS/*reaxff*", "src/KOKKOS/kokkos_type.h", "src/QEQ/fix_qeq.*", "src/QEQ/fix_qeq_shielded.*",
         "src/PLUGIN/*", "src/pair.h", "src/pair.cpp", "src/force.h", "src/neighbor.h", "src/neigh_list.h", "src/neigh_request.h",
         "src/comm.h", "src/lammpsplugin.h", "src/library.h", "src/lmptype.h", "src/fix.h",
         "examples/plugins/*", "potentials/ffield.reax.*", "potentials/README.reax",
         "unittest/force-styles/tests/atomic-pair-reaxff*.yaml", "unittest/force-styles/tests/reaxff.control",
         "src/fmt/format.h", "LICENSE"]

def classify(text):
    head = "\n".join(text.splitlines()[:60])
    cls = []
    if re.search(r"or \(at your option\) any later version", head): cls.append("GPL-2.0-or-later-PuReMD")
    if re.search(r"distributed under\s+the GNU General Public License", head): cls.append("GPL-LAMMPS")
    if re.search(r"Lesser General Public", head): cls.append("LGPL")
    if re.search(r"Permission is hereby granted, free of charge", head): cls.append("MIT-style")
    if re.search(r"GNU GENERAL PUBLIC LICENSE\s+Version 2", text[:400]): cls.append("GPL-2.0-text")
    return "+".join(cls) if cls else "NO-NOTICE"

print("path\tsha256_16\tnotice_class\tcopyright_lines\tcontributing_author")
seen = set()
for pat in SCOPE:
    for p in sorted(glob.glob(os.path.join(root, pat))):
        rel = os.path.relpath(p, root)
        if rel in seen or not os.path.isfile(p): continue
        seen.add(rel)
        data = open(p, "rb").read()
        txt = data.decode("utf-8", "replace")
        head = "\n".join(txt.splitlines()[:60])
        cr = "; ".join(sorted(set(m.strip() for m in re.findall(r"(Copyright[^\n]*?\d{4}[^\n]*)", head))))[:200]
        au = re.search(r"Contributing authors?:\s*(.*?)(?:\n\s*\n|-{10,}|\*/)", head, re.S)
        au = re.sub(r"\s+", " ", au.group(1)).strip()[:160] if au else ""
        print("%s\t%s\t%s\t%s\t%s" % (rel, hashlib.sha256(data).hexdigest()[:16], classify(txt), cr, au))
