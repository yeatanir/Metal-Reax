#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# Fetch the pinned LAMMPS tree (sparse, shallow) and verify it against the pins.
#   usage: tools/fetch_lammps.sh [--full] <empty-or-new-destination-dir>
#     --full  complete shallow checkout (needed to BUILD LAMMPS); default is the sparse audit subset
# Never vendors upstream code into this repository; the checkout is a build/audit input only.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pin="$here/third_party/lammps/PIN.txt"
full=0
if [[ "${1:-}" == "--full" ]]; then full=1; shift; fi
dest="${1:?usage: fetch_lammps.sh [--full] <destination-dir>}"

get() { grep -E "^$1=" "$pin" | head -1 | cut -d= -f2-; }
repo="$(get lammps_repo)"; tag="$(get lammps_tag)"; commit="$(get lammps_commit)"
tagobj="$(get lammps_tag_object)"; dirs="$(get sparse_dirs)"

if [[ -e "$dest" && -n "$(ls -A "$dest" 2>/dev/null || true)" ]]; then
  echo "refusing to use non-empty destination: $dest" >&2; exit 2
fi

if [[ $full -eq 1 ]]; then
  git clone --quiet --depth 1 --branch "$tag" "$repo" "$dest"
else
  git clone --quiet --depth 1 --branch "$tag" --filter=blob:none --sparse "$repo" "$dest"
  git -C "$dest" sparse-checkout set $dirs
fi
got_commit="$(git -C "$dest" rev-parse HEAD)"
got_tagobj="$(git -C "$dest" rev-parse "refs/tags/$tag")"
[[ "$got_commit" == "$commit"  ]] || { echo "COMMIT MISMATCH: got $got_commit want $commit" >&2; exit 3; }
[[ "$got_tagobj" == "$tagobj" ]] || { echo "TAG OBJECT MISMATCH: got $got_tagobj want $tagobj" >&2; exit 3; }

# every audited file must hash identically to the manifest recorded at M0
( cd "$dest" && sha256sum --quiet -c "$here/third_party/lammps/SOURCE_HASHES.sha256" )
echo "OK: $tag @ $got_commit verified; $(wc -l < "$here/third_party/lammps/SOURCE_HASHES.sha256") files match manifest"
