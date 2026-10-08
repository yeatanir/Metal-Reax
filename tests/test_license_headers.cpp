// SPDX-License-Identifier: GPL-2.0-only
// Provenance enforcement (owner decision: GPL-2.0-only, per-file SPDX, no assumption that upstream files share terms):
//  1. every source/script file we author carries an SPDX identifier;
//  2. any file that says "Adapted-from:" must (a) retain an upstream notice block and (b) be listed in THIRD_PARTY_NOTICES.md;
//  3. LICENSE is the GPL-2.0 text and equals LICENSES/GPL-2.0-only.txt and third_party/lammps/COPYING.
#include <filesystem>
#include <set>
#include <string>

#include "test_util.hpp"

namespace fs = std::filesystem;

static bool is_authored_source(const fs::path& p) {
  const std::string name = p.filename().string(), ext = p.extension().string();
  return ext == ".cpp" || ext == ".hpp" || ext == ".h" || ext == ".py" || ext == ".sh" || ext == ".cmake" ||
         name == "CMakeLists.txt";
}

int main() {
  const fs::path root = REAXMETAL_SOURCE_DIR;
  const std::string notices = rmtest::read_file((root / "THIRD_PARTY_NOTICES.md").string());
  std::size_t scanned = 0, adapted = 0;
  for (const char* dir : {"include", "src", "tests", "plugin", "tools", "cmake"}) {
    if (!fs::exists(root / dir)) continue;
    for (const auto& e : fs::recursive_directory_iterator(root / dir)) {
      if (!e.is_regular_file() || !is_authored_source(e.path())) continue;
      ++scanned;
      const std::string text = rmtest::read_file(e.path().string());
      const std::string rel = fs::relative(e.path(), root).generic_string();
      RM_CHECK_MSG(text.substr(0, 800).find("SPDX-License-Identifier: GPL-2.0") != std::string::npos,
                   "missing SPDX identifier: " + rel);
      if (text.find("Adapted-from:") != std::string::npos && rel != "tests/test_license_headers.cpp") {
        ++adapted;
        RM_CHECK_MSG(text.find("Sandia Corporation") != std::string::npos || text.find("Purdue University") != std::string::npos,
                     "adapted file lacks upstream notice block: " + rel);
        RM_CHECK_MSG(notices.find(rel) != std::string::npos, "adapted file not listed in THIRD_PARTY_NOTICES.md: " + rel);
      }
    }
  }
  RM_CHECK(scanned >= 15);
  const std::string lic = rmtest::read_file((root / "LICENSE").string());
  RM_CHECK(lic.find("GNU GENERAL PUBLIC LICENSE") != std::string::npos && lic.find("Version 2, June 1991") != std::string::npos);
  RM_CHECK(lic == rmtest::read_file((root / "LICENSES/GPL-2.0-only.txt").string()));
  RM_CHECK(lic == rmtest::read_file((root / "third_party/lammps/COPYING").string()));
  RM_CHECK(rmtest::read_file((root / "REUSE.toml").string()).find("GPL-2.0-only") != std::string::npos);
  std::printf("license_headers: scanned %zu authored files, %zu with Adapted-from\n", scanned, adapted);
  return rmtest::finish("license_headers");
}
