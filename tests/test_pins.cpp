// SPDX-License-Identifier: GPL-2.0-only
// The compiled-in pins must equal third_party/lammps/PIN.txt, which in turn must carry full-length SHAs.
#include <cctype>

#include "reaxmetal/pins.hpp"
#include "test_util.hpp"

static bool is_sha1(const std::string& s) {
  if (s.size() != 40) return false;
  for (char c : s)
    if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
  return true;
}

int main() {
  const auto kv = rmtest::read_kv(std::string(REAXMETAL_SOURCE_DIR) + "/third_party/lammps/PIN.txt");
  auto get = [&](const char* k) {
    auto it = kv.find(k);
    RM_CHECK_MSG(it != kv.end(), std::string("PIN.txt missing key ") + k);
    return it == kv.end() ? std::string() : it->second;
  };
  RM_CHECK(get("lammps_repo") == reaxmetal::kLammpsRepo);
  RM_CHECK(get("lammps_tag") == reaxmetal::kLammpsTag);
  RM_CHECK(get("lammps_tag_object") == reaxmetal::kLammpsTagObject);
  RM_CHECK(get("lammps_commit") == reaxmetal::kLammpsCommit);
  RM_CHECK(get("lammps_license") == reaxmetal::kLammpsLicense);
  RM_CHECK(is_sha1(get("lammps_commit")));
  RM_CHECK(is_sha1(get("lammps_tag_object")));

  // the vendored license text must still be the one whose hash we recorded at M0
  const std::string manifest = rmtest::read_file(std::string(REAXMETAL_SOURCE_DIR) +
                                                 "/third_party/lammps/SOURCE_HASHES.sha256");
  RM_CHECK(manifest.find("be38e38d9482c2beae35") != std::string::npos);
  return rmtest::finish("pins");
}
