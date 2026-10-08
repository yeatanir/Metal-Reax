// SPDX-License-Identifier: GPL-2.0-only
// docs/FEATURE_MATRIX.md must list every feature id with the same status word as the code table, and
// every mandated document must exist. A claim in the docs that the code does not carry (or vice versa)
// is exactly the kind of silent drift this project forbids.
#include <sstream>
#include <string>

#include "reaxmetal/capabilities.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

int main() {
  const std::string root = REAXMETAL_SOURCE_DIR;
  for (const char* doc : {"ENGINE_SPEC", "SOURCE_MAP", "FEATURE_MATRIX", "NUMERICAL_POLICY", "VALIDATION",
                          "DEVELOPMENT_LOG", "ARCHITECTURE_DECISIONS"}) {
    const std::string text = rmtest::read_file(root + "/docs/" + doc + ".md");
    RM_CHECK_MSG(text.size() > 500, std::string("docs/") + doc + ".md missing or suspiciously short");
  }

  const std::string matrix = rmtest::read_file(root + "/docs/FEATURE_MATRIX.md");
  std::istringstream in(matrix);
  std::string line;
  std::size_t table_rows_seen = 0;
  std::string all_ids_in_doc;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] != '|') continue;
    for (const Feature& f : feature_table()) {
      const std::string tag = "`" + std::string(f.id) + "`";
      // match the id as a whole backticked token in the first cell
      const auto first_cell_end = line.find('|', 1);
      if (first_cell_end == std::string::npos) continue;
      if (line.substr(0, first_cell_end).find(tag) == std::string::npos) continue;
      ++table_rows_seen;
      all_ids_in_doc += std::string(f.id) + "\n";
      RM_CHECK_MSG(line.find(std::string(to_string(f.status))) != std::string::npos,
                   "status mismatch for " + std::string(f.id) + " in: " + line);
      if (f.status == Status::Planned)
        RM_CHECK_MSG(line.find(std::string(f.milestone)) != std::string::npos, "milestone mismatch for " + std::string(f.id));
    }
  }
  for (const Feature& f : feature_table())
    RM_CHECK_MSG(all_ids_in_doc.find(std::string(f.id) + "\n") != std::string::npos,
                 "feature missing from FEATURE_MATRIX.md: " + std::string(f.id));
  RM_CHECK(table_rows_seen == feature_table().size());  // no id documented twice
  return rmtest::finish("docs_sync");
}
