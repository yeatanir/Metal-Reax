// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Minimal assertion helpers (no third-party test framework dependency at M0).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace rmtest {

inline int& failures() {
  static int n = 0;
  return n;
}

#define RM_CHECK(cond)                                                                \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
      ++rmtest::failures();                                                           \
    }                                                                                 \
  } while (0)

#define RM_CHECK_MSG(cond, msg)                                                       \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "CHECK FAILED %s:%d: %s -- %s\n", __FILE__, __LINE__, #cond, \
                   std::string(msg).c_str());                                         \
      ++rmtest::failures();                                                           \
    }                                                                                 \
  } while (0)

#define RM_EXPECT_THROW(expr, ExcType)                                                \
  do {                                                                                \
    bool thrown_ = false;                                                             \
    try { (void)(expr); } catch (const ExcType&) { thrown_ = true; } catch (...) {}   \
    if (!thrown_) {                                                                   \
      std::fprintf(stderr, "EXPECTED %s FROM %s at %s:%d\n", #ExcType, #expr, __FILE__, __LINE__); \
      ++rmtest::failures();                                                           \
    }                                                                                 \
  } while (0)

inline std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    ++failures();
    return {};
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// key=value lines, '#' comments, as used by third_party/lammps/PIN.txt
inline std::map<std::string, std::string> read_kv(const std::string& path) {
  std::map<std::string, std::string> kv;
  std::istringstream in(read_file(path));
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    kv.emplace(line.substr(0, eq), line.substr(eq + 1));
  }
  return kv;
}

inline int finish(const char* name) {
  if (failures() == 0) {
    std::printf("%s: all checks passed\n", name);
    return EXIT_SUCCESS;
  }
  std::printf("%s: %d check(s) FAILED\n", name, failures());
  return EXIT_FAILURE;
}

}  // namespace rmtest
