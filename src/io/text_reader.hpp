// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// Line / token reader reproducing the behaviour of LAMMPS' TextFileReader + ValueTokenizer as used by the ReaxFF readers
// (src/text_file_reader.cpp, src/tokenizer.cpp, src/utils.cpp @ stable_30Sep2026), with `ignore_comments = false`:
//   * words are separated by space, tab, CR, LF, FF; a trailing "! comment" is just more words (it counts as columns);
//   * next_line()/next_values() skip lines without any word; skip_line() consumes exactly one raw line;
//   * numbers: whole-token std::stod / std::stoi semantics (hex floats and exponent forms accepted, partial parses rejected).
// Differences (documented in ENGINE_SPEC 2.7): non-finite numbers are rejected; physical line numbers are reported;
// lines longer than LAMMPS' 1023-character buffer are not split.
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "reaxmetal/forcefield.hpp"

namespace reaxmetal::detail {

struct EofReached {};  // thrown when the file ends where more input is required

inline bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f'; }

class Tokens {
 public:
  Tokens(std::vector<std::string_view> t, std::string where) : toks_(std::move(t)), where_(std::move(where)) {}
  std::size_t count() const { return toks_.size(); }
  bool has_next() const { return pos_ < toks_.size(); }
  std::string_view next_string() {
    if (pos_ >= toks_.size()) throw FfieldError(where_ + ": missing value");
    return toks_[pos_++];
  }
  void skip() { next_string(); }
  double next_double() {
    const std::string s(next_string());
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || static_cast<std::size_t>(end - s.c_str()) != s.size())
      throw FfieldError(where_ + ": '" + s + "' is not a valid floating point number");
    if (!std::isfinite(v)) throw FfieldError(where_ + ": '" + s + "' is not finite (strict mode; LAMMPS would accept inf/nan)");
    return v;
  }
  int next_int() {
    const std::string s(next_string());
    std::size_t i = 0;
    bool neg = false;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) neg = (s[i++] == '-');
    if (i >= s.size()) throw FfieldError(where_ + ": '" + s + "' is not a valid integer");
    long long v = 0;
    for (; i < s.size(); ++i) {
      if (s[i] < '0' || s[i] > '9') throw FfieldError(where_ + ": '" + s + "' is not a valid integer");
      v = v * 10 + (s[i] - '0');
      if (v > static_cast<long long>(std::numeric_limits<int>::max()) + 1) throw FfieldError(where_ + ": integer '" + s + "' out of range");
    }
    if (neg) v = -v;
    if (v > std::numeric_limits<int>::max() || v < std::numeric_limits<int>::min()) throw FfieldError(where_ + ": integer '" + s + "' out of range");
    return static_cast<int>(v);
  }
  // is the first token a floating point number? (LAMMPS: values.matches(R"(^\s*\f+\s*)") on the line)
  bool first_is_number() const {
    if (toks_.empty()) return false;
    const std::string s(toks_[0]);
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && static_cast<std::size_t>(end - s.c_str()) == s.size();
  }

 private:
  std::vector<std::string_view> toks_;
  std::size_t pos_ = 0;
  std::string where_;
};

class Reader {
 public:
  Reader(std::string_view text, std::string_view name) : text_(text), name_(name) {}
  int lineno() const { return lineno_; }
  const std::string& name() const { return name_; }
  std::string where() const { return name_ + ":" + std::to_string(lineno_); }

  // one raw line, or nullopt at EOF
  std::optional<std::string_view> raw_line() {
    if (pos_ >= text_.size()) return std::nullopt;
    const std::size_t nl = text_.find('\n', pos_);
    const std::size_t end = (nl == std::string_view::npos) ? text_.size() : nl;
    std::string_view line = text_.substr(pos_, end - pos_);
    pos_ = (nl == std::string_view::npos) ? text_.size() : nl + 1;
    ++lineno_;
    return line;
  }
  static std::vector<std::string_view> split(std::string_view line) {
    std::vector<std::string_view> t;
    std::size_t i = 0;
    while (i < line.size()) {
      while (i < line.size() && is_ws(line[i])) ++i;
      if (i >= line.size()) break;
      const std::size_t s = i;
      while (i < line.size() && !is_ws(line[i])) ++i;
      t.push_back(line.substr(s, i - s));
    }
    return t;
  }
  // next line containing at least one word (blank lines skipped); nullopt at EOF
  std::optional<Tokens> next_line() {
    while (auto l = raw_line()) {
      auto t = split(*l);
      if (!t.empty()) return Tokens(std::move(t), where());
    }
    return std::nullopt;
  }
  Tokens next_values() {
    if (auto t = next_line()) return std::move(*t);
    throw EofReached{};
  }
  void skip_line() {
    if (!raw_line()) throw EofReached{};
  }

 private:
  std::string_view text_;
  std::string name_;
  std::size_t pos_ = 0;
  int lineno_ = 0;
};

}  // namespace reaxmetal::detail
