// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
#pragma once
// Machine-readable capability matrix. docs/FEATURE_MATRIX.md is the human-readable twin; the `docs_sync`
// test fails if the two disagree. Architectural rule 4: unsupported variants and charge models are
// detected and rejected explicitly -- never ignored, never partially honoured.
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace reaxmetal {

enum class Status {
  Implemented,  // exists AND has passing tests recorded in docs/VALIDATION.md
  Planned,      // scheduled for the named milestone; using it before then is an error
  Deferred,     // not supported NOW, may be scheduled later (owner decision); using it is an error (not a warning)
  Rejected,     // deliberately unsupported; using it is an error (not a warning)
  Ignored       // accepted, has no effect on physics in this engine; a notice must be logged
};

struct Feature {
  std::string_view id;                 // stable dotted id, also used verbatim in FEATURE_MATRIX.md
  Status status;
  std::string_view milestone;          // "M4", ... ; "-" for Rejected/Ignored
  std::string_view lammps_construct;   // how it appears in pinned LAMMPS (keyword/style/ffield feature)
  std::string_view note;
};

class UnsupportedFeatureError : public std::runtime_error {
 public:
  explicit UnsupportedFeatureError(const std::string& m) : std::runtime_error(m) {}
};
class NotImplementedError : public std::runtime_error {
 public:
  explicit NotImplementedError(const std::string& m) : std::runtime_error(m) {}
};

enum class Verdict { Supported, IgnoredWithNotice };

std::span<const Feature> feature_table() noexcept;
const Feature* find_feature(std::string_view id) noexcept;
// Exact match on Feature::lammps_construct (used by the ffield / input parsers from M2 on).
const Feature* find_by_lammps_construct(std::string_view construct) noexcept;

// Gatekeeper. Throws UnsupportedFeatureError (Rejected or Deferred) or NotImplementedError (Planned);
// throws std::logic_error for an id that is not in the table (a bug, never a user error).
Verdict require_supported(std::string_view id);

std::string_view to_string(Status s) noexcept;

}  // namespace reaxmetal
