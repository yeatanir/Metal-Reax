// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// Rule 4: unsupported variants/charge models are detected explicitly. Rule 5: nothing is silently dropped.
#include <set>
#include <string>

#include "reaxmetal/capabilities.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

int main() {
  const auto table = feature_table();
  RM_CHECK(table.size() > 40);

  std::set<std::string> ids, rejected_constructs;
  for (const Feature& f : table) {
    RM_CHECK_MSG(!f.id.empty(), "empty feature id");
    RM_CHECK_MSG(ids.insert(std::string(f.id)).second, "duplicate id " + std::string(f.id));
    RM_CHECK_MSG(find_feature(f.id) == &f, "find_feature round trip " + std::string(f.id));

    if (f.status == Status::Planned)
      RM_CHECK_MSG(f.milestone.size() >= 2 && f.milestone[0] == 'M', "Planned needs a milestone: " + std::string(f.id));
    else
      RM_CHECK_MSG(f.milestone == "-", "non-planned rows use '-' milestone: " + std::string(f.id));

    switch (f.status) {
      case Status::Planned:  RM_EXPECT_THROW(require_supported(f.id), NotImplementedError); break;
      case Status::Deferred:
      case Status::Rejected:
        RM_EXPECT_THROW(require_supported(f.id), UnsupportedFeatureError);
        if (f.lammps_construct == "-") {
          // only engine-internal rows (backends) may lack a LAMMPS spelling, and "-" must not be matchable
          RM_CHECK_MSG(f.id.substr(0, 8) == "backend.", "rejected non-backend row needs a LAMMPS construct: " + std::string(f.id));
          RM_CHECK(find_by_lammps_construct(f.lammps_construct) == nullptr);
        } else {
          // a parser must be able to recognise a rejected construct by its LAMMPS spelling
          RM_CHECK_MSG(rejected_constructs.insert(std::string(f.lammps_construct)).second, "construct not unique: " + std::string(f.lammps_construct));
          RM_CHECK(find_by_lammps_construct(f.lammps_construct) == &f);
        }
        break;
      case Status::Ignored:
        RM_CHECK(require_supported(f.id) == Verdict::IgnoredWithNotice);
        break;
      case Status::Implemented:
        RM_CHECK(require_supported(f.id) == Verdict::Supported);
        break;
    }
  }

  // The headline rejections from the mission statement must be present and must be errors, not warnings.
  for (const char* id : {"opt.tabulate"}) {
    const Feature* f = find_feature(id);
    RM_CHECK_MSG(f != nullptr, std::string("missing ") + id);
    if (f) RM_CHECK(f->status == Status::Deferred || f->status == Status::Rejected);
    if (f) RM_EXPECT_THROW(require_supported(id), UnsupportedFeatureError);
  }
  // Owner decision (M0 approval #8): standard EEM/QEq and LAMMPS-compatible shielded QEq must NOT be rejected.
  for (const char* id : {"qeq.reaxff", "qeq.shielded", "eem.external_cpu_fix", "eem.strict_convergence", "eem.compat_warn_continue"}) {
    const Feature* f = find_feature(id);
    RM_CHECK_MSG(f != nullptr, std::string("missing ") + id);
    if (f) RM_CHECK(f->status == Status::Planned || f->status == Status::Implemented);
  }
  // Barostat is gated on virial support, not forbidden: it may be Implemented only together with the virial feature (INT-2 pressure + INT-4 NPT).
  RM_CHECK(find_feature("md.barostat") && find_feature("lammps.virial_fdotr"));
  RM_CHECK(find_feature("md.barostat")->status == find_feature("lammps.virial_fdotr")->status);
  RM_CHECK(find_feature("md.barostat")->milestone == find_feature("lammps.virial_fdotr")->milestone);
  // Multi-rank is supported (INT-2 with 2 and 4 ranks); only the GPU charge matrix is limited to one rank.
  RM_CHECK(find_feature("lammps.multi_rank") && find_feature("lammps.multi_rank")->status == Status::Implemented);
  RM_CHECK(find_feature("qeq.gpu_single_rank") != nullptr);
  // Every one of the 13 energy terms must be represented by a term.* feature (rule 5).
  for (const char* id : {"term.bond", "term.lone_pair", "term.over_under", "term.valence", "term.penalty",
                         "term.coalition", "term.torsion", "term.conjugation", "term.hbond", "term.coulomb",
                         "term.polarization", "term.vdw.shielded", "term.vdw.inner_wall"})
    RM_CHECK_MSG(find_feature(id) != nullptr, std::string("missing ") + id);

  RM_EXPECT_THROW(require_supported("no.such.feature"), std::logic_error);
  RM_CHECK(find_feature("no.such.feature") == nullptr);
  return rmtest::finish("capabilities");
}
