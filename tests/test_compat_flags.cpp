// SPDX-License-Identifier: GPL-2.0-only
// Explicit regression tests for every LAMMPS element/mass compatibility predicate (M0 approval #2): boundaries,
// exceptions, and the exact-equality behaviour. Expected values are derived from the cited upstream expressions,
// NOT from running this engine.
#include <cmath>
#include <limits>
#include <vector>

#include "reaxmetal/compat_flags.hpp"
#include "test_util.hpp"

using namespace reaxmetal;

int main() {
  // ---- symbol normalisation: uppercase, truncate to 3 (reaxff_ffield.cpp:175-178)
  RM_CHECK(normalize_symbol("c") == "C");
  RM_CHECK(normalize_symbol("Fe") == "FE");
  RM_CHECK(normalize_symbol("CARBON") == "CAR");
  RM_CHECK(normalize_symbol("") == "");

  // ---- C2 species: normalized symbol must equal "C" exactly (strcmp)
  RM_CHECK(derive_species_flags("C", 12.0).c2_species);
  RM_CHECK(derive_species_flags("c", 12.0).c2_species);       // lower case is normalised first
  RM_CHECK(!derive_species_flags("CL", 35.45).c2_species);    // prefix is not a match
  RM_CHECK(!derive_species_flags("CA", 40.08).c2_species);
  RM_CHECK(!derive_species_flags("Cu", 63.5).c2_species);
  RM_CHECK(!derive_species_flags("CARBON", 12.0).c2_species); // truncates to "CAR", not "C"
  RM_CHECK(!derive_species_flags("O", 15.999).c2_species);
  RM_CHECK(!derive_species_flags("", 12.0).c2_species);
  RM_CHECK(derive_species_flags("C", 99.0).c2_species);       // the flag depends on the NAME only, never the mass
  RM_CHECK(derive_species_flags("X", 12.0).c2_species == false); // and a carbon-mass dummy is not "C"

  // ---- mass > 21.0 (strict) and mass < 21 (strict): exactly 21 is NEITHER
  RM_CHECK(!derive_species_flags("A", 20.999999).heavy_atom_terms && derive_species_flags("A", 20.999999).light_valency_override);
  RM_CHECK(!derive_species_flags("A", 21.0).heavy_atom_terms && !derive_species_flags("A", 21.0).light_valency_override);
  RM_CHECK(derive_species_flags("A", std::nextafter(21.0, 100.0)).heavy_atom_terms);
  RM_CHECK(!derive_species_flags("A", std::nextafter(21.0, 100.0)).light_valency_override);
  RM_CHECK(derive_species_flags("A", std::nextafter(21.0, 0.0)).light_valency_override);
  RM_CHECK(!derive_species_flags("A", std::nextafter(21.0, 0.0)).heavy_atom_terms);
  RM_CHECK(derive_species_flags("A", 1.008).light_valency_override);
  RM_CHECK(derive_species_flags("A", 63.546).heavy_atom_terms);
  RM_CHECK(derive_species_flags("A", 0.0).light_valency_override && !derive_species_flags("A", 0.0).heavy_atom_terms);
  RM_CHECK(derive_species_flags("A", -1.0).light_valency_override);  // upstream does not validate the sign (parser will, M2)
  // NaN: every upstream comparison is false -> no flag set
  const double nan = std::numeric_limits<double>::quiet_NaN();
  {
    SpeciesFlags f = derive_species_flags("A", nan);
    RM_CHECK(!f.heavy_atom_terms && !f.light_valency_override && !f.mass_exactly_12_0000 && !f.mass_exactly_15_9990);
  }

  // ---- light-element valency override applies only when the two valencies differ (exact !=)
  const SpeciesFlags light = derive_species_flags("H", 1.008), exact21 = derive_species_flags("A", 21.0);
  RM_CHECK(light_valency_override_applies(light, 1.0, 1.0000001));
  RM_CHECK(!light_valency_override_applies(light, 1.0, 1.0));
  RM_CHECK(!light_valency_override_applies(exact21, 3.0, 4.0));                    // mass == 21: not overridden
  RM_CHECK(!light_valency_override_applies(derive_species_flags("A", 22.0), 3.0, 4.0));

  // ---- exact mass equality (double ==): "12.0000" and "15.9990" as parsed from text
  RM_CHECK(derive_species_flags("C", 12.0000).mass_exactly_12_0000);
  RM_CHECK(!derive_species_flags("C", 12.011).mass_exactly_12_0000);               // natural-abundance carbon does NOT qualify
  RM_CHECK(!derive_species_flags("C", std::nextafter(12.0, 13.0)).mass_exactly_12_0000);
  RM_CHECK(derive_species_flags("O", 15.9990).mass_exactly_15_9990);
  RM_CHECK(!derive_species_flags("O", std::nextafter(15.9990, 16.0)).mass_exactly_15_9990);
  RM_CHECK(!derive_species_flags("O", 15.9994).mass_exactly_15_9990);
  RM_CHECK(!derive_species_flags("O", 16.0).mass_exactly_15_9990);
  RM_CHECK(derive_species_flags("Q", 12.0).mass_exactly_12_0000);                  // name-independent

  // ---- terminal-triple-bond stabilisation pair rule (either order, or gp37 rule)
  const SpeciesFlags c12 = derive_species_flags("C", 12.0), o16 = derive_species_flags("O", 15.9990),
                     c11 = derive_species_flags("C", 12.011), n14 = derive_species_flags("N", 14.0067);
  RM_CHECK(triple_bond_stabilisation_pair(false, c12, o16));
  RM_CHECK(triple_bond_stabilisation_pair(false, o16, c12));    // symmetric
  RM_CHECK(!triple_bond_stabilisation_pair(false, c12, c12));
  RM_CHECK(!triple_bond_stabilisation_pair(false, o16, o16));
  RM_CHECK(!triple_bond_stabilisation_pair(false, c11, o16));   // 12.011 is not 12.0000
  RM_CHECK(!triple_bond_stabilisation_pair(false, c12, n14));
  RM_CHECK(triple_bond_stabilisation_pair(true, n14, n14));     // gp37 rule makes it unconditional
  RM_CHECK(triple_bond_stabilisation_pair(true, c11, c11));
  // masses chosen to match by mass but named differently still qualify (reference tests masses, not names)
  RM_CHECK(triple_bond_stabilisation_pair(false, derive_species_flags("Zz", 12.0), derive_species_flags("Yy", 15.9990)));

  // ---- gp[37]: (int) truncation toward zero, == 2
  auto gp_with37 = [](double v) { std::vector<double> g(39, 0.0); g[37] = v; return g; };
  RM_CHECK(gp37_forces_stabilisation(gp_with37(2.0)));
  RM_CHECK(gp37_forces_stabilisation(gp_with37(2.9999)));       // truncates to 2
  RM_CHECK(!gp37_forces_stabilisation(gp_with37(1.9999)));      // truncates to 1
  RM_CHECK(!gp37_forces_stabilisation(gp_with37(3.0)));
  RM_CHECK(!gp37_forces_stabilisation(gp_with37(0.0)));
  RM_CHECK(!gp37_forces_stabilisation(gp_with37(-2.0)));
  RM_CHECK(!gp37_forces_stabilisation(gp_with37(-2.5)));        // (int)-2.5 == -2, not 2
  RM_CHECK(gp37_forces_stabilisation(gp_with37(2.0000001)));
  RM_EXPECT_THROW(gp37_forces_stabilisation(gp_with37(nan)), std::invalid_argument);
  RM_EXPECT_THROW(gp37_forces_stabilisation(gp_with37(1e300)), std::invalid_argument);
  {  // ffield with <= 37 general parameters: upstream reads out of bounds; the engine must refuse
    std::vector<double> shortgp(37, 0.0);
    RM_EXPECT_THROW(gp37_forces_stabilisation(shortgp), std::invalid_argument);
    std::vector<double> exactly38(38, 0.0);
    RM_CHECK(!gp37_forces_stabilisation(exactly38));            // gp[37] exists (== 0)
  }

  // ---- C2-correction gate: gp[5] > 0.001 strictly
  auto gp_with5 = [](double v) { std::vector<double> g(39, 0.0); g[5] = v; return g; };
  RM_CHECK(!c2_correction_active(gp_with5(0.001)));
  RM_CHECK(c2_correction_active(gp_with5(std::nextafter(0.001, 1.0))));
  RM_CHECK(c2_correction_active(gp_with5(70.0)));               // typical value (ffield.reax.cho: 70.0)
  RM_CHECK(!c2_correction_active(gp_with5(0.0)));
  RM_CHECK(!c2_correction_active(gp_with5(-5.0)));
  RM_CHECK(!c2_correction_active(gp_with5(nan)));
  RM_EXPECT_THROW(c2_correction_active(std::vector<double>(5, 1.0)), std::invalid_argument);
  return rmtest::finish("compat_flags");
}
