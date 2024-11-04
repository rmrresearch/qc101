#include <catch2/catch.hpp>
#include <qc101/particle_in_a_box/energy.hpp>

using namespace qc101;

TEST_CASE("particle_in_a_box::energy") {
  using particle_in_a_box::energy;
  SECTION("electron, L == 1.0") {
    REQUIRE(energy(0, 1.0, 1.0) == 0.0);
    REQUIRE_THAT(energy(1, 1.0, 1.0), Catch::WithinRel(4.9348022005, 1.0e-6));
    REQUIRE_THAT(energy(2, 1.0, 1.0), Catch::WithinRel(19.7392088022, 1.0e-6));
  }
}