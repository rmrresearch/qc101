#include <catch2/catch.hpp>
#include <qc101/particle_in_a_box/particle_in_a_box.hpp>

using namespace qc101;

double corr_energy(int n) {
  double h = 6.62607015e-34;   // J s
  double m = 9.109e-31;        // kg
  double L = 5.29177e-11;      // m
  double si2au = 2.2937104e17; // Hartrees per Joule
  return si2au * n * n * h * h / (8 * m * L * L);
}

TEST_CASE("particle_in_a_box::energy") {
  using particle_in_a_box::energy;
  SECTION("electron") {
    REQUIRE(energy(0, 1.0, 1.0) == 0.0);
    REQUIRE_THAT(energy(1, 1.0, 1.0), Catch::WithinRel(4.9348022005, 1.0e-6));
    REQUIRE_THAT(energy(2, 1.0, 1.0), Catch::WithinRel(19.7392088022, 1.0e-6));
  }
}

TEST_CASE("particle_in_a_box::time_dependent_wavefunction") {
  using particle_in_a_box::time_dependent_wavefunction;
  using particle_in_a_box::energy;

  SECTION("stationary state") {
    Eigen::VectorXcd c(1), egy(1);
    c[0] = 1.0;
    egy[0] = energy(1, 1.0, 1.0);
    auto c0 = time_dependent_wavefunction(c, egy, 0);
    REQUIRE(c0[0] == c[0]);
  }

  SECTION("Two-state") {
    Eigen::VectorXcd c(2), egy(2);
    c[0] = 1.0 / std::sqrt(2.0);
    c[1] = 1.0 / std::sqrt(2.0);
    egy[0] = energy(1, 1.0, 1.0);
    egy[1] = energy(2, 1.0, 1.0);

    auto ct = time_dependent_wavefunction(c, egy, 0.1);
    REQUIRE_THAT(ct[0].real(), Catch::WithinRel(0.62274161, 1.0e-6));
    REQUIRE_THAT(ct[0].imag(), Catch::WithinRel(-0.33495206, 1.0e-6));
    REQUIRE_THAT(ct[1].real(), Catch::WithinRel(-0.27739399, 1.0e-6));
    REQUIRE_THAT(ct[1].imag(), Catch::WithinRel(-0.65042492, 1.0e-6));
  }
}

TEST_CASE("particle_in_a_box::euler_step") {
  using particle_in_a_box::euler_step;
  using particle_in_a_box::energy;

  SECTION("stationary state") {
    Eigen::VectorXcd c(1), egy(1);
    c[0] = 1.0;
    egy[0] = energy(1, 1.0, 1.0);
    auto c0 = euler_step(c, egy, 0);
    REQUIRE(c0[0] == c[0]);
  }
  
  SECTION("Two-state") {
    Eigen::VectorXcd c(2), egy(2);
    c[0] = 1.0 / std::sqrt(2.0);
    c[1] = 1.0 / std::sqrt(2.0);
    egy[0] = energy(1, 1.0, 1.0);
    egy[1] = energy(2, 1.0, 1.0);

    auto ct = euler_step(c, egy, 0.01);
    REQUIRE_THAT(ct[0].real(), Catch::WithinRel(0.7071067812, 1.0e-6));
    REQUIRE_THAT(ct[0].imag(), Catch::WithinRel(-0.034894321, 1.0e-6));
    REQUIRE_THAT(ct[1].real(), Catch::WithinRel(0.7071067812, 1.0e-6));
    REQUIRE_THAT(ct[1].imag(), Catch::WithinRel(-0.139577284, 1.0e-6));
  }
}