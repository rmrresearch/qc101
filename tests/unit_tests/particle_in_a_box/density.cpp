#include <catch2/catch.hpp>
#include <qc101/particle_in_a_box/density.hpp>
#include <qc101/particle_in_a_box/energy.hpp>

using namespace qc101;


TEST_CASE("particle_in_a_box::time_dependent_density<Eigen::Matrix>") {
  using particle_in_a_box::time_dependent_density;
  using particle_in_a_box::energy;

  particle_in_a_box::mass_type mass = 1.0;
  particle_in_a_box::distance_type L = 1.0;

  using matrix_type = Eigen::MatrixXd;
  using density_type = particle_in_a_box::Density<matrix_type>;

  density_type p0(2);
  p0.m_real(0, 0) = 0.5;
  p0.m_imag(0, 0) = 0.0;
  p0.m_real(0, 1) = 0.5;
  p0.m_imag(0, 1) = 0.0;
  p0.m_real(1, 0) = 0.5;
  p0.m_imag(1, 0) = 0.0;
  p0.m_real(1, 1) = 0.5;
  p0.m_imag(1, 1) = 0.0;

 SECTION("Energy = Eigen::VectorXd"){
  Eigen::VectorXd energies(2);
  energies[0] = energy(1, mass, L);
  energies[1] = energy(2, mass, L);
  SECTION("dt == 0"){
    auto pt = time_dependent_density(p0, energies, 0.0);
    // Multiplies by 1.0, so floating-point equality should be fine
    REQUIRE(pt.m_real(0, 0) == 0.5);
    REQUIRE(pt.m_real(0, 1) == 0.5);
    REQUIRE(pt.m_real(1, 0) == 0.5);
    REQUIRE(pt.m_real(1, 1) == 0.5);
    REQUIRE(pt.m_imag(0, 0) == 0.0);
    REQUIRE(pt.m_imag(0, 1) == 0.0);
    REQUIRE(pt.m_imag(1, 0) == 0.0);
    REQUIRE(pt.m_imag(1, 1) == 0.0);
  }

    SECTION("dt == 0.01"){
        auto pt = time_dependent_density(p0, energies, 0.01);
        REQUIRE_THAT(pt.m_real(0, 0), Catch::WithinRel(0.5, 1e-6));
        REQUIRE_THAT(pt.m_imag(0, 0) + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(pt.m_real(0, 1), Catch::WithinRel(0.4945307388, 1e-6));
        REQUIRE_THAT(pt.m_imag(0, 1), Catch::WithinRel(0.07375194, 1e-6));
        REQUIRE_THAT(pt.m_real(1, 0), Catch::WithinRel(0.4945307388, 1e-6));
        REQUIRE_THAT(pt.m_imag(1, 0), Catch::WithinRel(-0.07375194, 1e-6));
        REQUIRE_THAT(pt.m_real(1, 1), Catch::WithinRel(0.5, 1e-6));
        REQUIRE_THAT(pt.m_imag(1, 1) + 1.0, Catch::WithinRel(1.0, 1e-6));
   }

   SECTION("dt == 0.1"){
     auto pt = time_dependent_density(p0, energies, 0.1);
     REQUIRE_THAT(pt.m_real(0, 0), Catch::WithinRel(0.5, 1e-6));
     REQUIRE_THAT(pt.m_imag(0, 0) + 1.0, Catch::WithinRel(1.0, 1e-6));
     REQUIRE_THAT(pt.m_real(0, 1), Catch::WithinRel(0.0451163853, 1e-6));
     REQUIRE_THAT(pt.m_imag(0, 1), Catch::WithinRel(0.49796035, 1e-6));
     REQUIRE_THAT(pt.m_real(1, 0), Catch::WithinRel(0.0451163853, 1e-6));
     REQUIRE_THAT(pt.m_imag(1, 0), Catch::WithinRel(-0.49796035, 1e-6));
     REQUIRE_THAT(pt.m_real(1, 1), Catch::WithinRel(0.5, 1e-6));
     REQUIRE_THAT(pt.m_imag(1, 1) + 1.0, Catch::WithinRel(1.0, 1e-6));
    }
 }
}

TEST_CASE("particle_in_a_box::time_dependent_density<sigma::UDouble>") {
  using particle_in_a_box::time_dependent_density;
  using particle_in_a_box::energy;

  particle_in_a_box::mass_type mass = 1.0;
  particle_in_a_box::distance_type L = 1.0;

  using matrix_type = Eigen::MatrixXd;
  using density_type = particle_in_a_box::Density<matrix_type>;

  density_type p0(2);
  p0.m_real(0, 0) = 0.5;
  p0.m_imag(0, 0) = 0.0;
  p0.m_real(0, 1) = 0.5;
  p0.m_imag(0, 1) = 0.0;
  p0.m_real(1, 0) = 0.5;
  p0.m_imag(1, 0) = 0.0;
  p0.m_real(1, 1) = 0.5;
  p0.m_imag(1, 1) = 0.0;

 SECTION("Energy = Eigen::VectorXd"){
  Eigen::Vector<sigma::UDouble, Eigen::Dynamic> energies(2);
  energies[0] = sigma::UDouble(energy(1, mass, L), 0.1);
  energies[1] = sigma::UDouble(energy(2, mass, L), 0.1);
  
  SECTION("dt == 0"){
    auto pt = time_dependent_density(p0, energies, 0.0);
    // Multiplies by 1.0, so floating-point equality should be fine
    REQUIRE(pt.m_real(0, 0).mean() == 0.5);
    REQUIRE(pt.m_real(0, 1).mean() == 0.5);
    REQUIRE(pt.m_real(1, 0).mean() == 0.5);
    REQUIRE(pt.m_real(1, 1).mean() == 0.5);
    REQUIRE(pt.m_imag(0, 0).mean() == 0.0);
    REQUIRE(pt.m_imag(0, 1).mean() == 0.0);
    REQUIRE(pt.m_imag(1, 0).mean() == 0.0);
    REQUIRE(pt.m_imag(1, 1).mean() == 0.0);

    REQUIRE(pt.m_real(0, 0).sd() == 0.0);
    REQUIRE(pt.m_real(0, 1).sd() == 0.0);
    REQUIRE(pt.m_real(1, 0).sd() == 0.0);
    REQUIRE(pt.m_real(1, 1).sd() == 0.0);
    REQUIRE(pt.m_imag(0, 0).sd() == 0.0);
    REQUIRE(pt.m_imag(0, 1).sd() == 0.0);
    REQUIRE(pt.m_imag(1, 0).sd() == 0.0);
    REQUIRE(pt.m_imag(1, 1).sd() == 0.0);
  }

    SECTION("dt == 0.01"){
        auto pt = time_dependent_density(p0, energies, 0.01);
        REQUIRE_THAT(pt.m_real(0, 0).mean(), Catch::WithinRel(0.5, 1e-6));
        REQUIRE_THAT(pt.m_imag(0, 0).mean() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(pt.m_real(0, 1).mean(), Catch::WithinRel(0.4945307388, 1e-6));
        REQUIRE_THAT(pt.m_imag(0, 1).mean(), Catch::WithinRel(0.07375194, 1e-6));
        REQUIRE_THAT(pt.m_real(1, 0).mean(), Catch::WithinRel(0.4945307388, 1e-6));
        REQUIRE_THAT(pt.m_imag(1, 0).mean(), Catch::WithinRel(-0.07375194, 1e-6));
        REQUIRE_THAT(pt.m_real(1, 1).mean(), Catch::WithinRel(0.5, 1e-6));
        REQUIRE_THAT(pt.m_imag(1, 1).mean() + 1.0, Catch::WithinRel(1.0, 1e-6));

        REQUIRE_THAT(pt.m_real(0, 0).sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(pt.m_imag(0, 0).sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(pt.m_real(0, 1).sd(), Catch::WithinRel(0.000104301, 1e-6));
        REQUIRE_THAT(pt.m_imag(0, 1).sd(), Catch::WithinRel(0.0006993721, 1e-6));
        REQUIRE_THAT(pt.m_real(1, 0).sd(), Catch::WithinRel(0.000104301, 1e-6));
        REQUIRE_THAT(pt.m_imag(1, 0).sd(), Catch::WithinRel(0.0006993721, 1e-6));
        REQUIRE_THAT(pt.m_real(1, 1).sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(pt.m_imag(1, 1).sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
   }

   SECTION("dt == 0.1"){
     auto pt = time_dependent_density(p0, energies, 0.1);
     REQUIRE_THAT(pt.m_real(0, 0).mean(), Catch::WithinRel(0.5, 1e-6));
     REQUIRE_THAT(pt.m_imag(0, 0).mean() + 1.0, Catch::WithinRel(1.0, 1e-6));
     REQUIRE_THAT(pt.m_real(0, 1).mean(), Catch::WithinRel(0.0451163853, 1e-6));
     REQUIRE_THAT(pt.m_imag(0, 1).mean(), Catch::WithinRel(0.49796035, 1e-6));
     REQUIRE_THAT(pt.m_real(1, 0).mean(), Catch::WithinRel(0.0451163853, 1e-6));
     REQUIRE_THAT(pt.m_imag(1, 0).mean(), Catch::WithinRel(-0.49796035, 1e-6));
     REQUIRE_THAT(pt.m_real(1, 1).mean(), Catch::WithinRel(0.5, 1e-6));
     REQUIRE_THAT(pt.m_imag(1, 1).mean() + 1.0, Catch::WithinRel(1.0, 1e-6));

     REQUIRE_THAT(pt.m_real(0, 0).sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
     REQUIRE_THAT(pt.m_imag(0, 0).sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
     REQUIRE_THAT(pt.m_real(0, 1).sd(), Catch::WithinRel(0.0070422228, 1e-6));
     REQUIRE_THAT(pt.m_imag(0, 1).sd(), Catch::WithinRel(0.000638042, 1e-6));
     REQUIRE_THAT(pt.m_real(1, 0).sd(), Catch::WithinRel(0.0070422228, 1e-6));
     REQUIRE_THAT(pt.m_imag(1, 0).sd(), Catch::WithinRel(0.000638042, 1e-6));
     REQUIRE_THAT(pt.m_real(1, 1).sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
     REQUIRE_THAT(pt.m_imag(1, 1).sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
    }
 }
}