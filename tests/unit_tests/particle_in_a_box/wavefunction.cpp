#include <catch2/catch.hpp>
#include <qc101/particle_in_a_box/energy.hpp>
#include <qc101/particle_in_a_box/wavefunction.hpp>

using namespace qc101;

TEST_CASE("particle_in_a_box::time_dependent_wavefunction") {
  using particle_in_a_box::time_dependent_wavefunction;
  using particle_in_a_box::energy;

  SECTION("stationary state") {
    Eigen::VectorXcd c(1), egy(1);
    c[0] = 1.0;
    egy[0] = energy(1, 1.0, 1.0);
    auto c0 = time_dependent_wavefunction(c, egy, 0);
    REQUIRE(c0[0] == c[0]); // copies value, i.e. floating-point equality is ok
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
    REQUIRE(c0[0] == c[0]); // copies value, i.e. floating-point equality is ok
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


namespace {

template<typename ValueType>
void check_n_equal_1(const ValueType& value){
  REQUIRE_THAT(value(0, 0) + 1.0, Catch::WithinRel(1.0, 1.0e-6));
  REQUIRE_THAT(value(1, 0), Catch::WithinRel(1.224744871391589, 1.0e-6));
  REQUIRE_THAT(value(2, 0), Catch::WithinRel(1.224744871391589, 1.0e-6));
  REQUIRE_THAT(value(3, 0) + 1.0, Catch::WithinRel(1.0, 1.0e-6));
}

}

TEST_CASE("particle_in_a_box::real_space_wavefunction"){
    // N.B. The shift by 1.0 is because the expected value is 0 and Catch2
    // has a hard time with zero comparisons.
    using particle_in_a_box::real_space_wavefunction;
    std::vector<double> grid{0.0, 1.0/3.0, 2.0/3.0, 1.0};
    SECTION("L = 1.0"){
        SECTION("n=0"){
            auto value = real_space_wavefunction(0, 1.0, grid);
            REQUIRE(value.rows() == 0);
            REQUIRE(value.cols() == 0);
        }
        SECTION("n=1"){
            auto value = real_space_wavefunction(1, 1.0, grid);
            check_n_equal_1(value);
        }
        SECTION("n=2"){
            auto value = real_space_wavefunction(2, 1.0, grid);
            check_n_equal_1(value);
            REQUIRE_THAT(value(0, 1) + 1.0, Catch::WithinRel(1.0, 1.0e-6));
            REQUIRE_THAT(value(1, 1), Catch::WithinRel(1.224744871391589, 1.0e-6));
            REQUIRE_THAT(value(2, 1), Catch::WithinRel(-1.224744871391589, 1.0e-6));
            REQUIRE_THAT(value(3, 1) + 1.0, Catch::WithinRel(1.0, 1.0e-6));
        }
    }
}