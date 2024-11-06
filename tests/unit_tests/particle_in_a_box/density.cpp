#include <catch2/catch.hpp>
#include <qc101/particle_in_a_box/density.hpp>
#include <qc101/particle_in_a_box/energy.hpp>
#include <qc101/particle_in_a_box/wavefunction.hpp>

using namespace qc101;

using particle_in_a_box::energy;
using particle_in_a_box::real_space_time_dependent_density;
using particle_in_a_box::real_space_wavefunction;

namespace {

template<typename DensityType>
auto prepare_superposition() {
    DensityType p0(2);

    p0.m_real(0, 0) = 0.5;
    p0.m_real(0, 1) = 0.5;
    p0.m_real(1, 0) = 0.5;
    p0.m_real(1, 1) = 0.5;

    p0.m_imag(0, 0) = 0.0;
    p0.m_imag(0, 1) = 0.0;
    p0.m_imag(1, 0) = 0.0;
    p0.m_imag(1, 1) = 0.0;

    return p0;
}

} // namespace

TEST_CASE("PIB::time_dependent_density (elements == double)") {
    particle_in_a_box::mass_type mass  = 1.0;
    particle_in_a_box::distance_type L = 1.0;

    using matrix_type  = Eigen::MatrixXd;
    using density_type = particle_in_a_box::Density<matrix_type>;

    auto p0 = prepare_superposition<density_type>();

    SECTION("Energy = Eigen::VectorXd") {
        Eigen::VectorXd energies(2);
        energies[0] = energy(1, mass, L);
        energies[1] = energy(2, mass, L);
        SECTION("dt == 0") {
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

        SECTION("dt == 0.01") {
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

        SECTION("dt == 0.1") {
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

TEST_CASE("PIB::time_dependent_density (elements == sigma::UDouble>)") {
    particle_in_a_box::mass_type mass  = 1.0;
    particle_in_a_box::distance_type L = 1.0;

    using matrix_type  = Eigen::MatrixXd;
    using density_type = particle_in_a_box::Density<matrix_type>;

    auto p0 = prepare_superposition<density_type>();

    SECTION("Energy = Eigen::VectorXd") {
        Eigen::Vector<sigma::UDouble, Eigen::Dynamic> energies(2);
        energies[0] = sigma::UDouble(energy(1, mass, L), 0.1);
        energies[1] = sigma::UDouble(energy(2, mass, L), 0.1);

        SECTION("dt == 0") {
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

        SECTION("dt == 0.01") {
            auto pt = time_dependent_density(p0, energies, 0.01);
            REQUIRE_THAT(pt.m_real(0, 0).mean(), Catch::WithinRel(0.5, 1e-6));
            REQUIRE_THAT(pt.m_imag(0, 0).mean() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));
            REQUIRE_THAT(pt.m_real(0, 1).mean(),
                         Catch::WithinRel(0.4945307388, 1e-6));
            REQUIRE_THAT(pt.m_imag(0, 1).mean(),
                         Catch::WithinRel(0.07375194, 1e-6));
            REQUIRE_THAT(pt.m_real(1, 0).mean(),
                         Catch::WithinRel(0.4945307388, 1e-6));
            REQUIRE_THAT(pt.m_imag(1, 0).mean(),
                         Catch::WithinRel(-0.07375194, 1e-6));
            REQUIRE_THAT(pt.m_real(1, 1).mean(), Catch::WithinRel(0.5, 1e-6));
            REQUIRE_THAT(pt.m_imag(1, 1).mean() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));

            REQUIRE_THAT(pt.m_real(0, 0).sd() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));
            REQUIRE_THAT(pt.m_imag(0, 0).sd() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));
            REQUIRE_THAT(pt.m_real(0, 1).sd(),
                         Catch::WithinRel(0.000104301, 1e-6));
            REQUIRE_THAT(pt.m_imag(0, 1).sd(),
                         Catch::WithinRel(0.0006993721, 1e-6));
            REQUIRE_THAT(pt.m_real(1, 0).sd(),
                         Catch::WithinRel(0.000104301, 1e-6));
            REQUIRE_THAT(pt.m_imag(1, 0).sd(),
                         Catch::WithinRel(0.0006993721, 1e-6));
            REQUIRE_THAT(pt.m_real(1, 1).sd() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));
            REQUIRE_THAT(pt.m_imag(1, 1).sd() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));
        }

        SECTION("dt == 0.1") {
            auto pt = time_dependent_density(p0, energies, 0.1);
            REQUIRE_THAT(pt.m_real(0, 0).mean(), Catch::WithinRel(0.5, 1e-6));
            REQUIRE_THAT(pt.m_imag(0, 0).mean() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));
            REQUIRE_THAT(pt.m_real(0, 1).mean(),
                         Catch::WithinRel(0.0451163853, 1e-6));
            REQUIRE_THAT(pt.m_imag(0, 1).mean(),
                         Catch::WithinRel(0.49796035, 1e-6));
            REQUIRE_THAT(pt.m_real(1, 0).mean(),
                         Catch::WithinRel(0.0451163853, 1e-6));
            REQUIRE_THAT(pt.m_imag(1, 0).mean(),
                         Catch::WithinRel(-0.49796035, 1e-6));
            REQUIRE_THAT(pt.m_real(1, 1).mean(), Catch::WithinRel(0.5, 1e-6));
            REQUIRE_THAT(pt.m_imag(1, 1).mean() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));

            REQUIRE_THAT(pt.m_real(0, 0).sd() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));
            REQUIRE_THAT(pt.m_imag(0, 0).sd() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));
            REQUIRE_THAT(pt.m_real(0, 1).sd(),
                         Catch::WithinRel(0.0070422228, 1e-6));
            REQUIRE_THAT(pt.m_imag(0, 1).sd(),
                         Catch::WithinRel(0.000638042, 1e-6));
            REQUIRE_THAT(pt.m_real(1, 0).sd(),
                         Catch::WithinRel(0.0070422228, 1e-6));
            REQUIRE_THAT(pt.m_imag(1, 0).sd(),
                         Catch::WithinRel(0.000638042, 1e-6));
            REQUIRE_THAT(pt.m_real(1, 1).sd() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));
            REQUIRE_THAT(pt.m_imag(1, 1).sd() + 1.0,
                         Catch::WithinRel(1.0, 1e-6));
        }
    }
}

TEST_CASE("PIB::real_space_time_dependent_density (elements == double)") {
    particle_in_a_box::mass_type mass  = 1.0;
    particle_in_a_box::distance_type L = 1.0;

    using matrix_type  = Eigen::MatrixXd;
    using density_type = particle_in_a_box::Density<matrix_type>;

    auto p0 = prepare_superposition<density_type>();

    Eigen::VectorXd energies(2);
    energies[0] = energy(1, mass, L);
    energies[1] = energy(2, mass, L);

    std::vector<double> grid_points{0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0};
    auto grid = real_space_wavefunction(2, L, grid_points);

    SECTION("t == 0") {
        auto value = real_space_time_dependent_density(p0, grid);
        REQUIRE_THAT(value[0] + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[1], Catch::WithinRel(3.0, 1e-6));
        REQUIRE_THAT(value[2] + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[3] + 1.0, Catch::WithinRel(1.0, 1e-6));
    }
    SECTION("t == 0.1") {
        auto pt    = time_dependent_density(p0, energies, 0.1);
        auto value = real_space_time_dependent_density(pt, grid);
        REQUIRE_THAT(value[0] + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[1], Catch::WithinRel(1.63534916, 1e-6));
        REQUIRE_THAT(value[2], Catch::WithinRel(1.36465084, 1e-6));
        REQUIRE_THAT(value[3] + 1.0, Catch::WithinRel(1.0, 1e-6));
    }
    SECTION("t == 0.2") {
        auto pt    = time_dependent_density(p0, energies, 0.1);
        auto pt2   = time_dependent_density(pt, energies, 0.1);
        auto value = real_space_time_dependent_density(pt2, grid);
        REQUIRE_THAT(value[0] + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[1], Catch::WithinRel(2.442587e-2, 1e-6));
        REQUIRE_THAT(value[2], Catch::WithinRel(2.97557414, 1e-6));
        REQUIRE_THAT(value[3] + 1.0, Catch::WithinRel(1.0, 1e-6));
    }
}

TEST_CASE("PIB::real_space_time_dependent_density (element == sigma::double)") {
    particle_in_a_box::mass_type mass  = 1.0;
    particle_in_a_box::distance_type L = 1.0;

    using matrix_type  = Eigen::MatrixXd;
    using density_type = particle_in_a_box::Density<matrix_type>;

    auto p0 = prepare_superposition<density_type>();

    Eigen::Vector<sigma::UDouble, Eigen::Dynamic> energies(2);
    energies[0] = sigma::UDouble(energy(1, mass, L), 0.1);
    energies[1] = sigma::UDouble(energy(2, mass, L), 0.1);

    std::vector<double> grid_points{0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0};
    auto grid = real_space_wavefunction(2, L, grid_points);

    SECTION("t == 0") {
        auto pt    = time_dependent_density(p0, energies, 0.0);
        auto value = real_space_time_dependent_density(pt, grid);
        REQUIRE_THAT(value[0].mean() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[1].mean(), Catch::WithinRel(3.0, 1e-6));
        REQUIRE_THAT(value[2].mean() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[3].mean() + 1.0, Catch::WithinRel(1.0, 1e-6));

        REQUIRE_THAT(value[0].sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[1].sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[2].sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[3].sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
    }
    SECTION("t == 0.1") {
        auto pt    = time_dependent_density(p0, energies, 0.1);
        auto value = real_space_time_dependent_density(pt, grid);
        REQUIRE_THAT(value[0].mean() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[1].mean(), Catch::WithinRel(1.63534916, 1e-6));
        REQUIRE_THAT(value[2].mean(), Catch::WithinRel(1.36465084, 1e-6));
        REQUIRE_THAT(value[3].mean() + 1.0, Catch::WithinRel(1.0, 1e-6));

        REQUIRE_THAT(value[0].sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[1].sd(), Catch::WithinRel(0.0211266685, 1e-6));
        REQUIRE_THAT(value[2].sd(), Catch::WithinRel(0.0211266685, 1e-6));
        REQUIRE_THAT(value[3].sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
    }
    SECTION("t == 0.2") {
        auto pt    = time_dependent_density(p0, energies, 0.1);
        auto pt2   = time_dependent_density(pt, energies, 0.1);
        auto value = real_space_time_dependent_density(pt2, grid);

        REQUIRE_THAT(value[0].mean() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[1].mean(), Catch::WithinRel(2.442587e-2, 1e-6));
        REQUIRE_THAT(value[2].mean(), Catch::WithinRel(2.97557414, 1e-6));
        REQUIRE_THAT(value[3].mean() + 1.0, Catch::WithinRel(1.0, 1e-6));

        REQUIRE_THAT(value[0].sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
        REQUIRE_THAT(value[1].sd(), Catch::WithinRel(0.00762527, 1e-6));
        REQUIRE_THAT(value[2].sd(), Catch::WithinRel(0.00762527, 1e-6));
        REQUIRE_THAT(value[3].sd() + 1.0, Catch::WithinRel(1.0, 1e-6));
    }
}
