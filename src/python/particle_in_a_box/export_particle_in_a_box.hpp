#include "../../qc101/particle_in_a_box/particle_in_a_box.hpp"
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <sigma/sigma.hpp>

namespace qc101::particle_in_a_box {

inline auto proof_of_concept(std::vector<double> weight,
                             std::vector<double> grid,
                             double energy_uncertainty, double time_step,
                             int n_time_steps) {
    using sigma_type = sigma::UDouble;
    using matrix_type =
      Eigen::Matrix<sigma_type, Eigen::Dynamic, Eigen::Dynamic>;
    using density_type = Density<matrix_type>;

    mass_type mass  = 1.0;
    distance_type L = 1.0;
    auto n_states   = weight.size();
    density_type p0(n_states);
    for(decltype(n_states) i = 0; i < n_states; ++i) {
        const auto ci = weight[i];
        for(decltype(n_states) j = 0; j < n_states; ++j) {
            p0.m_real(i, j) = sigma_type(ci * weight[j], 0.0);
            p0.m_imag(i, j) = sigma_type(0.0, 0.0);
        }
    }

    Eigen::Vector<sigma_type, Eigen::Dynamic> energies(n_states);
    for(decltype(n_states) i = 0; i < n_states; ++i)
        energies[i] = sigma_type(energy(i + 1, mass, L), energy_uncertainty);

    std::vector<std::vector<double>> values(n_time_steps + 1);
    std::vector<std::vector<double>> errors(n_time_steps + 1);

    auto n_grid_points = grid.size();
    auto grid_vector   = real_space_wavefunction(n_states, L, grid);

    for(int i = 0; i <= n_time_steps; ++i) {
        double dt  = (i != 0 ? time_step : 0.0);
        auto pt    = time_dependent_density(p0, energies, dt);
        auto value = real_space_time_dependent_density(pt, grid_vector);
        std::vector<double> value_i(n_grid_points);
        std::vector<double> error_i(n_grid_points);
        for(decltype(n_grid_points) j = 0; j < n_grid_points; ++j) {
            value_i[j] = value[j].mean();
            error_i[j] = value[j].sd();
        }
        values[i] = std::move(value_i);
        errors[i] = std::move(error_i);
        p0        = pt;
    }
    return std::make_pair(values, errors);
}

inline void export_modules(pybind11::module_& m) {
    m.def("proof_of_concept", &proof_of_concept);
}

} // namespace qc101::particle_in_a_box