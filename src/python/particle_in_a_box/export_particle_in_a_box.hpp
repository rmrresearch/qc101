#include "../../qc101/particle_in_a_box/particle_in_a_box.hpp"
#include <Eigen/Dense>
#include <sigma/sigma.hpp>
#include <pybind11/pybind11.h>

namespace qc101::particle_in_a_box {

inline auto proof_of_concept(std::vector<double> weight,
                             std::vector<double> grid,
                              double energy_uncertainty){
    
    using sigma_t  = sigma::Uncertain<double>;
    using vector_t = Eigen::Vector<sigma_t, Eigen::Dynamic>;
    using matrix_t = Eigen::Matrix<sigma_t, Eigen::Dynamic, Eigen::Dynamic>;

    distance_type L = 1.0;
    mass_type m = 1.0;
    auto n_states = weight.size();
    vector_t egy(n_states);
    matrix_t p0(n_states, n_states);
    
    for(decltype(n_states) i = 0; i < n_states; ++i){
        egy[i] = sigma_t{energy(i + 1, m, L), energy_uncertainty};
        for(decltype(n_states) j = 0; j < n_states; ++j){
            p0(i,j) = sigma_t{weight[i]*weight[j], 0.0};
        }
    }

    double dt = 0.01;
    decltype(n_states) n_time_steps = 10;
    for(decltype(n_states) step_i = 0; step_i < n_time_steps; ++step_i){
        auto pt = time_dependent_density(p0, egy, step_i * dt);
        auto x = real_space_value(pt, grid, L);
        for(const auto& xi : x)
            std::cout << xi << std::endl;
        p0 = pt;
    }
    return 2;
}

inline void export_modules(pybind11::module_& m){
    m.def("proof_of_concept", &proof_of_concept);
}

}