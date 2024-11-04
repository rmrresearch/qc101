#pragma once
#include "energy.hpp"
#include "wavefunction.hpp"
#include "types.hpp"

namespace qc101::particle_in_a_box {

template<typename MatrixType, typename VectorType>
auto time_dependent_density(const MatrixType& p0, 
                            const VectorType& energies, 
                            time_type dt){

    auto n = p0.rows();
    MatrixType pt(n, n);
    for(decltype(n) i = 0; i < n; ++i){
        for(decltype(n) j = 0; j < n; ++j){
            auto de = energies[i] - energies[j];
            pt(i, j) = sigma::cos(de * dt) * p0(i, j);
        }
    }
    return pt;
}

template<typename MatrixType, typename GridType>
inline auto real_space_value(const MatrixType& p, 
const GridType& x, distance_type L){
    auto n = p.rows();
    auto grid_size = x.size();

    // Evaluate spatial part on the grid 
    std::vector<GridType> wavefunction(n);
    for(decltype(n) i = 0; i < n; ++i){
        wavefunction[i] = real_space_wavefunction(i+1, L, x);
    }

    using value_type = std::decay_t<decltype(p(0, 0))>;
    std::vector<value_type> value(n, 0.0);
    for(decltype(grid_size) k = 0; k < grid_size; ++k){
        for(decltype(n) i = 0; i < n; ++i ){
            auto psi_ik = wavefunction[i][k];
            for(decltype(n) j = 0; j < n; ++j){
                auto psi_jk = wavefunction[j][k];
                value[k] += p(i, j) * psi_ik * psi_jk;
            }
        }
    }
    return value;
}

} // namespace qc101::particle_in_a_box