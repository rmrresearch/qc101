#pragma once
#include "types.hpp"
#include <cmath>
#include <sigma/sigma.hpp>

namespace qc101::particle_in_a_box {

template<typename MatrixType>
struct Density {
    using size_type = std::size_t;

    explicit Density(size_type n) : m_real(n, n), m_imag(n, n) {}

    auto basis_size() const noexcept { return m_real.rows(); }

    MatrixType m_real;
    MatrixType m_imag;
};

/** @brief Propagates the density matrix forward in time by @p dt using the
 *         analytic solution to the time-dependent Schrodinger equation.
 *
 *  @tparam MatrixType The type of the density matrix. Will also be the return
 *                     type.
 *  @tparam VectorType The type of the container holding the energies. Assumed
 *                     to be container-like.
 *
 *  Given the density of the system at time @f$t_0@f$, @p p0, this function will
 *  propagate the density forward in time by @p dt atomic time units according
 *  to the equation:
 *
 *  @f[
 *      P_{mn}(t) = P_{mn}(t_0)e^{-i\left(E_m - E_n\right)\Delta t},
 *  @f]
 *
 *  where @f$E_m@f$ is the energy of the @f$m@f$-th basis function.
 *
 *
 *  @param[in] p0 The density at time @f$t_0@f$.
 *  @param[in] energies The energies of the basis functions.
 *  @param[in] dt How far forward to propagate the wavefunction, in atomic
 *                units.
 *
 *  @return The density matrix at time @f$t=t_0 + \Delta t@f$.
 */
template<typename DensityType, typename VectorType>
auto time_dependent_density(const DensityType& p0, const VectorType& energies,
                            time_type dt) {
    using sigma::cos;
    using sigma::exp;
    using sigma::sin;
    using std::cos;
    using std::exp;
    using std::sin;
    using namespace std::complex_literals;
    auto n          = p0.basis_size();
    using element_t = decltype(p0.m_real(0, 0) * cos(energies[0]));
    using matrix_t  = Eigen::Matrix<element_t, Eigen::Dynamic, Eigen::Dynamic>;
    Density<matrix_t> pt(n);
    for(decltype(n) i = 0; i < n; ++i) {
        for(decltype(n) j = 0; j < n; ++j) {
            auto de = energies[i] - energies[j];
            pt.m_real(i, j) =
              p0.m_real(i, j) * cos(de * dt) + p0.m_imag(i, j) * sin(de * dt);
            pt.m_imag(i, j) =
              p0.m_imag(i, j) * cos(de * dt) - p0.m_real(i, j) * sin(de * dt);
        }
    }
    return pt;
}

/** @brief Evaluates the time-dependent density on a weighted grid.
 *
 *  @param[in] p0 The density to evaluate.
 *  @param[in] grid The weights for the grid points (i.e., the value of the
 *                  basis function at each grid point).
 *
 *  @return The value of the density at the grid points.
 */
template<typename DensityType, typename GridType>
auto real_space_time_dependent_density(const DensityType& p0,
                                       const GridType& grid) {
    using element_t = std::decay_t<decltype(p0.m_real(0, 0) * grid(0, 0))>;
    using vector_t  = Eigen::Vector<element_t, Eigen::Dynamic>;

    auto n_grid  = grid.rows();
    auto n_state = grid.cols();
    // N.b., sum involves c.c., so the imaginary components sum to zero
    vector_t value_real = vector_t::Zero(n_grid);
    for(decltype(n_grid) k = 0; k < n_grid; ++k) {
        for(decltype(n_state) i = 0; i < n_state; ++i) {
            auto x_ki = grid(k, i);
            for(decltype(n_state) j = 0; j < n_state; ++j) {
                auto x_kj = grid(k, j);
                value_real(k) += x_ki * p0.m_real(i, j) * x_kj;
            }
        }
    }
    return value_real;
}

} // namespace qc101::particle_in_a_box