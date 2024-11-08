#pragma once
#include "types.hpp"

namespace qc101::particle_in_a_box {

/** @brief Propagates the provided wavefunction by @p dt atomic time units using
 *         the analytic solution of the time-dependent Schrodinger equation.
 *
 *  @tparam VectorType The type storing the weights of the input wavefunction.
 *                     Will also be the type of the return. Assumed to be
 *                     container-like and capable of handling complex values.
 *
 *  Given the weights for a superposition of particle-in-a-box (PIB)
 *  wavefunctions at time @f$t_0@f$, this function will compute the weights for
 *  the time-dependent wavefunction at a time @f$\Delta t@f$ atomic time units
 *  later by solving the time-dependent Schrodinger equation. For the PIB, the
 *  analytic solution is known and the @f$m@f$-th component of the wavefunction
 *  at time @f$t=t_0 + \Delta t@f$ is:
 *
 *  @f[
 *     c_m(t) = c_m(t_0) e^{-i E_m \Delta t},
 *  @f]
 *
 *  where @f$E_m@f$ is the energy of the @f$m@f$-th PIB state.
 *
 *  @param[in] c The wavefunction weights at time @f$t_0@f$.
 *  @param[in] energies The energies of the PIB states.
 *  @param[in] dt How far in a.u. to propagate in time.
 *
 *  @return The wavefunction weights at time @f$t=t_0 + \Delta t@f$.
 */
template<typename VectorType>
auto time_dependent_wavefunction(const VectorType& c,
                                 const VectorType& energies, time_type dt) {
    using namespace std::complex_literals;
    auto n = c.size();
    VectorType ct(n);
    for(decltype(n) j = 0; j < n; ++j) {
        ct[j] = std::exp(energies[j] * -1.0i * dt) * c[j];
    }
    return ct;
}

/** @brief Propagates the provided wavefunction by @p dt atomic time units using
 *         Euler's method for numerical integration.
 *
 *  @tparam VectorType The type storing the weights of the input wavefunction.
 *                     Will also be the type of the return. Assumed to be
 *                     container-like.
 *
 *  Given the weights for a superposition of particle-in-a-box (PIB)
 *  wavefunctions at time @f$t_0@f$, this function will compute the weights for
 *  the time-dependent wavefunction at a time @f$\Delta t@f$ atomic time units
 *  later by using Euler's method. According to Euler's method, the @f$m@f$-th
 *  component of the wavefunction at time @f$t=t_0 + \Delta t@f$ is:
 *
 *  @f[
 *     c_m(t) = c_m(t_0) - ic_m(t_0)E_m\Delta t,
 *  @f]
 *
 *  where @f$E_m@f$ is the energy of the @f$m@f$-th PIB state. This is only
 *  valid for small @f$\Delta t@f$
 *
 *  @param[in] c The wavefunction weights at time @f$t_0@f$.
 *  @param[in] energies The energies of the PIB states.
 *  @param[in] dt How far in a.u. to propagate in time.
 *
 *  @return The wavefunction weights at time @f$t=t_0 + \Delta t@f$.
 */
template<typename VectorType>
auto euler_step(const VectorType& c, const VectorType& egy, time_type dt) {
    using namespace std::complex_literals;
    auto n = c.size();
    VectorType ct(n);
    for(decltype(n) j = 0; j < n; ++j) {
        ct[j] = c[j] - 1.0i * c[j] * egy[j] * dt;
    }
    return ct;
}

/** @brief Computes the value of the spatial wavefunction on a real-space grid.
 *
 *  @tparam VectorType The type of the container specifying the grid points.
 *                     Assumed to be container-like. Will also be the return
 *                     type.
 *
 *  In real-space, the spatial wavefunction for the @f$n@f$-th state of a
 *  particle-in-a-box (PIB) of length @f$L@f$ is given by:
 *
 *  @f[
 *    \psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right).
 *  @f]
 *
 *  This function will evaluate the first @p max_n wavefunctions at the real-
 *  space points specified by @p x.
 *
 *  @param[in] max_n The number of quantum states to evaluate. Note that max_n
 *  @param[in] L The length of the box in a.u.
 *  @param[in] x The real-space values at which to evaluate the wavefunction.
 *
 *  @return A "number of grid points" by "number of quantum states" matrix where
 *          the i,j-th element is the value of j-th wavefunction evaluated on
 *          the i-th grid point.
 *
 */
template<typename VectorType>
auto real_space_wavefunction(quantum_number_type max_n, distance_type L,
                             const VectorType& x) {
    distance_type prefactor = std::sqrt(2.0 / L);
    auto pi                 = 2 * std::acos(0.0);
    auto grid_size          = x.size();
    if(grid_size == 0 || max_n == 0) return Eigen::MatrixXd{};

    Eigen::MatrixXd value(grid_size, max_n);
    for(decltype(max_n) n = 0; n < max_n; ++n) {
        for(decltype(grid_size) k = 0; k < grid_size; ++k) {
            value(k, n) = std::sin((n + 1) * pi * x[k] / L) * prefactor;
        }
    }
    return value;
}

} // namespace qc101::particle_in_a_box