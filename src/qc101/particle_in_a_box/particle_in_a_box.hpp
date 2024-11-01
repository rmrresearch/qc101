#include <Eigen/Dense>
#include <cmath>
#include <complex>

namespace qc101::particle_in_a_box {

using energy_type = double;
using mass_type = double;
using distance_type = double;
using time_type = double;
using quantum_number_type = unsigned short;

/** @brief Computes the energy of a particle in a box.
 *
 *  In atomic units (a.u.), the @f$n@f$-th energy level of a particle of mass
 *  @f$m@f$ in a box of length @f$L@f$ is given by:
 *
 *  @f[
 *      E_n = \frac{n^2 \pi^2}{2mL^2}
 *  @f]
 *
 *  @param[in] n The quantum number of the state.
 *  @param[in] m The mass of the particle, in a.u.
 *  @param[in] L The length of the box, in a.u.
 *
 *  @return The energy of the @p n-th state for a particle of mass @p m in a box
 *          of length @p L.
 *
 *  @throw None No throw guarantee.
 */
inline energy_type energy(quantum_number_type n, mass_type m, distance_type L) {
  auto pi = 2 * std::acos(0.0);
  return n * n * pi * pi / (2 * m * L * L);
}

/** @brief Computes the time-dependent wavefunction at time @p t.
 * 
 *  Given the weights for a superposition of particle-in-a-box 
 *  wavefunctions at time @f$t_0@f$, this function will compute the weights at 
 *  the time @f$t=t_0 + \Delta t@f$. The @f$m@f$-th component of the 
 *  new wavefunction is given by:
 * 
 *  @f[
 *     c_m(t) = c_m(t_0) e^{-i E_m \Delta t},
 *  @f]
 * 
 *  where @f$E_m@f$ is the energy of the @f$m@f$-th particle-in-a-box state.
 * 
 *  @param[in] c The wavefunction weights at time @f$t_0@f$.
 *  @param[in] energies The energies of the PIB states.
 *  @param[in] dt How far in a.u. to propagate in time.
 * 
 *  @return The wavefunction weights at time @f$t=t_0 + \Delta t@f$.
 */
template <typename VectorType>
inline auto time_dependent_wavefunction(const VectorType &c, 
 const VectorType& energies, time_type dt) {
  using namespace std::complex_literals;
  auto n = c.size();
  VectorType ct(n);
  for (decltype(n) j = 0; j < n; ++j) {
    ct[j] = c[j] * std::exp(-1.0i * energies[j] * dt);
  }
  return ct;
}

template <typename VectorType>
inline auto euler_step(const VectorType &c, const VectorType& egy, time_type dt) {
  using namespace std::complex_literals;
  auto n = c.size();
  VectorType ct(n);
  for (decltype(n) j = 0; j < n; ++j) {
    ct[j] = c[j] - 1.0i * c[j] * egy[j] * dt;
  }
  return ct;
}

} // namespace qc101::particle_in_a_box