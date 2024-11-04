#pragma once
#include "types.hpp"
#include <cmath>

namespace qc101::particle_in_a_box{

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

}