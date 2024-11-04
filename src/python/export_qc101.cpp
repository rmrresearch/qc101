#include <qc101/qc101.hpp>
#include <simde/simde.hpp>
#include "particle_in_a_box/export_particle_in_a_box.hpp"

namespace qc101 {

EXPORT_PLUGIN(qc101, m) {
    particle_in_a_box::export_modules(m);
}

} // namespace qc101