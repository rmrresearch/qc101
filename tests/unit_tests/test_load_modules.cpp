#include <catch2/catch.hpp>
#include <qc101/qc101.hpp>
#include <simde/simde.hpp>

TEST_CASE("load_modules") {
  pluginplay::ModuleManager mm;
  qc101::load_modules(mm);
  chemist::ChemicalSystem sys;
}
