cmake_minimum_required(VERSION 3.14)

## Set Project and Version
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" VERSION)
project(qc101 VERSION "${VERSION}" LANGUAGES CXX)

## Get CMaize
include(cmake/get_cmaize.cmake)

## Paths ##
set(${PROJECT_NAME}_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(${PROJECT_NAME}_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(${PROJECT_NAME}_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")

## Options ##
cmaize_option_list(
    BUILD_TESTING OFF "Should the tests be built?"
    BUILD_PYBIND11_PYBINDINGS ON "Should we build Python bindings?"
)

## Find or build dependencies ##
cmaize_find_or_build_dependency(
    simde
    URL github.com/NWChemEx/SimDE
    VERSION master
    BUILD_TARGET simde
    FIND_TARGET nwx::simde
    CMAKE_ARGS BUILD_TESTING=OFF
)

cmaize_find_or_build_dependency(
    sigma
    URL github.com/QCUncertainty/sigma
    VERSION main
    BUILD_TARGET sigma
    FIND_TARGET sigma::simde
    CMAKE_ARGS BUILD_TESTING=OFF
)

## Add libraries ##
cmaize_add_library(
    ${PROJECT_NAME}
    SOURCE_DIR "${${PROJECT_NAME}_SOURCE_DIR}/${PROJECT_NAME}"
    INCLUDE_DIRS "${${PROJECT_NAME}_INCLUDE_DIR}/${PROJECT_NAME}"
    DEPENDS simde sigma
)

include(nwx_pybind11)
nwx_add_pybind11_module(
    ${PROJECT_NAME}
    SOURCE_DIR "${${PROJECT_NAME}_SOURCE_DIR}/python"
    DEPENDS ${PROJECT_NAME}
)

## Build tests ##
if("${BUILD_TESTING}")
    ## Find or build dependencies for tests
    cmaize_find_or_build_dependency(
        Catch2
        URL github.com/catchorg/Catch2
        BUILD_TARGET Catch2
        FIND_TARGET Catch2::Catch2
        VERSION v2.x
    )

    ## Add Tests ##
    cmaize_add_tests(
        test_${PROJECT_NAME}
        SOURCE_DIR "${${PROJECT_NAME}_TESTS_DIR}/unit_tests"
        INCLUDE_DIRS "${${PROJECT_NAME}_SOURCE_DIR}/${PROJECT_NAME}"
        DEPENDS Catch2 ${PROJECT_NAME}
    )

    nwx_pybind11_tests(
        py_${PROJECT_NAME}
        ${${PROJECT_NAME}_TESTS_DIR}/python/unit_tests/test_qc101.py
        DEPENDS ${PROJECT_NAME}
        SUBMODULES simde pluginplay parallelzone
    )

endif()

## Add package ##
cmaize_add_package(
    ${PROJECT_NAME} NAMESPACE ${PROJECT_NAME}::
)