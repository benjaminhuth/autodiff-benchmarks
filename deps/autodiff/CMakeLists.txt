cmake_minimum_required(VERSION 3.0)

project(autodiff)

include(ExternalProject)

ExternalProject_Add(autodiff
    PREFIX ${CMAKE_BINARY_DIR}/autodiff
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    GIT_REPOSITORY "https://github.com/autodiff/autodiff.git"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

