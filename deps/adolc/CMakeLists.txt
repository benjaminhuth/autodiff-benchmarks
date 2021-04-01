cmake_minimum_required(VERSION 3.0)

project(adolc)

include(ExternalProject)

ExternalProject_Add(adolc
    PREFIX ${CMAKE_BINARY_DIR}
    GIT_REPOSITORY "https://github.com/coin-or/ADOL-C.git"
    CONFIGURE_COMMAND ../adolc/configure --prefix=${CMAKE_BINARY_DIR} CXXFLAGS=-O3
    BUILD_COMMAND make -j16
    UPDATE_COMMAND make -j16
    INSTALL_COMMAND make install
)
