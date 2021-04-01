cmake_minimum_required(VERSION 3.0)

project(adept2)

include(ExternalProject)

ExternalProject_Add(adept2
    PREFIX ${CMAKE_BINARY_DIR}
    BUILD_IN_SOURCE ON
    GIT_REPOSITORY "https://github.com/rjhogan/Adept-2.git"
    CONFIGURE_COMMAND autoreconf -i && ./configure --prefix=${CMAKE_BINARY_DIR} CXXFLAGS=-O3
    BUILD_COMMAND make -j16
    UPDATE_COMMAND make -j16
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/lib/libadept.a
)
