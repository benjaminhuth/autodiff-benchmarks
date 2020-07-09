cmake_minimum_required(VERSION 3.0)
include(ExternalProject)

## Build type
# if(NOT ${CMAKE_BUILD_TYPE})
#   set(${CMAKE_BUILD_TYPE} Release)
# endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING "Will compile in '${CMAKE_BUILD_TYPE}' mode. The Performance may be not optimal. Consider compiling in 'Release' mode")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(autodiff_performance_overview)

set(CMAKE_CXX_STANDARD 17)
set(JOBS 20)

############################
## REQUIRED EXTERNAL LIBS ##
############################

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# mcl for tablular printing
set(MCL_ROOT "${CMAKE_BINARY_DIR}/deps/mcl")

ExternalProject_Add(mcl
    PREFIX ${MCL_ROOT}
    GIT_REPOSITORY "https://github.com/benjaminhuth/mcl"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)


#######################
## THE AUTODIFF LIBS ##
#######################

# autodiff
set(AUTODIFF_ROOT "${CMAKE_BINARY_DIR}/deps/autodiff")

ExternalProject_Add(autodiff
    PREFIX ${AUTODIFF_ROOT}
    GIT_REPOSITORY "https://github.com/autodiff/autodiff.git"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

# Adept 2
set(ADEPT2_ROOT "${CMAKE_BINARY_DIR}/deps/adept2")

ExternalProject_Add(adept2
    PREFIX ${ADEPT2_ROOT}
    GIT_REPOSITORY "https://github.com/rjhogan/Adept-2.git"
    BINARY_DIR "${ADEPT2_ROOT}/src/adept2"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND autoreconf -i && ./configure --prefix=<INSTALL_DIR> CXXFLAGS=-O3
    BUILD_COMMAND make -j${JOBS}
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libadept.a
)

add_library(libadept STATIC IMPORTED GLOBAL)
set_target_properties(libadept PROPERTIES IMPORTED_LOCATION "${ADEPT2_ROOT}/lib/libadept.a")

add_dependencies(libadept adept2 BLAS::BLAS OpenMP::OpenMP_CXX)

# ADOL-C
set(ADOLC_ROOT "${CMAKE_BINARY_DIR}/deps/ADOL-C")

ExternalProject_Add(adolc
    PREFIX ${ADOLC_ROOT}
    GIT_REPOSITORY "https://github.com/coin-or/ADOL-C.git"
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ${ADOLC_ROOT}/src/adolc/configure --prefix=${ADOLC_ROOT} CXXFLAGS=-O3
    BUILD_COMMAND make -j${JOBS}
    INSTALL_COMMAND make install
)

add_library(libadolc SHARED IMPORTED GLOBAL)
set_target_properties(libadolc PROPERTIES IMPORTED_LOCATION "${ADOLC_ROOT}/lib64/libadolc.so")

add_dependencies(libadolc adolc)


##########################
## THE ACTUAL BENCHMARK ##
##########################

# Includes #
include_directories("${MCL_ROOT}/src/mcl")
include_directories("${AUTODIFF_ROOT}/src/autodiff/autodiff")
include_directories("${ADEPT2_ROOT}/include")
include_directories("${ADOLC_ROOT}/include")

# Executable #
set(SOURCES main.cpp test_autodiff.cpp test_adept.cpp test_adolc.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})
add_dependencies(${PROJECT_NAME} libadept)
add_dependencies(${PROJECT_NAME} autodiff)
add_dependencies(${PROJECT_NAME} mcl)
add_dependencies(${PROJECT_NAME} libadolc)

# Link #
target_link_libraries(${PROJECT_NAME} libadept)
target_link_libraries(${PROJECT_NAME} libadolc)
target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenMP_CXX_LIBRARIES})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
