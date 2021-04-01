cmake_minimum_required(VERSION 3.0)
include(ExternalProject)

## Build type
# if(NOT ${CMAKE_BUILD_TYPE})
#   set(${CMAKE_BUILD_TYPE} Release)
# endif()

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

#######################
## THE AUTODIFF LIBS ##
#######################

macro(make_dependency name)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/deps/${name})

    execute_process(
        COMMAND ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR}/deps/${name}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/deps/${name})

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/deps/${name})

    list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/deps/${name})
endmacro()

# Adol-C #
make_dependency(adolc)
add_library(adolc SHARED IMPORTED)
set_target_properties(adolc PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/deps/adolc/lib64/libadolc.so)
set_target_properties(adolc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR}/deps/adolc/include)

# autodiff #
make_dependency(autodiff)
find_package(autodiff REQUIRED)

# adept2 #
make_dependency(adept2)
add_library(adept2 STATIC IMPORTED)
set_target_properties(adept2 PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/deps/adept2/lib/libadept.a)
set_target_properties(adept2 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR}/deps/adept2/include)

# Executable #
set(SOURCES main.cpp test_adept.cpp  test_adolc.cpp  test_autodiff.cpp)

add_executable(autodiff_benchmark ${SOURCES})

# Link #
target_link_libraries(autodiff_benchmark BLAS::BLAS)
target_link_libraries(autodiff_benchmark OpenMP::OpenMP_CXX)
target_link_libraries(autodiff_benchmark Eigen3::Eigen)
target_link_libraries(autodiff_benchmark autodiff::autodiff)
target_link_libraries(autodiff_benchmark adolc)
target_link_libraries(autodiff_benchmark adept2)


