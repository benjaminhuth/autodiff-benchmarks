cmake_minimum_required(VERSION 3.0)

project(autodiff_performance_overview)

add_executable(autodiff_performance_overview main.cpp)

install(TARGETS autodiff_performance_overview RUNTIME DESTINATION bin)
