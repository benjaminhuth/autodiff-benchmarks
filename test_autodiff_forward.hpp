#ifndef AUTODIFF_FORWARD_HPP
#define AUTODIFF_FORWARD_HPP

#include <chrono>

#include <Eigen/Core>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#include "test_function.hpp"

auto test_autodiff_forward(const std::size_t input_size, const std::size_t output_size, const std::size_t n_param_matrices)
{
    // Random input vector
    VectorX<autodiff::dual> x = VectorX<autodiff::dual>::Random(input_size);
    
    // Random parameter matrices
    std::vector<MatrixX<autodiff::dual>> params;
    
    for(std::size_t i=0ul; i<n_param_matrices-1; ++i)
        params.push_back( MatrixX<autodiff::dual>::Random(input_size, input_size) );
    
    params.push_back( MatrixX<autodiff::dual>::Random(output_size, input_size) );
    
    // run test_function
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = test_function<autodiff::dual>(x, params);
    auto t1 = std::chrono::high_resolution_clock::now();
    
    std::cout << "Run time: " << std::chrono::duration<double>(t1 - t0).count() << std::endl;
    
    t0 = std::chrono::high_resolution_clock::now();
    auto J = autodiff::forward::jacobian(test_function<autodiff::dual>, wrt(x), at(x, params), result);
    t1 = std::chrono::high_resolution_clock::now();
    
    std::cout << "Jacobian time: " << std::chrono::duration<double>(t1 - t0).count() << std::endl;
}

#endif
