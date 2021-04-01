#pragma once

#include <adolc/adolc.h>

#include "common.hpp"

template<typename T, int R, int C>
struct RowMajorIfPossible
{
    using type = Eigen::Matrix<double, R, C, Eigen::RowMajorBit>;
};

template<typename T, int R>
struct RowMajorIfPossible<T, R, 1>
{
    using type = Eigen::Matrix<double, R, 1>;
};


template<std::size_t InputSize, std::size_t OutputSize>
duration_t test_adolc(std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
{
    constexpr int tape = 0;

    auto [x_double, params_double] = initialize_params<double, InputSize, OutputSize>(n_param_matrices);
    Eigen::Matrix<double, OutputSize, 1> y_double;
    
    // need c-style jacobian, Eigen uses column-major by default
    typename RowMajorIfPossible<double, OutputSize, InputSize>::type jac;
    std::array<double *, OutputSize> jac_ptrs;
    
    for(std::size_t i=0; i<jac.rows(); ++i)
        jac_ptrs[i] = jac.row(i).data();
    
    // start recording
    trace_on(tape);
    
    // initialize ADOL-C types
    Eigen::Matrix<adouble, InputSize, 1> x;
        
    for(std::size_t i=0; i<x_double.size(); ++i)
        x(i) <<= x_double(i);
    
    std::vector<Eigen::Matrix<adouble, InputSize, InputSize>> params(n_param_matrices);
    
    for(std::size_t n=0; n<n_param_matrices; ++n)
    {
        const auto &src_mat = std::get<0>(params_double)[n];
        
        for(std::size_t i=0; i<src_mat.rows(); ++i)
            for(std::size_t j=0; j<src_mat.cols(); ++j)
                params[n](i,j) = src_mat(i,j);
    }   
    
    Eigen::Matrix<adouble, OutputSize, InputSize> last_param;
    last_param.resize(std::get<1>(params_double).rows(), std::get<1>(params_double).cols());
    
    for(std::size_t i=0; i<std::get<1>(params_double).rows(); ++i)
        for(std::size_t j=0; j<std::get<1>(params_double).cols(); ++j)
            last_param(i,j) = std::get<1>(params_double)(i,j);
    
    // run function
    auto t0 = std::chrono::high_resolution_clock::now();
    
    auto y = test_function(x, std::make_tuple(params, last_param), EigenMatmul{});
    
    for(std::size_t i=0; i<OutputSize; ++i)
        y(i) >>= y_double(i);
    
    trace_off();
    
    jacobian(tape, y.size(), x.size(), x_double.data(), jac_ptrs.data());
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // prevent optimizations
    if( jac.size() == 0 )
        throw;
    
    if constexpr( print_jacobians )
        std::cout << "ADOL-C:\n" << jac << "\n" << std::endl;
    
    // convert row-major to col-major (is there a smarter way?)
    Eigen::MatrixXd jac_colmajor(jac.rows(), jac.cols());
    for(std::size_t i=0; i<jac.rows(); ++i)
        for(std::size_t j=0; j<jac.cols(); ++j)
            jac_colmajor(i,j) = jac(i,j);
        
    js.push_back(jac_colmajor);
    
    return std::chrono::duration<double>(t1 - t0);
}
