#pragma once

#include "common.hpp"

#include <adept_arrays.h>


auto adept_matmul = [](const auto &vec, const auto &mat){ return vec ** mat; };
using adept_matmul_t = decltype(adept_matmul);

namespace 
{
    template<modes m, std::size_t InputSize, std::size_t OutputSize>
    duration_t test_adept(std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
    {
        auto [x_double, params_double] = initialize_params<double, InputSize, OutputSize>(n_param_matrices);
        
        // must be at the beginning
        adept::Stack stack;
        stack.set_max_jacobian_threads(1);

        // initialize objects 
        adept::aVector x(InputSize);
        std::vector<adept::aMatrix> params(n_param_matrices);
        
        // copy random data to adept types
        for(std::size_t i=0; i<x_double.size(); ++i)
            x(i) = x_double(i);
        
        for(std::size_t n=0; n<n_param_matrices; ++n)
        {
            const auto &src_mat = std::get<0>(params_double)[n];
            params[n].resize(src_mat.rows(), src_mat.cols());
            
            for(std::size_t i=0; i<src_mat.rows(); ++i)
                for(std::size_t j=0; j<src_mat.cols(); ++j)
                    params[n](i,j) = src_mat(i,j);
        }
        
        const auto &src_mat = std::get<1>(params_double);
        adept::aMatrix last_param;
        last_param.resize(src_mat.rows(), src_mat.cols());
                    
        for(std::size_t i=0; i<src_mat.rows(); ++i)
            for(std::size_t j=0; j<src_mat.cols(); ++j)
                last_param(i,j) = src_mat(i,j);
            
        const auto adept_params = std::make_tuple(params, last_param);
        
        // Allocate jacobian
        Eigen::MatrixXd jac(OutputSize, InputSize);
        
        // run test function
        auto t0 = std::chrono::high_resolution_clock::now();
        
        stack.new_recording();
        
        adept::aVector y = test_function(x, adept_params, adept_matmul);
        
        std::cout << "y = " << y << std::endl;
        
        stack.dependent(y);
        stack.independent(x);
        
        if constexpr( m == modes::forward )
            stack.jacobian_forward(jac.data());
        else
            stack.jacobian_reverse(jac.data());
        
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // prevent optimizations
        if( jac.size() == 0 )
            throw;
        
        if constexpr( print_jacobians )
            std::cout << "adept:\n" << jac << "\n" << std::endl;
        
        js.push_back(jac);
        
        std::exit(0);
        
        return std::chrono::duration<double>(t1 - t0);
    }
}

template<std::size_t InputSize, std::size_t OutputSize>
duration_t test_adept_forward(std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
{
    return test_adept<modes::forward, InputSize, OutputSize>(n_param_matrices, js);
}

template<std::size_t InputSize, std::size_t OutputSize>
duration_t test_adept_reverse(std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
{
    return test_adept<modes::reverse, InputSize, OutputSize>(n_param_matrices, js);
}
