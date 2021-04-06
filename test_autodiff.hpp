#pragma once

#include <iostream>

#include "common.hpp"

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>


// Helper traits
template<modes m> struct autodiff_type {};
template<> struct autodiff_type<modes::forward> { using type = autodiff::dual; };
template<> struct autodiff_type<modes::reverse> { using type = autodiff::var; };

namespace detail
{
    // Helper function to compute reverse jacobian
    template<int Fdim, int Xdim>
    auto autodiff_reverse_jacobian(Eigen::Matrix<autodiff::var, Fdim, 1> &f,
                                   Eigen::Matrix<autodiff::var, Xdim, 1> &x)
    {
        Eigen::Matrix<autodiff::var, Fdim, Xdim> j;
        
        std::size_t rows = Fdim;
        
        if constexpr( Fdim == -1 || Xdim == -1 )
        {
            j.resize(f.size(), x.size());
            rows = f.size();
        }
        
        for(std::size_t i=0; i<rows; ++i)
            j.row(i) = autodiff::gradient(f(i), x);
        
        return j;
    }


    // Actual test function
    template<modes m, std::size_t InputSize, std::size_t OutputSize>
    auto test_autodiff(std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
    {    
        using AutodiffType = typename autodiff_type<m>::type;
        using InVector = Eigen::Matrix<AutodiffType, InputSize, 1>;
        using InInMatrix = Eigen::Matrix<AutodiffType, InputSize, InputSize>; 
        using OutInMatrix = Eigen::Matrix<AutodiffType, OutputSize, InputSize>; 
        using OutVector = Eigen::Matrix<AutodiffType, OutputSize, 1>; 
        using MatrixX = Eigen::Matrix<AutodiffType, Eigen::Dynamic, Eigen::Dynamic>;
        
        // Random input vector and matrices
        auto [x, params] = initialize_params<AutodiffType, InputSize, OutputSize>(n_param_matrices);
        
        // Result and jacobian
        MatrixX jac;
        OutVector y;
        
        // Run test_function    
        auto t0 = std::chrono::high_resolution_clock::now();
        
        if constexpr( m == modes::forward )
        {
            jac = autodiff::forward::jacobian([](const auto &x, const auto &params){ return test_function(x, params, EigenMatmul{}); },
                                              wrt(x), at(x, params), y);
        }
        else
        {
            y = test_function(x, params, EigenMatmul{});
            jac = autodiff_reverse_jacobian(y, x);
        }
        
        std::cout << "y = " << y.transpose() << std::endl;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // prevent optimization
        if( jac.size() == 0)
            throw;
        
        if constexpr( print_jacobians )
            std::cout << "autodiff:\n" << jac << "\n" << std::endl;
        
        js.push_back(jac.template cast<double>());
        
        return std::chrono::duration<double>(t1 - t0);
    }
}

template<std::size_t InputSize, std::size_t OutputSize>
duration_t test_autodiff_forward(std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
{
    return detail::test_autodiff<modes::forward, InputSize, OutputSize>(n_param_matrices, js);
}

template<std::size_t InputSize, std::size_t OutputSize>
duration_t test_autodiff_reverse(std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
{
    return detail::test_autodiff<modes::reverse, InputSize, OutputSize>(n_param_matrices, js);
}
