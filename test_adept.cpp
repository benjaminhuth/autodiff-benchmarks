#include "tests.hpp"

#include <adept_arrays.h>


auto adept_matmul = [](auto vec, auto mat){ return vec ** mat; };
using adept_matmul_t = decltype(adept_matmul);

namespace 
{
    template<modes m>
    duration_t test_adept(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
    {
        auto [x_double, params_double] = initialize_params<double>(input_size, output_size, n_param_matrices);
        
        // must be at the beginning
        adept::Stack stack;
        stack.set_max_jacobian_threads(1);

        // initialize objects 
        adept::aVector x(input_size);
        std::vector<adept::aMatrix> params(n_param_matrices);
        Eigen::MatrixXd jac(output_size, input_size);
        
        // copy random data to adept types
        for(std::size_t i=0; i<x_double.size(); ++i)
            x(i) = x_double(i);
        
        for(std::size_t n=0; n<n_param_matrices; ++n)
        {
            const auto &src_mat = params_double[n];
            params[n].resize(src_mat.rows(), src_mat.cols());
            
            for(std::size_t i=0; i<src_mat.rows(); ++i)
                for(std::size_t j=0; j<src_mat.cols(); ++j)
                    params[n](i,j) = src_mat(i,j);
        }
            
        
        // run test function
        auto t0 = std::chrono::high_resolution_clock::now();
        
        stack.new_recording();
        
        adept::aVector y = test_function(x, params, adept_matmul);
        
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
        
        return std::chrono::duration<double>(t1 - t0);
    }
}

duration_t test_adept_forward(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
{
    return test_adept<modes::forward>(input_size, output_size, n_param_matrices, js);
}

duration_t test_adept_reverse(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
{
    return test_adept<modes::reverse>(input_size, output_size, n_param_matrices, js);
}
