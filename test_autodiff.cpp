#include "tests.hpp"

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>


// Helper traits
template<modes m> struct autodiff_type {};
template<> struct autodiff_type<modes::forward> { using type = autodiff::dual; };
template<> struct autodiff_type<modes::reverse> { using type = autodiff::var; };

namespace 
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
    template<modes m>
    auto test_autodiff(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
    {    
        using MatrixX = Eigen::Matrix<typename autodiff_type<m>::type, Eigen::Dynamic, Eigen::Dynamic>;
        
        // Random input vector and matrices
        auto [x, params] = initialize_params<typename autodiff_type<m>::type>(input_size, output_size, n_param_matrices);
        MatrixX jac;
        
        // run test_function    
        auto t0 = std::chrono::high_resolution_clock::now();
        
        if constexpr( m == modes::forward )
        {
            jac = autodiff::forward::jacobian(test_function<autodiff::VectorXdual, autodiff::MatrixXdual, eigen_matmul_t>, 
                                            wrt(x), at(x, params, eigen_matmul));
        }
        else
        {
            auto y = test_function(x, params, eigen_matmul);
            jac = autodiff_reverse_jacobian(y, x);
        }
        
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

duration_t test_autodiff_forward(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
{
    return test_autodiff<modes::forward>(input_size, output_size, n_param_matrices, js);
}

duration_t test_autodiff_reverse(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
{
    return test_autodiff<modes::reverse>(input_size, output_size, n_param_matrices, js);
}
