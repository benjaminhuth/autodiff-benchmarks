#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <cstdlib>
#include <vector>

#include <Eigen/Core>

// Global switch to control printing
constexpr static bool print_jacobians = false;

// Eigen typedefs
template<typename T>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template<typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;


// generic test function
template<typename Tval, typename Tparam, typename Fmatmul>
inline Tval test_function(const Tval &x, const std::vector<Tparam> &params, const Fmatmul &matmul)
{
    Tval ret = x;
    
    for(std::size_t i=0; i<params.size()-1; ++i)
    {
        ret = matmul(params[i], ret);
    }
    
    return matmul(params.back(), ret);
}


// initialize random things on Eigen basis
template<typename T>
auto initialize_params(const std::size_t input_size, const std::size_t output_size, const std::size_t n_param_matrices)
{
    // ensure same numbers in each test
    std::srand(42);
    
    VectorX<T> x = VectorX<T>::Random(input_size);
    
    std::vector<MatrixX<T>> params;
    
    for(std::size_t i=0ul; i<n_param_matrices-1; ++i)
        params.push_back( MatrixX<T>::Random(input_size, input_size) );
    
    params.push_back( MatrixX<T>::Random(output_size, input_size) );
    
    return std::make_pair(x, params);    
}


// helper enum for compile time branches
enum class modes { forward, reverse };


// eigen matmul
static auto eigen_matmul = [](auto vec, auto mat){ return vec * mat; };
using eigen_matmul_t = decltype(eigen_matmul);

#endif
