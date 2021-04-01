#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <cstdlib>
#include <vector>

#include <Eigen/Core>

// Time
using duration_t = std::chrono::duration<double>;

// Global switch to control printing
constexpr static bool print_jacobians = true;

// Eigen typedefs
template<typename T>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template<typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;


// generic test function
template<typename InVec, typename MatInIn, typename MatInOut, typename Fmatmul>
inline auto test_function(const InVec &x, const std::tuple<std::vector<MatInIn>, MatInOut> &params, const Fmatmul &matmul)
{
    InVec ret = x;
    
    for(std::size_t i=0; i<std::get<0>(params).size(); ++i)
    {
        ret = matmul(std::get<0>(params)[i], ret);
    }
    
    return matmul(std::get<1>(params), ret);
}


// initialize random things on Eigen basis
template<typename T, std::size_t InputSize, std::size_t OutputSize>
auto initialize_params(const std::size_t n_param_matrices)
{
    using Vector = Eigen::Matrix<T, InputSize, 1>;
    using ParMatrix = Eigen::Matrix<T, InputSize, InputSize>;
    using OutMatrix = Eigen::Matrix<T, OutputSize, InputSize>;
    
    // ensure same numbers in each test
    std::srand(42);
    
    Vector x = Vector::Random();
    
    std::vector<ParMatrix> param_vec;
    
    for(std::size_t i=0ul; i<n_param_matrices-1; ++i)
        param_vec.push_back( ParMatrix::Random() );
    
    OutMatrix out = OutMatrix::Random();
    
    return std::make_pair(x, std::make_tuple(param_vec, out));    
}


// helper enum for compile time branches
enum class modes { forward, reverse };


// eigen matmul
struct EigenMatmul
{
    template<typename T, int InDim, int OutDim, int SameDim>
    auto operator()(const Eigen::Matrix<T, InDim, SameDim> &a, const Eigen::Matrix<T, SameDim, OutDim> &b) const
        -> Eigen::Matrix<T, OutDim, InDim>
    {
        return a * b;
    }
};

#endif
