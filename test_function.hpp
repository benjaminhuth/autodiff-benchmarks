#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <Eigen/Core>
#include <vector>
#include <iostream>

template<typename T>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template<typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template<typename Tval, typename Tparam = Tval>
VectorX<Tval> test_function(const VectorX<Tval> &x, const std::vector<MatrixX<Tparam>> &params)
{
    VectorX<Tval> ret = x;
    
    for(const auto &mat : params)
        ret = mat * ret;
    
    return ret;
}

#endif
