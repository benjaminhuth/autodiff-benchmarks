#include <adolc/adolc.h>

#include "tests.hpp"

duration_t test_adolc(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js)
{
    constexpr int tape = 0;

    auto [x_double, params_double] = initialize_params<double>(input_size, output_size, n_param_matrices);
    VectorX<double> y_double(output_size);
    
    // need c-style jacobian, Eigen uses column-major by default
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> jac(output_size, input_size);
    std::vector<double *> jac_ptrs(output_size);
    
    for(std::size_t i=0; i<jac.rows(); ++i)
        jac_ptrs[i] = jac.row(i).data();
    
    // start recording
    trace_on(tape);
    
    // initialize ADOL-C types
    VectorX<adouble> x;
    x.resize(input_size);
        
    for(std::size_t i=0; i<x_double.size(); ++i)
        x(i) <<= x_double(i);
    
    std::vector<MatrixX<adouble>> params(n_param_matrices);
    
    for(std::size_t n=0; n<n_param_matrices; ++n)
    {
        const auto &src_mat = params_double[n];
        params[n].resize(src_mat.rows(), src_mat.cols());
        
        for(std::size_t i=0; i<src_mat.rows(); ++i)
            for(std::size_t j=0; j<src_mat.cols(); ++j)
                params[n](i,j) = src_mat(i,j);
    }   
    
    // run function
    auto t0 = std::chrono::high_resolution_clock::now();
    
    auto y = test_function(x, params, eigen_matmul);
    
    for(std::size_t i=0; i<output_size; ++i)
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
