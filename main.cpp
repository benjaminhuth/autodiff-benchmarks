#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <mcl/mcl_tabular.hpp>

#include "test_adept.hpp"
#include "test_autodiff.hpp"
#include "test_adolc.hpp"

/*
 *  Benchmarks autodiff libraries by computing 
 *  the jacobian of a generic function:
 *  
 *  | f(x, matrices) 
 *  | {
 *  |     for( mat : matrices )
 *  |         x = mat * x;
 *  |
 *  |     return x;
 *  | }
 *  |
 *  | J = df_i/dx_j
 * 
 * The following parameters are controllable:
 *  - number of matrices
 *  - input size (jacobian cols)
 *  - output size (jacobian rows)
 * 
 */

template<std::size_t JC, std::size_t JR>
void test_configuration(std::size_t nm)
{
    std::vector<Eigen::MatrixXd> jacobians;
    
    std::vector<double> test_times_ms = {
        test_autodiff_forward<JC, JR>(nm, jacobians).count() * 1000.,
        test_autodiff_reverse<JC, JR>(nm, jacobians).count() * 1000.,
        test_adept_forward<JC, JR>(nm, jacobians).count() * 1000.,
        test_adept_reverse<JC, JR>(nm, jacobians).count() * 1000.,
        test_adolc<JC, JR>(nm, jacobians).count() * 1000.
    };
    
    // Do consistency check
    std::vector<std::string> consistencies;
    for(const auto &jac : jacobians)
        if( (jac - jacobians.front()).norm() < 1.e-3 )
            consistencies.push_back("yes");
        else
        {
            consistencies.push_back("NO!");
            std::cout << "\n CONSISTENCY CHECK FAILED, difference matrix:\n" << jac - jacobians.front() << "\n" << std::endl;
        }
    
    // Compute speed factors
    double min_val = *std::min_element(test_times_ms.begin(), test_times_ms.end());
    std::vector<double> time_factors(test_times_ms.size());
    std::transform(test_times_ms.begin(), test_times_ms.end(), time_factors.begin(), [&](const auto &a) { return a / min_val; });
    
    mc::table result_table;
    result_table.create()
    ( "Library",  "Mode",          "Time (ms)",      "Factor to best", "Passed consistency check" )
    ( mc::horizontal_line('=') )
    ( "autodiff", "forward",       test_times_ms[0], time_factors[0],  consistencies[0]           )
    ( "",         "reverse",       test_times_ms[1], time_factors[1],  consistencies[1]           )
    ( mc::horizontal_line('-') )
    ( "adept2", "forward",         test_times_ms[2], time_factors[2],  consistencies[2]           )
    ( "",       "reverse",         test_times_ms[3], time_factors[3],  consistencies[3]           )
    ( mc::horizontal_line('-') )
    ( "ADOL-C", "forward/reverse", test_times_ms[4], time_factors[4],  consistencies[4]           )
    ;
    
    result_table.print();
}

int main() 
{
    std::size_t num_matrices = 5;
    
    // symmetric jacobian
    constexpr std::size_t JacRowsEq = 10; 
    constexpr std::size_t JacColsEq = 10;
    
    std::cout << "\n### TEST ROWS == COLS (" << JacRowsEq << "x" << JacColsEq << ") ###\n" << std::endl;
    test_configuration<JacRowsEq, JacColsEq>(num_matrices);
    std::cout << std::endl;
    
    
    // jacobian rows < cols 
    constexpr std::size_t JacRowsRv = 1; 
    constexpr std::size_t JacColsRv = 10;
    
    std::cout << "\n### TEST ROWS < COLS (" << JacRowsRv << "x" << JacColsRv << ") ### (favours reverse?)\n" << std::endl;
    test_configuration<JacRowsRv, JacColsRv>(num_matrices);
    std::cout << std::endl;
    
    
    // jacobian rows > cols 
    constexpr std::size_t JacRowsFw = 10; 
    constexpr std::size_t JacColsFw = 1;
    
    std::cout << "\n### TEST ROWS > COLS (" << JacRowsFw << "x" << JacColsFw << ") ### (favours forward?)\n" << std::endl;
    test_configuration<JacRowsFw, JacColsFw>(num_matrices);
    std::cout << std::endl;
}
