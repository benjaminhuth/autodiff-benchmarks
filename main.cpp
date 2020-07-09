#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <mcl/mcl_tabular.hpp>

#include "tests.hpp"

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

void test_configuration(std::size_t nm, std::size_t jc, std::size_t jr)
{
    std::vector<Eigen::MatrixXd> jacobians;
    
    std::vector<double> test_times_ms = {
        test_autodiff_forward(jc,jr,nm, jacobians).count() * 1000.,
        test_autodiff_reverse(jc,jr,nm, jacobians).count() * 1000.,
        test_adept_forward(jc,jr,nm, jacobians).count() * 1000.,
        test_adept_reverse(jc,jr,nm, jacobians).count() * 1000.,
        test_adolc(jc,jr,nm, jacobians).count() * 1000.
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
    std::size_t jac_rows = 10; 
    std::size_t jac_cols = 8;
    
    std::cout << "\n### TEST ROWS == COLS (" << jac_rows << "x" << jac_cols << ") ###\n" << std::endl;
    test_configuration(num_matrices, jac_cols, jac_rows);
    std::cout << std::endl;
    
    
    // jacobian rows < cols 
    jac_rows = 1;
    jac_cols = 10;
    
    std::cout << "\n### TEST ROWS < COLS (" << jac_rows << "x" << jac_cols << ") ### (favours reverse?)\n" << std::endl;
    test_configuration(num_matrices, jac_cols, jac_rows);
    std::cout << std::endl;
    
    
    // jacobian rows > cols 
    jac_rows = 10;
    jac_cols = 1;
    
    std::cout << "\n### TEST ROWS > COLS (" << jac_rows << "x" << jac_cols << ") ### (favours forward?)\n" << std::endl;
    test_configuration(num_matrices, jac_cols, jac_rows);
    std::cout << std::endl;
}
