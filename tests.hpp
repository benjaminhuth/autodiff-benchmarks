#ifndef TESTS_HPP
#define TESTS_HPP

#include <chrono>
#include <iostream>

#include "common.hpp"

using duration_t = std::chrono::duration<double>;

duration_t test_autodiff_forward(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js);
duration_t test_autodiff_reverse(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js);

duration_t test_adept_forward(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js);
duration_t test_adept_reverse(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js);

// internally chooses the best mode
duration_t test_adolc(std::size_t input_size, std::size_t output_size, std::size_t n_param_matrices, std::vector<Eigen::MatrixXd> &js);

#endif
