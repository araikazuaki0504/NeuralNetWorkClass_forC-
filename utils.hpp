#pragma once

#include <vector>
#include <limits>
#include <cmath>
#include <string>

enum ActivationType
{
    Sigmoid,
    Linear,
    SoftMax,
    ReLu
};

enum ErrorFunctionType
{
    Cen,
    Mse
};

namespace ErrorFunction
{
    long double mean_squared_error(std::vector<std::vector<long double>> &y, std::vector<std::vector<long double>> &t);
    long double mean_cross_entropy_error(std::vector<std::vector<long double>> &y, std::vector<std::vector<long double>> &t);
    std::vector<std::vector<long double>> mean_squared_error_back(std::vector<std::vector<long double>>absolute_error);
    std::vector<std::vector<long double>> mean_cross_entropy_error_back(std::vector<std::vector<long double>>error);
}

namespace std
{
    std::string to_string(ActivationType _Val);
    std::string to_string(ErrorFunctionType _Val);
}