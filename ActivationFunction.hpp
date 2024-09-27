#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "utils.hpp"

class ActivationFunction
{
    public:
        ActivationFunction(ActivationType TypeName);
        ~ActivationFunction();
        std::vector<std::vector<long double>>forward(std::vector<std::vector<long double>> &x);
        std::vector<std::vector<long double>>backward(std::vector<std::vector<long double>> &x);

    private:
        ActivationType _TypeName;
        std::vector<std::vector<long double>> _last_data;

        std::vector<std::vector<long double>>Sigmoid(std::vector<std::vector<long double>> &x);
        std::vector<std::vector<long double>>ReLu(std::vector<std::vector<long double>> &x);
        std::vector<std::vector<long double>>SoftMax(std::vector<std::vector<long double>> &x);
        std::vector<std::vector<long double>>Sigmoid_back(std::vector<std::vector<long double>> &x);
        std::vector<std::vector<long double>>ReLu_back(std::vector<std::vector<long double>> &x);
        std::vector<std::vector<long double>>SoftMax_back(std::vector<std::vector<long double>> &x);
};
