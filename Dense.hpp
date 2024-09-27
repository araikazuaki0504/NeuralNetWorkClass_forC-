#pragma once

#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "ActivationFunction.hpp"

class Dense
{
    friend class Model;

    public:
        Dense(int input_qty, int output_qty, ActivationType typeName, std::string prefix = "", std::string neuron_coeff_Path = "", std::string bias_coeff_Path = "");
        ~Dense();

    private:
        int _input_qty;
        int _output_qty;
        std::string _neuron_coeff_Path;
        std::string _bias_coeff_Path;
        std::string _file_prefix;
        ActivationType _typeName;
        ActivationFunction _activation;
        std::vector<long double> _bias;
        std::vector<std::vector<long double>> _neuron;
        std::vector<std::vector<long double>> _last_data;
        std::vector<std::vector<long double>> _grad_layer;
        std::vector<long double> _grad_bias;

        std::vector<std::vector<long double>> forward(std::vector<std::vector<long double>>& data);
        std::vector<std::vector<long double>> backward(std::vector<std::vector<long double>>& data);

};
