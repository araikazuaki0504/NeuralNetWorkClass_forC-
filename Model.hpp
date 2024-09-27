#pragma once

#include "Dense.hpp"

class Model
{
    public:
        Model(int input_qty);
        ~Model();
        void addDenseLayer(int output_qty,ActivationType typeName, std::string prefix = "");
        void addDenseLayer(int output_qty, ActivationType typeName, std::string neuron_coeff_Path, std::string bias_coeff_Path);
        std::vector<std::vector<long double>>predict(const std::vector<std::vector<long double>>&data);
        std::vector<long double> fit(int step, long double learning_rate, std::vector<std::vector<long double>> &x, std::vector<std::vector<long double>> &y, int batch_size, ErrorFunctionType loss);

    private:
        int _input_qty;
        int _output_qty;
        ErrorFunctionType _errorFunctionType;
        std::vector<Dense> _model;
        std::vector<std::vector<long double>> diff_error;

        void backward();
        long double caluculate_loss(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y);
        std::vector<std::vector<long double>>numerical_gradient_layer(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y, int index);
        std::vector<long double>numerical_gradient_bias(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y, int index);
};