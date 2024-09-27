#include "ActivationFunction.hpp"

ActivationFunction::ActivationFunction(ActivationType TypeName):_TypeName(TypeName)
{

}

ActivationFunction::~ActivationFunction()
{

}

std::vector<std::vector<long double>> ActivationFunction::forward(std::vector<std::vector<long double>> &x)
{
    _last_data = x;

    if(_TypeName == ActivationType::Sigmoid)return Sigmoid(x);
    else if (_TypeName == ActivationType::ReLu)return ReLu(x);
    else if(_TypeName == ActivationType::SoftMax)return SoftMax(x);
    else return x;
}

std::vector<std::vector<long double>> ActivationFunction::Sigmoid(std::vector<std::vector<long double>> &x)
{
    std::vector<std::vector<long double>> res;

    for(auto batch : x)
    {
         std::vector<long double> t;
         for( auto i : batch)
         {
            t.push_back(1.0 / (1.0 + std::exp(-i)));
         }

         res.push_back(t);
    }

    _last_data = res;

    return res;
}

std::vector<std::vector<long double>> ActivationFunction::ReLu(std::vector<std::vector<long double>> &x)
{
    std::vector<std::vector<long double>> res;

    for(auto batch : x)
    {
         std::vector<long double> t;
         for( auto i : batch)
         {
            t.push_back((i >= 0) * i);
         }

         res.push_back(t);
    }

    return res;  
}

std::vector<std::vector<long double>> ActivationFunction::SoftMax(std::vector<std::vector<long double>> &x)
{
    std::vector<std::vector<long double>> res;

    for(auto batch : x)
    {
         std::vector<long double> t;
         long double c = *std::max_element(batch.begin(),batch.end());
         long double sum = 0;

         for( auto i : batch)
         {
            sum += std::exp(i - c);
         }

         for( auto i : batch)
         {
            t.push_back(std::exp(i - c) / sum);
         }

         res.push_back(t);
    }

    return res;
}

std::vector<std::vector<long double>> ActivationFunction::Sigmoid_back(std::vector<std::vector<long double>> &x)
{
    std::vector<std::vector<long double>>res;

    for(int i = 0; i < x.size(); i++)
    {
        std::vector<long double>t;

        for(int j = 0; j < x[i].size(); j++)
        {
            t.push_back(x[i][j] * (1 - _last_data[i][j]) * _last_data[i][j]);
        }

        res.push_back(t);
    }

    return res;
}

std::vector<std::vector<long double>> ActivationFunction::ReLu_back(std::vector<std::vector<long double>> &x)
{
    std::vector<std::vector<long double>> res = x;

    for (int i = 0; i < x.size(); ++i)
    {
        for (int j = 0; j < x[i].size(); ++j)
        {
            if (_last_data[i][j] < 0) res[i][j] = 0;
        }
    }

    return res;
}

std::vector<std::vector<long double>> ActivationFunction::SoftMax_back(std::vector<std::vector<long double>> &x)
{
    return x;
}

std::vector<std::vector<long double>> ActivationFunction::backward(std::vector<std::vector<long double>> &x)
{
    if(_TypeName == ActivationType::Sigmoid) return Sigmoid_back(x);
    else if(_TypeName == ActivationType::SoftMax) return SoftMax_back(x);
    if (_TypeName == ActivationType::ReLu) return ReLu_back(x);
    else return x;
}