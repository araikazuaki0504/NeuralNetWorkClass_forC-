#include "Dense.hpp"

Dense::Dense(int input_qty, int output_qty, ActivationType typeName, std::string prefix, std::string neuron_coeff_path, std::string bias_coeff_path) :_input_qty(input_qty), _output_qty(output_qty), _typeName(typeName), _activation(typeName), _bias(output_qty), _neuron(output_qty, std::vector<long double>(input_qty)), _file_prefix(prefix), _neuron_coeff_Path(neuron_coeff_path), _bias_coeff_Path(bias_coeff_path)
{
    double sigma = 0.05;
    if(typeName == ActivationType::ReLu) sigma = std::sqrt(2.0 / (double)input_qty);
    else if(typeName == ActivationType::Sigmoid || typeName == ActivationType::SoftMax) sigma = std::sqrt(1.0 / (double)input_qty);
    else sigma = 0.05;

    std::random_device seed;
    std::mt19937 engine(seed());
    std::normal_distribution<> generator(0.0, sigma);

    if (neuron_coeff_path.empty() && bias_coeff_path.empty())
    {
        for (int i = 0; i < output_qty; i++)
        {
            _bias[i] = generator(engine);
            for (int j = 0; j < input_qty; j++)
            {
                _neuron[i][j] = generator(engine);
            }
        }
    }
    else
    {
        try
        {
            int i = 0, j = 0, k = 0;

            std::ifstream neuron_coeff_Reader(neuron_coeff_path);
            std::ifstream bias_coeff_Reader(bias_coeff_path);
            std::string coef_tmp;

            while (std::getline(neuron_coeff_Reader, coef_tmp))
            {
                _neuron[i][j] = std::stod(coef_tmp);

                i += 1;
                if (i == output_qty)
                {
                    j += 1;
                    i = 0;
                }

            }

            while (std::getline(bias_coeff_Reader, coef_tmp))
            {
                _bias[k] = std::stod(coef_tmp);
                k++;
            }

            neuron_coeff_Reader.close();
            bias_coeff_Reader.close();

        }
        catch (...)
        {
            std::cout << "ファイル読み取り不可のため、乱数で初期化しました。" << std::endl;
            for (int i = 0; i < output_qty; i++)
            {
                _bias[i] = generator(engine);
                for (int j = 0; j < input_qty; j++)
                {
                    _neuron[i][j] = generator(engine);
                }
            }
        }
    }

}

Dense::~Dense()
{
    if (_neuron_coeff_Path.empty() && _bias_coeff_Path.empty())
    {
        _neuron_coeff_Path = _file_prefix +  "Neuron_Coeffient_" + std::to_string(_input_qty) + "_" + std::to_string(_output_qty) + "_" + std::to_string(_typeName) + ".txt";
        _bias_coeff_Path = _file_prefix + "Bias_Coeffient_" + std::to_string(_input_qty) + "_" + std::to_string(_output_qty) + "_" + std::to_string(_typeName) + ".txt";

        std::ofstream neuron_coeff_Writer(_neuron_coeff_Path, std::ios::trunc);
        std::ofstream bias_coeff_Writer(_bias_coeff_Path, std::ios::trunc);

        for (size_t i = 0; i < _neuron.size(); i++)
        {
            for (size_t j = 0; j < _neuron[0].size(); j++)
            {
                if ((i == _neuron.size() - 1) && (j == _neuron[0].size() - 1)) neuron_coeff_Writer << _neuron[i][j];
                else neuron_coeff_Writer << _neuron[i][j] << std::endl;
            }
        }

        for (size_t i = 0; i < _bias.size(); i++)
        {
            if(i == _bias.size() - 1)bias_coeff_Writer << _bias[i];
            else bias_coeff_Writer << _bias[i] << std::endl;
        }

        neuron_coeff_Writer.close();
        bias_coeff_Writer.close();
    }
    else if (!_neuron_coeff_Path.empty() && !_bias_coeff_Path.empty())
    {
        std::ofstream neuron_coeff_Writer(_neuron_coeff_Path, std::ios::trunc);
        std::ofstream bias_coeff_Writer(_bias_coeff_Path, std::ios::trunc);

        for (size_t i = 0; i < _neuron.size(); i++)
        {
            for (size_t j = 0; j < _neuron[0].size(); j++)
            {
                if ((i == _neuron.size() - 1) && (j == _neuron[0].size() - 1)) neuron_coeff_Writer << _neuron[i][j];
                else neuron_coeff_Writer << _neuron[i][j] << std::endl;
            }
        }

        for (size_t i = 0; i < _bias.size(); i++)
        {
            if (i == _bias.size() - 1)bias_coeff_Writer << _bias[i];
            else bias_coeff_Writer << _bias[i] << std::endl;
        }

        neuron_coeff_Writer.close();
        bias_coeff_Writer.close();
    }
}

std::vector<std::vector<long double>> Dense::forward(std::vector<std::vector<long double>> &data)
{
    std::vector<std::vector<long double>> ans;

    _last_data = data;

    for(int index = 0; auto &i : data)
    {
        std::vector<long double> res;
        
        for(int j = 0; j < _neuron.size(); j++)
        {
            long double t = 0;

            for(int k = 0; k < _neuron[0].size(); k++)
            {
                t += i[k] * _neuron[j][k];
            }

            t -= _bias[j];

            res.push_back(t);
        }

        ans.push_back(res);

        index++;
    }

    ans = _activation.forward(ans);

    return ans;
}

std::vector<std::vector<long double>> Dense::backward(std::vector<std::vector<long double>> &data)
{
    data = _activation.backward(data);

    std::vector<std::vector<long double>> ans;

    for(int i = 0; i < data.size(); i++)
    {
        std::vector<long double>res;

        for(int j = 0; j < _neuron[0].size(); j++)
        {
            long double t=0;

            for(int k = 0; k < _neuron.size(); k++)
            {
                t += _neuron[k][j] * data[i][k];
            }

            res.push_back(t);
        }

        ans.push_back(res);
    }

    _grad_layer.clear();
    for(int i = 0; i < _neuron.size(); i++)
    {
        std::vector<long double>res;

        for(int j = 0; j < _neuron[i].size(); j++)
        {
            long double t = 0;

            for(int k = 0; k < data.size(); k++)
            {
                t += data[k][i] * _last_data[k][j];
            }

            res.push_back(t);
        }
        _grad_layer.push_back(res);
    }

    _grad_bias.clear();

    for(int i = 0; i < _bias.size(); i++)
    {
        long double t = 0;

        for(int j = 0; j < data.size(); j++)
        {
            t += data[j][i];
        }

        _grad_bias.push_back(t);
    }

    return ans;
}
