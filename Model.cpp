#include "Model.hpp"

Model::Model(int input_qty):_input_qty(input_qty),_output_qty(input_qty)
{

}

Model::~Model()
{

}

void Model::addDenseLayer(int output_qty,ActivationType typeName, std::string prefix)
{
    Dense dense(_output_qty,output_qty,typeName,prefix);
    _model.push_back(dense);
    _output_qty = output_qty;
}

void Model::addDenseLayer(int output_qty, ActivationType typeName, std::string neuron_coeff_Path, std::string bias_coeff_Path)
{
    Dense dense(_output_qty, output_qty, typeName,"",neuron_coeff_Path, bias_coeff_Path);
    _model.push_back(dense);
    _output_qty = output_qty;
}

std::vector<std::vector<long double>> Model::predict(const std::vector<std::vector<long double>>&data)
{
    std::vector<std::vector<long double>>res = data;

    for(auto &layer : _model){
        res = layer.forward(res);
    }

    return res;
}


std::vector<long double> Model::fit(int step, long double learning_rate, std::vector<std::vector<long double>> &x, std::vector<std::vector<long double>> &y, int batch_size, ErrorFunctionType loss)
{

    std::vector<long double> history;
    _errorFunctionType = loss;

    for (int current_step = 0; current_step < step; current_step++)
    {
        std::vector<std::vector<long double>> batch_x, batch_y;
        diff_error.clear(); 

        for (int i = 0; i < batch_size; i++)
        {
            batch_x.push_back(x[(batch_size * current_step + i) % x.size()]);
            batch_y.push_back(y[(batch_size * current_step + i) % y.size()]);
        }

        auto test_y = predict(batch_x);
        
        for (int i = 0; i < test_y.size(); i++)
        {
            std::vector<long double> t;

            for (int j = 0; j < test_y[i].size(); j++)
            {
                t.push_back((test_y[i][j] - batch_y[i][j]) / batch_size);
            }

            diff_error.push_back(t);
        }

        backward();
        long double loss_step;

        if (_errorFunctionType == Mse)
            loss_step = ErrorFunction::mean_squared_error(test_y, y);
        else
            loss_step = ErrorFunction::mean_cross_entropy_error(test_y, y);

        for (int index = 0; auto &layer : _model)
        {
            std::vector<std::vector<long double>> layer_grad = layer._grad_layer;
            std::vector<long double> bias_grad = layer._grad_bias;
            
            for (int i = 0; i < layer_grad.size(); i++)
            {
                for (int j = 0; j < layer_grad[i].size(); j++)
                {
                    layer._neuron[i][j] -= learning_rate * layer_grad[i][j];
                }
            }
            
            for (int i = 0; i < bias_grad.size(); i++)
            {
                layer._bias[i] -= learning_rate * bias_grad[i];
            }

            index++;
        }

        history.push_back(loss_step);

    }
    
    return history;
}

void Model::backward()
{
    std::vector<std::vector<long double>> data;

    if (_errorFunctionType == ErrorFunctionType::Mse)
        data = ErrorFunction::mean_squared_error_back(diff_error);
    else
        data = ErrorFunction::mean_cross_entropy_error_back(diff_error);

    for (int i = _model.size() - 1; i >= 0; --i)
    {
        data = _model[i].backward(data);
    }
}

long double Model::caluculate_loss(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y)
{
    std::vector<std::vector<long double>>t = predict(batch_x);
    if(_errorFunctionType == Cen) return ErrorFunction::mean_cross_entropy_error(t, batch_y);
    return ErrorFunction::mean_squared_error(t, batch_y);
}

std::vector<std::vector<long double>> Model::numerical_gradient_layer(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y, int index)
{
    long double h = 1e-4;
    std::vector<std::vector<long double>> grad(_model[index]._neuron.size(), std::vector<long double>(_model[index]._neuron[0].size()));

    for(int i = 0; i < grad.size(); i++)
    {
        for(int j = 0; j < grad[i].size(); j++)
        {
            long double tmp = _model[index]._neuron[i][j];

            _model[index]._neuron[i][j] = tmp + h;
            long double fxh1 = caluculate_loss(batch_x, batch_y);

            _model[index]._neuron[i][j] = tmp - h;
            long double fxh2 = caluculate_loss(batch_x, batch_y);

            grad[i][j] = (fxh1 - fxh2) / (2 * h);
            _model[index]._neuron[i][j] = tmp;
        }
    }

    return grad;
}

std::vector<long double> Model::numerical_gradient_bias(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y, int index)
{
    long double h = 1e-4;
    std::vector<long double> grad(_model[index]._bias.size());

    for(int i = 0; i < grad.size(); i++)
    {
        long double tmp = _model[index]._bias[i];

        _model[index]._bias[i] = tmp + h;
        long double fxh1 = caluculate_loss(batch_x, batch_y);

        _model[index]._bias[i] = tmp - h;
        long double fxh2 = caluculate_loss(batch_x, batch_y);

        grad[i] = (fxh1 - fxh2) / (2 * h);
        _model[index]._bias[i] = tmp;
    }

    return grad;
}
