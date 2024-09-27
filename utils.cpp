#include "utils.hpp"

long double ErrorFunction::mean_squared_error(std::vector<std::vector<long double>> &y, std::vector<std::vector<long double>> &t)
{
    long double res = 0;

    for(int i = 0; i < y.size(); i++)
    {
        long double sum = 0;

        for(int j = 0; j < y[0].size(); j++)
        {
            sum += (y[i][j] - t[i][j]) * (y[i][j] - t[i][j]);
        }

        sum /= 2 * y.size();
        res += sum;
    }

    return res;
}

long double ErrorFunction::mean_cross_entropy_error(std::vector<std::vector<long double>> &y, std::vector<std::vector<long double>> &t)
{
    long double res = 0;

    for(int i = 0; i < y.size(); i++)
    {
        long double sum = 0;
        constexpr long double delta = std::numeric_limits<long double>::epsilon();

        for(int j = 0; j < y[0].size(); j++)
        {
            sum += t[i][j] * std::log(y[i][j] + delta);
        }

        sum *= -1;
        res += sum;
    }

    return res / y.size();
}

std::vector<std::vector<long double>> ErrorFunction::mean_squared_error_back(std::vector<std::vector<long double>>absolute_error)
{
    return absolute_error;

    std::vector<std::vector<long double>>res;

    for(int i = 0; i < absolute_error.size(); ++i)
    {
        std::vector<long double>t;

        for(int j = 0; j < absolute_error[i].size(); ++j)
        {
            t.push_back(2*abs(absolute_error[i][j]));
        }

        res.push_back(t);
    }

    return res;
}

std::vector<std::vector<long double>> ErrorFunction::mean_cross_entropy_error_back(std::vector<std::vector<long double>>error)
{
        return error;
}

std::string std::to_string(ActivationType _Val)
{
    switch (_Val)
    {
        case Sigmoid:
            return "Sigmoid";
            break;

        case Linear:
            return "Linear";
            break;

        case SoftMax:
            return "SoftMax";
            break;

        case ReLu:
            return "ReLu";
            break;
    }
}

std::string std::to_string(ErrorFunctionType _Val)
{
    switch (_Val)
    {
    case Cen:
        return "Cen";
        break;

    case Mse:
        return "Mse";
        break;
    }
}
