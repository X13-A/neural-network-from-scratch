#include "layers/activationLayers.hpp"
#include <cmath>

Eigen::VectorXf ReLULayer::forward(const Eigen::VectorXf& input, bool cacheEnabled)
{
    cachedOutput = input.array().max(0.0f).matrix();
    
    if (cacheEnabled)
    {
        cachedInput = input;
        cachedDerivatives = (input.array() > 0.0f).cast<float>();
    }

    return cachedOutput;
}

Eigen::VectorXf ReLULayer::backward(const Eigen::VectorXf& outputGradient, float learningRate)
{
    return outputGradient.cwiseProduct(cachedDerivatives);
}

Eigen::VectorXf LinearLayer::forward(const Eigen::VectorXf& input, bool cacheEnabled)
{
    cachedOutput = input;
    
    if (cacheEnabled)
    {
        cachedInput = input;
        cachedDerivatives = Eigen::VectorXf::Ones(input.size());
    }

    return cachedOutput;
}

Eigen::VectorXf LinearLayer::backward(const Eigen::VectorXf& outputGradient, float learningRate)
{
    return outputGradient;
}

Eigen::VectorXf SoftmaxLayer::forward(const Eigen::VectorXf& input, bool cacheEnabled)
{
    // Subtract max before exp for numerical stability
    float maxInput = input.maxCoeff();
    Eigen::VectorXf exps = (input.array() - maxInput).exp();
    float sumExps = exps.sum();
    cachedOutput = exps / sumExps;
    
    if (cacheEnabled)
    {
        cachedInput = input;
        // Here we just cache the output for use in backwards pass because the full Jacobian is memory intensive
        cachedDerivatives = cachedOutput;
    }

    return cachedOutput;
}

Eigen::VectorXf SoftmaxLayer::backward(const Eigen::VectorXf& outputGradient, float learningRate)
{
    float dotProduct = cachedDerivatives.dot(outputGradient);
    return (cachedDerivatives.array() * (outputGradient.array() - dotProduct)).matrix();
}