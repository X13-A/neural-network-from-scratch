#include "layers/activationLayers.hpp"
#include <cmath>

// ReLULayer: max(0, x)
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

// LinearLayer: f(x) = x
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

// SoftmaxLayer: e^x_i / sum(e^x_j)
Eigen::VectorXf SoftmaxLayer::forward(const Eigen::VectorXf& input, bool cacheEnabled)
{
    // Numerical stability: subtract max before exp
    float maxInput = input.maxCoeff();
    Eigen::VectorXf exps = (input.array() - maxInput).exp();
    float sumExps = exps.sum();
    cachedOutput = exps / sumExps;
    
    if (cacheEnabled)
    {
        cachedInput = input;
        // For softmax: da_i/dz_j = softmax_i * (delta_ij - softmax_j)
        // Store softmax values for backward pass
        cachedDerivatives = cachedOutput;
    }

    return cachedOutput;
}

Eigen::VectorXf SoftmaxLayer::backward(const Eigen::VectorXf& outputGradient, float learningRate)
{
    // For softmax, compute Jacobian-vector product
    // da_i/dz_j = softmax_i * (delta_ij - softmax_j)
    Eigen::VectorXf jacobianProduct = Eigen::VectorXf::Zero(cachedDerivatives.size());
    
    for (int i = 0; i < cachedDerivatives.size(); ++i)
    {
        for (int j = 0; j < cachedDerivatives.size(); ++j)
        {
            float jacobian = cachedDerivatives[i] * (i == j ? 1.0f : 0.0f) - cachedDerivatives[i] * cachedDerivatives[j];
            jacobianProduct[i] += jacobian * outputGradient[j];
        }
    }
    
    return jacobianProduct;
}