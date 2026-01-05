#include "mlp/mlp.hpp"
#include "layers/denseLayer.hpp"
#include <iostream>
#include <Eigen/Dense>

MLP::MLP(std::vector<std::unique_ptr<Layer>> layerConfig)
    : layers(std::move(layerConfig))
{
    if (layers.size() < 2)
    {
        throw std::invalid_argument("MLP must have at least input and output layers");
    }
}

Eigen::VectorXf MLP::forward(const Eigen::VectorXf& inputs, bool cacheEnabled)
{
    if (cacheEnabled)
    {
        layerOutputs.clear();
        layerOutputs.reserve(layers.size());
    }

    Eigen::VectorXf currentActivations = inputs;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        currentActivations = layers[i]->forward(currentActivations, cacheEnabled);
        if (cacheEnabled)
        {
            layerOutputs.push_back(currentActivations);
        }
    }

    return currentActivations;
}

void MLP::backward(const Eigen::VectorXf& expectedOutput, float learningRate, const LossFunction& lossFunc)
{
    Eigen::VectorXf output = layers.back()->getOutput();

    // Compute dc/da for output layer based on loss function
    Eigen::VectorXf dc_da = lossFunc.derivative(output, expectedOutput);

    // Backpropagate through layers from output to input
    for (int l = static_cast<int>(layers.size()) - 1; l >= 0; l--)
    {
        dc_da = layers[l]->backward(dc_da, learningRate);
    }
}