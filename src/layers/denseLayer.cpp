#include "layers/denseLayer.hpp"

DenseLayer::DenseLayer(size_t inputSize, size_t numNeurons)
{
    for (size_t i = 0; i < numNeurons; ++i)
    {
        neurons.emplace_back(inputSize);
    }
}

Eigen::VectorXf DenseLayer::forward(const Eigen::VectorXf& input, bool cacheEnabled)
{
    cachedOutput.resize(neurons.size());
    
    if (cacheEnabled)
    {
        cachedInput = input;
    }

    for (size_t i = 0; i < neurons.size(); ++i)
    {
        float zValue;
        cachedOutput[i] = neurons[i].forward(input, zValue);
    }

    return cachedOutput;
}

Eigen::VectorXf DenseLayer::backward(const Eigen::VectorXf& outputGradient, float learningRate)
{
    size_t inputSize = neurons[0].inputSize;
    Eigen::VectorXf inputGradient = Eigen::VectorXf::Zero(inputSize);

    for (size_t j = 0; j < neurons.size(); ++j)
    {
        float dc_dz = outputGradient[j];

        // Update weights and bias for this neuron
        const Eigen::VectorXf& weights = neurons[j].getWeights();
        
        // dc/dw = dc/dz * dz/dw, where dz/dw = input
        Eigen::VectorXf dc_dw = cachedInput * dc_dz;
        
        // dc/db = dc/dz * dz/db, where dz/db = 1
        float dc_db = dc_dz;

        // Gradient descent update
        Eigen::VectorXf newWeights = weights - learningRate * dc_dw;
        neurons[j].setWeights(newWeights);
        
        float oldBias = neurons[j].getBias();
        neurons[j].setBias(oldBias - learningRate * dc_db);

        // Accumulate gradient for previous layer: dc/da_prev = dc/dz * dz/da_prev
        // dz/da_prev = weights, so dc/da_prev = dc/dz * weights
        inputGradient += weights * dc_dz;
    }

    return inputGradient;
}

const Eigen::VectorXf& DenseLayer::getWeights(size_t neuronIdx) const
{
    return neurons[neuronIdx].getWeights();
}

float DenseLayer::getBias(size_t neuronIdx) const
{
    return neurons[neuronIdx].getBias();
}

void DenseLayer::setWeights(size_t neuronIdx, const Eigen::VectorXf& newWeights)
{
    neurons[neuronIdx].setWeights(newWeights);
}

void DenseLayer::setBias(size_t neuronIdx, float newBias)
{
    neurons[neuronIdx].setBias(newBias);
}
