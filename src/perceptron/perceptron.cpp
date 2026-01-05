#include "perceptron.hpp"

const Eigen::VectorXf& Perceptron::getWeights() const
{ 
    return weights; 
} 

const float Perceptron::getBias() const
{ 
    return bias; 
}

void Perceptron::setWeights(const Eigen::VectorXf& newWeights)
{
    if (newWeights.size() != weights.size()) {
        throw std::invalid_argument("Weights size mismatch");
    }
    weights = newWeights;
}

void Perceptron::setBias(float newBias)
{
    bias = newBias;
}

Perceptron::Perceptron(size_t inputSize) : weights(inputSize), bias(0.0f), inputSize(inputSize)
{
    // Initialize weights and bias with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    for (size_t i = 0; i < inputSize; ++i) 
    {
        weights[i] = dis(gen);
    }
    bias = dis(gen);
}

float Perceptron::forward(const Eigen::VectorXf& inputs, float& zValue)
{
    if (inputs.size() != inputSize) {
        throw std::invalid_argument("Input size mismatch");
    }
    float sum = bias;
    for (size_t i = 0; i < inputSize; ++i) 
    {
        sum += weights[i] * inputs[i];
    }
    zValue = sum;
    return sum;
}