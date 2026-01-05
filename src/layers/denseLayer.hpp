#pragma once

#include "layers/layer.hpp"
#include "perceptron/perceptron.hpp"
#include <vector>

class DenseLayer : public Layer
{
private:
    // TODO: Optimize by using a weight matrix and bias vector
    // TODO: Double check caches
    std::vector<Perceptron> neurons;
    Eigen::VectorXf cachedInput;
    Eigen::VectorXf cachedOutput;

public:
    DenseLayer(size_t inputSize, size_t numNeurons);

    Eigen::VectorXf forward(const Eigen::VectorXf& input, bool cacheEnabled = false) override;
    Eigen::VectorXf backward(const Eigen::VectorXf& outputGradient, float learningRate) override;

    size_t getOutputSize() const override { return neurons.size(); }
    const Eigen::VectorXf& getOutput() const override { return cachedOutput; }

    // Accessors for weights and biases
    const Eigen::VectorXf& getWeights(size_t neuronIdx) const;
    float getBias(size_t neuronIdx) const;
    void setWeights(size_t neuronIdx, const Eigen::VectorXf& newWeights);
    void setBias(size_t neuronIdx, float newBias);
};
