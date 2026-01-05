#pragma once

#include "layers/layer.hpp"
#include <cmath>

class ActivationLayer : public Layer
{
protected:
    // TODO: double check caches
    Eigen::VectorXf cachedInput;
    Eigen::VectorXf cachedOutput;
    Eigen::VectorXf cachedDerivatives;

public:
    virtual ~ActivationLayer() = default;

    const Eigen::VectorXf& getOutput() const override { return cachedOutput; }
};

class ReLULayer : public ActivationLayer
{
public:
    Eigen::VectorXf forward(const Eigen::VectorXf& input, bool cacheEnabled = false) override;
    Eigen::VectorXf backward(const Eigen::VectorXf& outputGradient, float learningRate) override;
    size_t getOutputSize() const override { return cachedOutput.size(); }
};

class LinearLayer : public ActivationLayer
{
public:
    Eigen::VectorXf forward(const Eigen::VectorXf& input, bool cacheEnabled = false) override;
    Eigen::VectorXf backward(const Eigen::VectorXf& outputGradient, float learningRate) override;
    size_t getOutputSize() const override { return cachedOutput.size(); }
};

class SoftmaxLayer : public ActivationLayer
{
public:
    Eigen::VectorXf forward(const Eigen::VectorXf& input, bool cacheEnabled = false) override;
    Eigen::VectorXf backward(const Eigen::VectorXf& outputGradient, float learningRate) override;
    size_t getOutputSize() const override { return cachedOutput.size(); }
};