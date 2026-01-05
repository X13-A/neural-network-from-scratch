#pragma once

#include <Eigen/Dense>
#include <memory>

class Layer
{
public:
    virtual ~Layer() = default;

    /// @brief Forward pass through the layer
    /// @param input Input vector
    /// @param cacheEnabled Whether to cache intermediate values for backprop
    /// @return Output vector
    virtual Eigen::VectorXf forward(const Eigen::VectorXf& input, bool cacheEnabled = false) = 0;

    /// @brief Backward pass through the layer
    /// @param outputGradient Gradient from the next layer (dc/da)
    /// @param learningRate Learning rate for weight updates
    /// @return Gradient to pass to previous layer (dc/da_prev)
    virtual Eigen::VectorXf backward(const Eigen::VectorXf& outputGradient, float learningRate) = 0;

    virtual size_t getOutputSize() const = 0;

    virtual const Eigen::VectorXf& getOutput() const = 0;
};
